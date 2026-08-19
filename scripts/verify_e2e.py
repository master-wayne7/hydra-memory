"""
End-to-end smoke test for HydraMemory.

Checks, in order:
  1. HydraDB HTTP API is reachable and answers a query.
  2. The /ask API is reachable.
  3. Reset database and run clean ingestion.
  4. A fixed set of questions returns the expected answer/abstention, proving
     supersession and abstention both work against whatever is currently in the graph.
  5. Idempotency of ingestion (re-running ingest.py produces no duplicates).
  6. Invariant check (no single-valued predicate has >1 current=true fact).
  7. Verification of the temporal history chain and provenance in response.
  8. Robustness of malformed queries to /ask.
"""

import os
import sys
import subprocess

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra_client as hc

API_URL = "http://127.0.0.1:5000/ask"

# (question, expect_found, substring expected in the answer if found, expected source session)
CASES = [
    ("Where does the user currently live?", True, "Pune", "sess-07"),
    ("Where does the user work?", True, "Zomato", "sess-04"),
    ("What is the user's favorite food?", True, "Dosa", "sess-05"),
    ("What are the user's travel plans?", True, "Manali", "sess-06"),
    ("What is the user's pet's name?", True, "Rocky", "sess-01"),
    ("What is the user's favorite color?", False, None, None),
    ("Where was the user born?", False, None, None),
]


def check_hydradb() -> bool:
    print("[1/8] HydraDB reachable...", end=" ")
    try:
        hc.run("MATCH (n {id: 1}) RETURN n.id")
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        print("       Is the container up? Try: docker compose up -d")
        return False


def check_api() -> bool:
    print("[2/8] /ask API reachable...", end=" ")
    try:
        requests.post(API_URL, json={"question": "ping"}, timeout=10)
        print("OK")
        return True
    except requests.exceptions.ConnectionError:
        print("FAIL")
        print("       Is the API running? Try: python api/app.py")
        return False


def reset_db() -> bool:
    print("[3/8] Resetting database and loading clean dataset...", end=" ")
    try:
        # Clear database
        hc.run("MATCH (f:Fact) DETACH DELETE f")
        hc.run("MATCH (s:Session) DETACH DELETE s")
        # Run clean ingestion
        subprocess.check_call([sys.executable, "ingest/ingest.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def run_cases() -> bool:
    print("[4/8] Question/answer cases:")
    all_passed = True
    for question, expect_found, expect_substr, expect_session in CASES:
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=30).json()
        except Exception as e:
            print(f"  FAIL  {question!r} -> request error: {e}")
            all_passed = False
            continue

        found = resp.get("found")
        answer = resp.get("answer", "")
        sources = resp.get("source_sessions", [])

        ok = found == expect_found
        if expect_found:
            ok = ok and expect_substr.lower() in answer.lower() and expect_session in sources

        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {question!r} -> found={found}, answer={answer!r}, sources={sources}")
        all_passed = all_passed and ok

    return all_passed


def check_idempotency() -> bool:
    print("[5/8] Verifying Ingestion Idempotency...", end=" ")
    
    # 1. Capture initial node counts
    try:
        sess_before = len(hc.run("MATCH (s:Session) RETURN s.id"))
        facts_before = len(hc.run("MATCH (f:Fact) RETURN f.id"))
    except Exception as e:
        print(f"FAIL to query initial state: {e}")
        return False

    # 2. Run ingestion again
    try:
        subprocess.check_call([sys.executable, "ingest/ingest.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"FAIL to run ingestion: {e}")
        return False

    # 3. Capture post-ingestion counts
    try:
        sess_after = len(hc.run("MATCH (s:Session) RETURN s.id"))
        facts_after = len(hc.run("MATCH (f:Fact) RETURN f.id"))
    except Exception as e:
        print(f"FAIL to query final state: {e}")
        return False

    # 4. Assert no duplicates were created
    if sess_before != sess_after or facts_before != facts_after:
        print(f"FAIL\n       Duplicates created!")
        print(f"       Sessions: {sess_before} -> {sess_after}")
        print(f"       Facts: {facts_before} -> {facts_after}")
        return False

    print("OK (perfectly idempotent)")
    return True


def check_cardinality_invariants() -> bool:
    print("[6/8] Verifying Cardinality Invariants...", end=" ")
    
    predicates = ["lives_in", "works_at", "job_title", "favorite_food", "travel_plan", "pet_name"]
    for pred in predicates:
        rows = hc.run(
            "MATCH (f:Fact {subject: 'user', predicate: $p, current: true}) RETURN f.id AS id",
            p=pred
        )
        count = len(rows)
        if count > 1:
            print(f"FAIL (Predicate '{pred}' has {count} current facts, expected <= 1)")
            return False
            
    print("OK")
    return True


def check_history_and_provenance() -> bool:
    print("[7/8] Verifying Temporal History and Provenance...", end=" ")
    
    try:
        resp = requests.post(API_URL, json={"question": "Where does the user currently live?"}, timeout=10).json()
        memory = resp.get("memory")
        if not memory:
            print("FAIL (No 'memory' block returned)")
            return False

        # Validate current state
        curr = memory.get("current")
        if not curr or curr.get("value") != "Pune" or curr.get("session") != "sess-07":
            print(f"FAIL (Incorrect current fact value: {curr})")
            return False

        # Validate history chain
        history = memory.get("history")
        if not history or len(history) < 2:
            print(f"FAIL (Empty or incomplete history chain: {history})")
            return False

        # Check chronological order: Delhi -> Bangalore -> Pune
        values = [h["value"] for h in history]
        expected_values = ["Delhi", "Bangalore", "Pune"]
        val_idx = 0
        for val in values:
            if val == expected_values[val_idx]:
                val_idx += 1
                if val_idx == len(expected_values):
                    break
        if val_idx < len(expected_values):
            print(f"FAIL (History chain not chronologically ordered. Got: {values}, Expected: {expected_values})")
            return False

        # Verify semantic normalization variants map to the same lives_in history
        variants = [
            "Where does the user live?",
            "Where does the user reside?",
            "What is the user's current city?"
        ]
        for q in variants:
            r = requests.post(API_URL, json={"question": q}, timeout=10).json()
            if not r.get("found") or r.get("memory", {}).get("current", {}).get("value") != "Pune":
                print(f"FAIL (Variant '{q}' failed semantic normalization mapping)")
                return False

        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def check_malformed_requests() -> bool:
    print("[8/8] Verifying Robustness against malformed inputs...", end=" ")
    try:
        # 1. Missing question
        r1 = requests.post(API_URL, json={}).json()
        if r1.get("status") != "invalid":
            print("FAIL (Expected invalid status for missing question)")
            return False

        # 2. Empty question
        r2 = requests.post(API_URL, json={"question": "   "}).json()
        if r2.get("status") != "invalid":
            print("FAIL (Expected invalid status for empty question)")
            return False

        # 3. Non-string question
        r3 = requests.post(API_URL, json={"question": 123}).json()
        if r3.get("status") != "invalid":
            print("FAIL (Expected invalid status for non-string question)")
            return False

        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def check_same_value_provenance() -> bool:
    print("[9/11] Verifying Same-Value Provenance (Rocky in sess-01 and sess-08)...", end=" ")
    try:
        rows = hc.run(
            "MATCH (f:Fact {predicate: 'pet_name', object: 'Rocky'})-[:STATED_IN]->(s:Session) "
            "RETURN s.key AS key"
        )
        keys = {r["key"] for r in rows}
        if "sess-01" not in keys or "sess-08" not in keys:
            print(f"FAIL (Stating sessions found: {keys}, expected both sess-01 and sess-08)")
            return False
        
        # Test API returns all source sessions for Rocky
        resp = requests.post(API_URL, json={"question": "What is the user's pet's name?"}, timeout=10).json()
        sources = resp.get("source_sessions", [])
        if "sess-01" not in sources or "sess-08" not in sources:
            print(f"FAIL (API source_sessions: {sources}, expected both sess-01 and sess-08)")
            return False
            
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def check_self_healing() -> bool:
    print("[10/11] Verifying Self-Healing/Recovery of current fact invariant...", end=" ")
    try:
        # Corrupt DB: insert an artificial duplicate current fact for job_title
        # First find sess-04
        sess_rows = hc.run("MATCH (s:Session {key: 'sess-04'}) RETURN s.id AS id")
        if not sess_rows:
            print("FAIL (sess-04 not found)")
            return False
        sid = sess_rows[0]["id"]
        
        # Create fact 9999
        hc.run(
            "CREATE (f:Fact {id: 9999, key: 'fact-sess-04-job_title-corrupted', subject: 'user', predicate: 'job_title', object: 'Tech Lead', current: true})-[:STATED_IN]->(s:Session {id: $sid})",
            sid=sid
        )
        
        # Call get_current_fact for job_title to trigger self-healing repair
        from ingest.ingest import get_current_fact
        healed_fact = get_current_fact("user", "job_title")
        
        # Verify only 1 is current now
        rows = hc.run("MATCH (f:Fact {subject: 'user', predicate: 'job_title', current: true}) RETURN f.id AS id, f.object AS object")
        if len(rows) != 1:
            print(f"FAIL (Found {len(rows)} current facts after repair: {[r['object'] for r in rows]})")
            return False
            
        # Clean up fact 9999 if it remains
        hc.run("MATCH (f:Fact {id: 9999}) DETACH DELETE f")
        
        print("OK (successfully self-healed)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def check_third_party_filtering() -> bool:
    print("[11/11] Verifying Third-Party statement filtering (works_at = Google skipped)...", end=" ")
    try:
        rows = hc.run("MATCH (f:Fact {object: 'Google'}) RETURN f.id AS id")
        if rows:
            print(f"FAIL (Third-party fact matching 'Google' was written! IDs: {[r['id'] for r in rows]})")
            return False
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


if __name__ == "__main__":
    ok = check_hydradb()
    ok = check_api() and ok
    if ok:
        ok = reset_db() and ok
        ok = run_cases() and ok
        ok = check_idempotency() and ok
        ok = check_cardinality_invariants() and ok
        ok = check_history_and_provenance() and ok
        ok = check_malformed_requests() and ok
        ok = check_same_value_provenance() and ok
        ok = check_self_healing() and ok
        ok = check_third_party_filtering() and ok

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
