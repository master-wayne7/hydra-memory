"""
End-to-end smoke test for HydraMemory.

Checks, in order:
  1. HydraDB HTTP API is reachable and answers a query.
  2. The /ask API is reachable.
  3. A fixed set of questions returns the expected answer/abstention, proving
     supersession and abstention both work against whatever is currently in the graph.

Run after `docker compose up -d`, `python ingest/ingest.py`, and `python api/app.py`
are all up. Exits non-zero if anything fails, so it can double as a CI-style gate.
"""

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra_client as hc  # noqa: E402  (after sys.path fix)

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
    print("[1/3] HydraDB reachable...", end=" ")
    try:
        hc.run("MATCH (n {id: 1}) RETURN n.id")
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        print("       Is the container up? Try: docker compose up -d")
        return False


def check_api() -> bool:
    print("[2/3] /ask API reachable...", end=" ")
    try:
        requests.post(API_URL, json={"question": "ping"}, timeout=10)
        print("OK")
        return True
    except requests.exceptions.ConnectionError:
        print("FAIL")
        print("       Is the API running? Try: python api/app.py")
        return False


def run_cases() -> bool:
    print("[3/3] Question/answer cases:")
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


if __name__ == "__main__":
    ok = check_hydradb()
    ok = check_api() and ok
    if ok:
        ok = run_cases() and ok

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
