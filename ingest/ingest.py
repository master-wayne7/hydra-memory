"""
sessions.json -> Groq fact extraction -> HydraDB writes with supersession detection.

For each session (processed in chronological order), extract (subject, predicate, object)
triples constrained to a fixed predicate vocabulary. For each triple, query HydraDB for the
existing *current* fact with the same (subject, predicate):
  - none exists            -> write the new fact as current
  - exists, same object    -> skip (duplicate restatement, nothing new)
  - exists, different obj  -> write the new fact as current, link it -[:SUPERSEDES]-> the old
                               fact, and flip the old fact's `current` flag to false

The supersession decision is driven by a real query against the live graph (not by remembering
state in this script), which is what makes the graph traversal the source of truth, not the
ingestion code.
"""

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra_client as hc
import llm_client as llm

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions.json")

EXTRACT_SYSTEM = f"""You extract factual statements about the user from what the user explicitly stated in a chat session.
Respond with ONLY a JSON object:
{{"facts": [{{"subject": "user", "predicate": "...", "object": "exact value", "text": "one clear factual sentence"}}]}}
The predicate field MUST be exactly one of these six values: {", ".join(llm.PREDICATES)}.
Do not invent other predicates. Ignore anything that does not clearly match one of these six concepts.

Rules:
- NEVER use angle brackets like <...> or placeholder words.
- Distinguish between facts about the user (subject "user") and other people (use their relationship, e.g. "brother").
- Only extract current facts. Do NOT extract historical facts ("I used to live in...") as current.
- If nothing in the transcript matches, return {{"facts": []}}.

Example:
User: I moved to Pune. I am now working at Zomato as a Senior Software Engineer.
Output:
{{"facts": [
  {{"subject": "user", "predicate": "lives_in", "object": "Pune", "text": "The user lives in Pune."}},
  {{"subject": "user", "predicate": "works_at", "object": "Zomato", "text": "The user works at Zomato."}},
  {{"subject": "user", "predicate": "job_title", "object": "Senior Software Engineer", "text": "The user is a Senior Software Engineer."}}
]}}"""

_next_id_counter = 0


def init_id_counter() -> None:
    global _next_id_counter
    try:
        rows = hc.run("MATCH (n) RETURN max(n.id) as max_id")
        if rows and rows[0]["max_id"] is not None:
            _next_id_counter = int(rows[0]["max_id"])
            return
    except Exception:
        pass
    _next_id_counter = 0


def next_id() -> int:
    global _next_id_counter
    _next_id_counter += 1
    return _next_id_counter


def transcript_for(session: dict) -> str:
    user_lines = [f"User: {m['text']}" for m in session["messages"] if m.get("role") == "user"]
    return "\n".join(user_lines)



def extract_facts(session: dict) -> list[dict]:
    try:
        result = llm.chat_json(EXTRACT_SYSTEM, transcript_for(session), schema=llm.FACTS_SCHEMA)
        if not isinstance(result, dict):
            print(f"  [WARNING] Expected JSON dict from LLM, got: {type(result)}")
            return []
        return result.get("facts", [])
    except Exception as e:
        print(f"  [ERROR] Fact extraction failed for session {session.get('id')}: {e}")
        return []


def get_current_fact(subject: str, predicate: str) -> Optional[dict]:
    # Query all facts matching subject, predicate, current=true to catch multiple current facts
    rows = hc.run(
        "MATCH (f:Fact {subject: $s, predicate: $p, current: true})-[:STATED_IN]->(sess:Session) "
        "RETURN f.id AS id, f.object AS object, sess.index AS index, f.timestamp AS timestamp",
        s=subject, p=predicate,
    )
    if not rows:
        # Fallback if somehow there's a fact without a session relation (to be safe)
        rows = hc.run(
            "MATCH (f:Fact {subject: $s, predicate: $p, current: true}) RETURN f.id AS id, f.object AS object",
            s=subject, p=predicate,
        )
        return rows[0] if rows else None

    # A single current fact can have multiple STATED_IN edges (one per session that
    # restated the same value via provenance merging) -- that's not a duplicate-fact
    # condition, so group rows by fact id before deciding whether repair is needed.
    by_id: dict = {}
    for row in rows:
        fid = row["id"]
        best = by_id.get(fid)
        if best is None or (row.get("index") or 0) > (best.get("index") or 0):
            by_id[fid] = row
    facts = list(by_id.values())

    if len(facts) == 1:
        return facts[0]

    # Multiple distinct current facts! Self-healing/repair logic.
    print(f"  [WARNING] Multiple current facts found for {subject}.{predicate}! Repairing...")
    # Sort descending by session index, then timestamp (or ID as final tie breaker)
    sorted_facts = sorted(
        facts,
        key=lambda r: (r.get("index") or 0, r.get("timestamp") or "", r["id"]),
        reverse=True
    )
    newest = sorted_facts[0]

    # Mark others as current = false
    for row in sorted_facts[1:]:
        print(f"    healing: marking fact id {row['id']} (val: {row['object']}) as current=false")
        hc.run("MATCH (f:Fact {id: $fid}) SET f.current = false", fid=row["id"])

    return newest


def get_session_by_key(skey: str) -> Optional[dict]:
    rows = hc.run("MATCH (s:Session {key: $key}) RETURN s.id AS id, s.key AS key", key=skey)
    return rows[0] if rows else None


def get_fact_by_key(fkey: str) -> Optional[dict]:
    rows = hc.run("MATCH (f:Fact {key: $key}) RETURN f.id AS id, f.key AS key, f.current AS current, f.object AS object", key=fkey)
    return rows[0] if rows else None


def ingest() -> None:
    init_id_counter()

    with open(DATA_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    # Sort sessions by index/date to guarantee chronological order
    sessions = sorted(data["sessions"], key=lambda s: s.get("index", 0))

    fact_count = 0
    supersede_count = 0

    session_keys_to_ids = {}

    for session in sessions:
        skey = session["id"]
        
        existing_sess = get_session_by_key(skey)
        if existing_sess:
            sid = existing_sess["id"]
            session_keys_to_ids[skey] = sid
        else:
            if skey not in session_keys_to_ids:
                session_keys_to_ids[skey] = next_id()
            sid = session_keys_to_ids[skey]

        facts = extract_facts(session)
        if not facts:
            print(f"{skey}: no facts extracted")
            continue

        print(f"{skey}:")

        for fact in facts:
            if not isinstance(fact, dict):
                print(f"  skip malformed fact node: {fact}")
                continue

            subject = fact.get("subject", "user")
            raw_predicate = fact.get("predicate")
            obj = str(fact.get("object", "")).strip()
            if obj.startswith("<") and obj.endswith(">"):
                obj = obj[1:-1].strip()
            text = str(fact.get("text", f"{subject} {raw_predicate} {obj}")).strip()
            if text.startswith("<") and text.endswith(">"):
                text = text[1:-1].strip()


            # Apply semantic normalization
            predicate = llm.normalize_predicate(raw_predicate)

            if predicate not in llm.PREDICATES or not obj:
                print(f"  skip malformed/unknown-predicate fact: {fact}")
                continue

            # Subject policy: ensure third party facts are not attributed to user
            if subject != "user":
                print(f"  skip non-user subject fact: {fact}")
                continue

            # Ensure logical fact key is stable and unique to session + normalized predicate
            fkey = f"fact-{skey}-{predicate}"

            existing_fact = get_fact_by_key(fkey)
            if existing_fact:
                print(f"  skip existing fact: {subject}.{predicate} = {obj}")
                continue

            # For single-valued predicates, find if there is a current fact in the database
            existing_curr = get_current_fact(subject, predicate)
            if existing_curr and existing_curr["object"] == obj:
                # Same value! Skip creating a new Fact node. Merge the STATED_IN relationship
                # to the current Session node to record provenance.
                hc.run(
                    "MERGE (f:Fact {id: $fid})-[:STATED_IN]->(sess:Session {id: $sid, key: $skey, index: $idx, date: $date})",
                    fid=existing_curr["id"], sid=sid, skey=skey, idx=session["index"], date=session["date"]
                )
                print(f"  skip duplicate restatement: {subject}.{predicate} = {obj} (merged provenance)")
                continue

            fid = next_id()
            
            # Create fact and relate to session in a single relationship MERGE
            hc.run(
                "MERGE (f:Fact {id: $fid, key: $fkey, subject: $s, predicate: $p, object: $o, text: $t, timestamp: $ts, current: true})"
                "-[:STATED_IN]->(sess:Session {id: $sid, key: $skey, index: $idx, date: $date})",
                fid=fid, fkey=fkey, s=subject, p=predicate, o=obj, t=text, ts=session["date"],
                sid=sid, skey=skey, idx=session["index"], date=session["date"]
            )
            fact_count += 1

            if existing_curr:
                hc.run(
                    "MERGE (f:Fact {id: $fid})-[:SUPERSEDES]->(old:Fact {id: $oldId})",
                    fid=fid, oldId=existing_curr["id"]
                )
                hc.run("MATCH (old:Fact {id: $oldId}) SET old.current = false", oldId=existing_curr["id"])
                supersede_count += 1
                print(f"  {subject}.{predicate}: {existing_curr['object']} -> {obj} (superseded)")
            else:
                print(f"  {subject}.{predicate} = {obj} (new)")

    print(f"\nDone. {fact_count} facts written, {supersede_count} supersessions.")


if __name__ == "__main__":
    ingest()
