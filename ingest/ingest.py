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

EXTRACT_SYSTEM = f"""You extract factual statements about the user from a chat session transcript.
Respond with ONLY a JSON object of this exact shape:
{{"facts": [{{"subject": "user", "predicate": "...", "object": "<short value>", "text": "<one sentence stating the fact>"}}]}}
The predicate field MUST be exactly one of these six values: {", ".join(llm.PREDICATES)}.
Do not invent other predicates. Ignore anything in the transcript that does not clearly match one of these six concepts
(hobbies, books, weather, one-off activities, opinions, non-travel plans, etc. all get ignored).
If nothing in the transcript matches one of the six predicates, return exactly {{"facts": []}}."""

_next_id_counter = 0


def next_id() -> int:
    global _next_id_counter
    _next_id_counter += 1
    return _next_id_counter


def transcript_for(session: dict) -> str:
    lines = [f"{m['role']}: {m['text']}" for m in session["messages"]]
    return "Transcript:\n" + "\n".join(lines)


def extract_facts(session: dict) -> list[dict]:
    result = llm.chat_json(EXTRACT_SYSTEM, transcript_for(session))
    return result.get("facts", [])


def get_current_fact(subject: str, predicate: str) -> Optional[dict]:
    rows = hc.run(
        "MATCH (f:Fact {subject: $s, predicate: $p, current: true}) RETURN f.id AS id, f.object AS object",
        s=subject, p=predicate,
    )
    return rows[0] if rows else None


def ingest() -> None:
    with open(DATA_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    fact_count = 0
    supersede_count = 0

    for session in data["sessions"]:
        facts = extract_facts(session)
        if not facts:
            print(f"{session['id']}: no facts extracted")
            continue

        sid = next_id()
        print(f"{session['id']}:")

        for fact in facts:
            subject = fact.get("subject", "user")
            predicate = fact.get("predicate")
            obj = fact.get("object")
            text = fact.get("text", f"{subject} {predicate} {obj}")

            if predicate not in llm.PREDICATES or not obj:
                print(f"  skip malformed/unknown-predicate fact: {fact}")
                continue

            existing = get_current_fact(subject, predicate)
            if existing and existing["object"] == obj:
                print(f"  skip duplicate restatement: {subject}.{predicate} = {obj}")
                continue

            fid = next_id()
            hc.run(
                "MERGE (f:Fact {id: $fid, key: $fkey, subject: $s, predicate: $p, object: $o, text: $t, timestamp: $ts, current: true})"
                "-[:STATED_IN]->(sess:Session {id: $sid, key: $skey, index: $idx, date: $date})",
                fid=fid, fkey=f"fact-{fid}", s=subject, p=predicate, o=obj, t=text, ts=session["date"],
                sid=sid, skey=session["id"], idx=session["index"], date=session["date"],
            )
            fact_count += 1

            if existing:
                hc.run(
                    "MERGE (f:Fact {id: $fid})-[:SUPERSEDES]->(old:Fact {id: $oldId})",
                    fid=fid, oldId=existing["id"],
                )
                hc.run("MATCH (old:Fact {id: $oldId}) SET old.current = false", oldId=existing["id"])
                supersede_count += 1
                print(f"  {subject}.{predicate}: {existing['object']} -> {obj} (superseded)")
            else:
                print(f"  {subject}.{predicate} = {obj} (new)")

    print(f"\nDone. {fact_count} facts written, {supersede_count} supersessions.")


if __name__ == "__main__":
    ingest()
