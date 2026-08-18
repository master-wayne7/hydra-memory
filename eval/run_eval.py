"""
Ingest each subset instance's full haystack (up to ~50 real sessions) into HydraDB with
open-vocabulary fact extraction, answer its question via a graph-curated candidate set, and grade
the result -- deterministically for abstention questions, via an LLM judge for knowledge-update
questions. Prints per-instance progress and a final accuracy table; writes results.json.

Differs from ingest/ingest.py + api/app.py deliberately (see plan Part 2 / README):
  - open predicate vocabulary instead of the fixed 6-value enum (real LongMemEval questions are
    open-domain; forcing them into the demo's closed vocabulary would just misfile everything)
  - retrieval fetches ALL current facts for the instance as a small candidate set and lets one
    LLM call pick/synthesize or decline, rather than requiring an exact predicate-string match
    (open vocabulary can't reliably round-trip through an exact match the way the closed demo
    vocabulary does)
  - every node is tagged with an integer `instance_id`, and ids start at a large per-instance
    offset, so the 8 instances (and the unrelated Part 1 demo data already in the same graph)
    can't collide or cross-contaminate
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra_client as hc  # noqa: E402
import llm_client as llm  # noqa: E402

SUBSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "subset.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

ID_OFFSET_PER_INSTANCE = 100_000
ID_BASE = 1_000_000

EXTRACT_SYSTEM = """You extract durable factual statements about the user from a chat session transcript.
Respond with ONLY a JSON object of this exact shape:
{"facts": [{"subject": "user", "predicate": "...", "object": "<short value>", "text": "<one sentence stating the fact>"}]}
The predicate must be a short, specific, descriptive snake_case label for the underlying topic,
with no "current_"/"previous_"/"old_"/"new_" qualifier in it (e.g. favorite_hiking_trail, not
current_favorite_hiking_trail) -- invent one that fits the fact precisely, and reuse the exact
same predicate whenever the same real-world topic reappears later with an updated value.
If a single message mentions both an old and a new value for the same topic (e.g. "I used to
prefer X but now I prefer Y"), extract only ONE fact for that topic, with the new/current value
as the object -- never split it into two facts like "previous_x" and "current_x".
Only extract durable facts the user states about themselves or people/things they mention -- not
the assistant's suggestions, generic advice, or one-off remarks with no lasting factual content.
If the transcript has no such fact, return exactly {"facts": []}."""

ANSWER_SYSTEM = """You answer a question using ONLY the given list of candidate facts about the user.
If one of the candidate facts answers the question, respond with ONLY JSON:
{"found": true, "answer": "<concise answer using only that fact's information>"}
If none of the candidate facts answer the question, respond with ONLY JSON:
{"found": false, "answer": "not found"}
Do not use any information beyond what's in the candidate facts. Do not guess."""

JUDGE_SYSTEM = """You grade whether a candidate answer conveys the same factual information as a reference answer, for a given question.
Respond with ONLY JSON: {"correct": true or false}
Be lenient about phrasing/formatting differences (e.g. "25:50" vs "25 minutes 50 seconds" both
count as correct) but strict about factual content -- the candidate must convey the same specific
fact as the reference, not merely a related or plausible-sounding one."""


def transcript_for(session: list) -> str:
    lines = [f"{turn['role']}: {turn['content']}" for turn in session]
    return "Transcript:\n" + "\n".join(lines)


def extract_facts(session: list) -> list:
    result = llm.chat_json(EXTRACT_SYSTEM, transcript_for(session))
    return result.get("facts", [])


def get_current_fact(instance_id: int, subject: str, predicate: str):
    rows = hc.run(
        "MATCH (f:Fact {instance_id: $iid, subject: $s, predicate: $p, current: true}) "
        "RETURN f.id AS id, f.object AS object",
        iid=instance_id, s=subject, p=predicate,
    )
    return rows[0] if rows else None


def ingest_instance(instance: dict, instance_id: int) -> None:
    id_counter = [ID_BASE + instance_id * ID_OFFSET_PER_INSTANCE]

    def next_id() -> int:
        id_counter[0] += 1
        return id_counter[0]

    sessions = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]
    session_dates = instance["haystack_dates"]

    fact_count = 0
    supersede_count = 0

    for idx, session in enumerate(sessions):
        facts = extract_facts(session)
        print(f"    session {idx + 1}/{len(sessions)} ({session_ids[idx]}): {len(facts)} fact(s)", end="")
        if not facts:
            print()
            continue

        sid = next_id()
        events = []

        for fact in facts:
            subject = fact.get("subject", "user")
            predicate = fact.get("predicate")
            obj = fact.get("object")
            text = fact.get("text", f"{subject} {predicate} {obj}")
            if not predicate or not obj:
                continue

            existing = get_current_fact(instance_id, subject, predicate)
            if existing and existing["object"] == obj:
                continue

            fid = next_id()
            hc.run(
                "MERGE (f:Fact {id: $fid, instance_id: $iid, subject: $s, predicate: $p, object: $o, "
                "text: $t, current: true})-[:STATED_IN]->"
                "(sess:Session {id: $sid, instance_id: $iid, key: $skey, date: $date})",
                fid=fid, iid=instance_id, s=subject, p=predicate, o=obj, t=text,
                sid=sid, skey=session_ids[idx], date=session_dates[idx],
            )
            fact_count += 1

            if existing:
                hc.run(
                    "MERGE (f:Fact {id: $fid})-[:SUPERSEDES]->(old:Fact {id: $oldId})",
                    fid=fid, oldId=existing["id"],
                )
                hc.run("MATCH (old:Fact {id: $oldId}) SET old.current = false", oldId=existing["id"])
                supersede_count += 1
                events.append(f"{predicate}: {existing['object']} -> {obj}")
            else:
                events.append(f"{predicate} = {obj}")

        print("  " + "; ".join(events) if events else "")

    print(f"    -> {fact_count} facts written, {supersede_count} supersessions")


def answer_question(instance_id: int, question: str) -> dict:
    candidates = hc.run(
        "MATCH (f:Fact {instance_id: $iid, current: true}) "
        "RETURN f.predicate AS predicate, f.text AS text",
        iid=instance_id,
    )
    if not candidates:
        return {"found": False, "answer": "not found", "n_candidates": 0}

    candidate_list = "\n".join(f"- ({c['predicate']}) {c['text']}" for c in candidates)
    result = llm.chat_json(ANSWER_SYSTEM, f"Question: {question}\nCandidate facts:\n{candidate_list}")
    result["n_candidates"] = len(candidates)
    return result


def judge(question: str, reference: str, candidate: str) -> bool:
    result = llm.chat_json(
        JUDGE_SYSTEM,
        f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}",
    )
    return bool(result.get("correct"))


def grade(instance: dict, response: dict) -> bool:
    is_abstention = instance["question_id"].endswith("_abs")
    if is_abstention:
        return response["found"] is False
    if response["found"] is False:
        return False
    return judge(instance["question"], instance["answer"], response["answer"])


def run() -> None:
    with open(SUBSET_PATH, encoding="utf-8") as fh:
        subset = json.load(fh)

    results = []
    for instance_id, instance in enumerate(subset):
        is_abstention = instance["question_id"].endswith("_abs")
        category = "abstention" if is_abstention else instance["question_type"]
        print(f"[{instance_id + 1}/{len(subset)}] {instance['question_id']} ({category})")
        print(f"  question: {instance['question']!r}")
        print(f"  reference answer: {instance['answer']!r}")

        try:
            ingest_instance(instance, instance_id)
            response = answer_question(instance_id, instance["question"])
            print(f"  our answer: found={response['found']} answer={response['answer']!r} (from {response['n_candidates']} candidates)")
            passed = grade(instance, response)
            print(f"  {'PASS' if passed else 'FAIL'}")
            result = {
                "question_id": instance["question_id"],
                "category": category,
                "question": instance["question"],
                "reference_answer": instance["answer"],
                "our_answer": response["answer"],
                "our_found": response["found"],
                "n_candidates": response["n_candidates"],
                "passed": passed,
            }
        except Exception as e:  # noqa: BLE001 - one instance's failure shouldn't lose the rest
            print(f"  ERROR: {type(e).__name__}: {e}")
            result = {
                "question_id": instance["question_id"],
                "category": category,
                "question": instance["question"],
                "reference_answer": instance["answer"],
                "error": str(e),
                "passed": False,
            }
        print()

        results.append(result)
        # write after every instance so a later failure doesn't lose earlier results
        with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    total = len(results)
    correct = sum(1 for r in results if r["passed"])
    print("=" * 60)
    print(f"{'question_id':20s} {'category':16s} {'result'}")
    for r in results:
        print(f"{r['question_id']:20s} {r['category']:16s} {'PASS' if r['passed'] else 'FAIL'}")
    print("=" * 60)
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.0f}%)")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
