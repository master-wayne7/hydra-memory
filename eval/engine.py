"""
Shared LongMemEval ingestion/answering/grading logic, used by both the CLI batch runner
(run_eval.py) and the API's live "ask a LongMemEval instance" endpoints (api/app.py).

See run_eval.py's module docstring for why this uses an open predicate vocabulary and
per-instance id offsets, instead of ingest/ingest.py + api/app.py's fixed 6-value demo scheme.
"""

import json
import os
import sys
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra_client as hc  # noqa: E402
import llm_client as llm  # noqa: E402

SUBSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "subset.json")

ID_OFFSET_PER_INSTANCE = 100_000
ID_BASE = 1_000_000

EXTRACT_SYSTEM = """You extract factual statements about the user from what the user explicitly stated in a conversation.
Respond with ONLY a JSON object:
{"facts": [{"subject": "user", "predicate": "snake_case_topic", "object": "exact value", "text": "one clear factual sentence", "cumulative": false}]}

Rules:
1. Extract ONLY concrete facts the user explicitly stated about themselves, their pets, friends, or activities (e.g. personal best run times, friends relocating, count of restaurants tried, mortgage preapproval amounts).
2. Do NOT extract questions, requests for tips/advice, or general discussion.
3. NEVER use angle brackets like <...> or placeholder words in object or text.
4. Predicate must be a short snake_case label (2-3 words, e.g. "charity_5k_time", "rachel_location", "korean_restaurants_count", "mortgage_preapproval_amount").
5. Set cumulative to true ONLY if it is an item in an open-ended list being collected; set to false if it is a single current value or updated state.
6. If no personal facts were stated by the user, return {"facts": []}.

Example 1:
User: I ran a charity 5K run and set a personal best time of 27:12.
Output:
{"facts": [{"subject": "user", "predicate": "charity_5k_time", "object": "27:12", "text": "The user's personal best time in the charity 5K run is 27:12.", "cumulative": false}]}

Example 2:
User: My friend Rachel actually just moved back to the suburbs again.
Output:
{"facts": [{"subject": "rachel", "predicate": "rachel_location", "object": "the suburbs", "text": "Rachel moved to the suburbs.", "cumulative": false}]}

Example 3:
User: I got pre-approved for $400,000 from Wells Fargo.
Output:
{"facts": [{"subject": "user", "predicate": "mortgage_preapproval_amount", "object": "$400,000", "text": "The user was pre-approved for $400,000 mortgage from Wells Fargo.", "cumulative": false}]}
"""

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "predicate": {"type": "string", "minLength": 1, "maxLength": 50},
                    "object": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "cumulative": {"type": "boolean"},
                },
                "required": ["subject", "predicate", "object", "text", "cumulative"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

ANSWER_SYSTEM = """You answer a question using ONLY the provided candidate facts.
Respond with ONLY a JSON object:
{"found": true, "answer": "concise factual answer"}
or if the candidate facts do not contain the answer:
{"found": false, "answer": "not found"}

Rules:
- State the answer directly and concisely (e.g. "25 minutes and 50 seconds", "four", "the suburbs", "$400,000").
- NEVER use angle brackets <...> or placeholder text.
- Do NOT guess or use outside knowledge.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string", "minLength": 1},
    },
    "required": ["found", "answer"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You grade whether a candidate answer conveys the same factual information as a reference answer, for a given question.
Respond with ONLY JSON: {"correct": true or false}
Be lenient about phrasing/formatting differences (e.g. "25:50" vs "25 minutes 50 seconds" both
count as correct) but strict about factual content -- the candidate must convey the same specific
fact as the reference, not merely a related or plausible-sounding one."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
    "additionalProperties": False,
}

# (instance_index, session_index, session_total, note) -> None
ProgressCallback = Optional[Callable[[int, int, str], None]]


def load_subset() -> list:
    with open(SUBSET_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def category_for(instance: dict) -> str:
    return "abstention" if instance["question_id"].endswith("_abs") else instance["question_type"]


def transcript_for(session: list) -> str:
    # Only user turns contain ground-truth facts stated by the user about their life.
    # Filtering out long assistant advice prevents distractor fact explosion and speeds up processing.
    user_turns = [f"User: {turn['content'].strip()}" for turn in session if turn.get("role") == "user" and turn.get("content", "").strip()]
    if not user_turns:
        return ""
    return "\n".join(user_turns)


MAX_TRANSCRIPT_CHARS = 10_000


def _split_text(text: str, max_len: int) -> list:
    return [text[i:i + max_len] for i in range(0, len(text), max_len)] or [text]


FAST_MODEL = "qwen2.5:3b-instruct"
FALLBACK_MODEL = "qwen2.5:7b-instruct"


def extract_facts(session: list) -> list:
    full_text = transcript_for(session)
    if not full_text:
        return []
    chunks = _split_text(full_text, MAX_TRANSCRIPT_CHARS) if len(full_text) > MAX_TRANSCRIPT_CHARS else [full_text]

    facts = []
    for chunk in chunks:
        try:
            llm.MODEL = FAST_MODEL
            result = llm.chat_json(EXTRACT_SYSTEM, chunk, schema=FACTS_SCHEMA)
            facts.extend(result.get("facts", []))
        except Exception as e:
            print(f"  [WARNING] {FAST_MODEL} failed for a chunk, retrying with {FALLBACK_MODEL}: {type(e).__name__}: {e}")
            try:
                llm.MODEL = FALLBACK_MODEL
                result = llm.chat_json(EXTRACT_SYSTEM, chunk, schema=FACTS_SCHEMA)
                facts.extend(result.get("facts", []))
            except Exception as e2:
                print(f"  [WARNING] {FALLBACK_MODEL} also failed for this chunk: {type(e2).__name__}: {e2}")
        finally:
            llm.MODEL = FAST_MODEL
    return facts


def _normalize_pred(p: str) -> str:
    p = p.lower().strip()
    for prefix in ("current_", "previous_", "user_", "recent_", "the_"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p


def get_current_fact(instance_id: int, subject: str, predicate: str) -> Optional[dict]:
    # 1. Exact match on subject + predicate
    rows = hc.run(
        "MATCH (f:Fact {instance_id: $iid, subject: $s, predicate: $p, current: true}) "
        "RETURN f.id AS id, f.object AS object, f.predicate AS predicate",
        iid=instance_id, s=subject, p=predicate,
    )
    if rows:
        return rows[0]

    # 2. Fuzzy match on subject + stripped predicate
    rows = hc.run(
        "MATCH (f:Fact {instance_id: $iid, subject: $s, current: true}) "
        "RETURN f.id AS id, f.object AS object, f.predicate AS predicate",
        iid=instance_id, s=subject,
    )
    norm_p = _normalize_pred(predicate).replace("_", "")
    for r in rows:
        norm_r = _normalize_pred(r["predicate"]).replace("_", "")
        if norm_p == norm_r or (len(norm_p) >= 4 and len(norm_r) >= 4 and (norm_p in norm_r or norm_r in norm_p)):
            return r
    return None


def _cumulative_fact_exists(instance_id: int, subject: str, predicate: str, obj: str) -> bool:
    rows = hc.run(
        "MATCH (f:Fact {instance_id: $iid, subject: $s, predicate: $p, object: $o, current: true}) "
        "RETURN f.id AS id LIMIT 1",
        iid=instance_id, s=subject, p=predicate, o=obj,
    )
    return bool(rows)


def is_ingested(instance_id: int) -> bool:
    rows = hc.run(
        "MATCH (f:Fact {instance_id: $iid}) RETURN f.id AS id",
        iid=instance_id,
    )
    return bool(rows)


def ingest_instance(instance: dict, instance_id: int, on_progress: ProgressCallback = None) -> dict:
    """Ingest one instance's full haystack into HydraDB. Returns {"fact_count", "supersede_count"}."""
    id_counter = [ID_BASE + instance_id * ID_OFFSET_PER_INSTANCE]

    def next_id() -> int:
        id_counter[0] += 1
        return id_counter[0]

    sessions = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]
    session_dates = instance["haystack_dates"]
    total = len(sessions)

    fact_count = 0
    supersede_count = 0

    for idx, session in enumerate(sessions):
        if on_progress:
            on_progress(idx, total, f"extracting session {session_ids[idx]}")

        facts = extract_facts(session)
        if not facts:
            continue

        sid = next_id()

        for fact in facts:
            subject = fact.get("subject", "user")
            predicate = fact.get("predicate")
            obj = fact.get("object")
            text = fact.get("text", f"{subject} {predicate} {obj}")
            cumulative = bool(fact.get("cumulative", False))
            if not predicate or not obj or obj in ("<short value>", "<none>", "-"):
                continue

            # Strip angle brackets if any slipped through
            if obj.startswith("<") and obj.endswith(">"):
                obj = obj[1:-1].strip()
            if text.startswith("<") and text.endswith(">"):
                text = text[1:-1].strip()

            if cumulative:
                if _cumulative_fact_exists(instance_id, subject, predicate, obj):
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

    if on_progress:
        on_progress(total, total, "done")

    return {"fact_count": fact_count, "supersede_count": supersede_count}


def get_candidates(instance_id: int) -> list:
    return hc.run(
        "MATCH (f:Fact {instance_id: $iid, current: true})-[:STATED_IN]->(sess:Session) "
        "RETURN f.predicate AS predicate, f.text AS text, f.object AS object, "
        "sess.key AS session_key, sess.date AS date",
        iid=instance_id,
    )


def _filter_candidates_for_question(candidates: list, question: str, max_candidates: int = 15) -> list:
    if len(candidates) <= max_candidates:
        return candidates
    
    # Extract substantive keywords from question
    import re
    stop_words = {"what", "when", "where", "who", "why", "how", "did", "was", "is", "are", "the", "a", "an", "in", "on", "at", "to", "for", "of", "my", "i", "me", "you", "your", "do", "does", "have", "had", "been", "there", "this", "that"}
    tokens = set(re.findall(r"\w+", question.lower())) - stop_words
    
    def score(c: dict) -> int:
        c_text = f"{c.get('predicate', '')} {c.get('text', '')} {c.get('object', '')}".lower()
        c_tokens = set(re.findall(r"\w+", c_text))
        return len(tokens & c_tokens)

    scored = [(score(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # If the top item has keyword matches, return relevant ones up to max_candidates
    if scored and scored[0][0] > 0:
        relevant = [c for s, c in scored if s > 0]
        return relevant[:max_candidates]
    return candidates[:max_candidates]


def answer_question(instance_id: int, question: str) -> dict:
    candidates = get_candidates(instance_id)
    if not candidates:
        return {"found": False, "answer": "not found", "n_candidates": 0, "candidates": []}

    ranked_candidates = _filter_candidates_for_question(candidates, question, max_candidates=15)
    candidate_list = "\n".join(f"- ({c['predicate']}) {c['text']}" for c in ranked_candidates)
    user_msg = f"Question: {question}\nCandidate facts:\n{candidate_list}"
    try:
        llm.MODEL = FAST_MODEL
        result = llm.chat_json(ANSWER_SYSTEM, user_msg, schema=ANSWER_SCHEMA, max_tokens=100)
    except Exception as e:
        print(f"  [WARNING] {FAST_MODEL} failed to answer, retrying with {FALLBACK_MODEL}: {type(e).__name__}: {e}")
        llm.MODEL = FALLBACK_MODEL
        result = llm.chat_json(ANSWER_SYSTEM, user_msg, schema=ANSWER_SCHEMA, max_tokens=100)
    finally:
        llm.MODEL = FAST_MODEL
    result["n_candidates"] = len(candidates)
    result["candidates"] = candidates
    return result


def judge(question: str, reference: str, candidate: str) -> bool:
    user_msg = f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}"
    try:
        llm.MODEL = FAST_MODEL
        result = llm.chat_json(JUDGE_SYSTEM, user_msg, schema=JUDGE_SCHEMA, max_tokens=20)
    except Exception as e:
        print(f"  [WARNING] {FAST_MODEL} failed to judge, retrying with {FALLBACK_MODEL}: {type(e).__name__}: {e}")
        llm.MODEL = FALLBACK_MODEL
        result = llm.chat_json(JUDGE_SYSTEM, user_msg, schema=JUDGE_SCHEMA, max_tokens=20)
    finally:
        llm.MODEL = FAST_MODEL
    return bool(result.get("correct"))


def grade(instance: dict, response: dict) -> bool:
    is_abstention = instance["question_id"].endswith("_abs")
    if is_abstention:
        return response["found"] is False
    if response["found"] is False:
        return False
    return judge(instance["question"], instance["answer"], response["answer"])
