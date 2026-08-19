"""Thin wrapper around Groq's OpenAI-compatible chat API. Shared by ingestion and the API."""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set — cannot authenticate against the Gemini endpoint.")

_client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# _client = OpenAI(
#     api_key=api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

MODEL = "gemini-3.6-flash"

# Fixed predicate vocabulary shared by extraction (ingest) and question parsing (api).
# Keeping this closed rather than open-vocabulary is deliberate: without it, extraction and
# question-parsing could phrase the same concept differently (e.g. "lives_in" vs "current_city"),
# causing a false abstention that looks like a bug rather than genuine "not in memory."
PREDICATES = ["lives_in", "works_at", "job_title", "pet_name", "favorite_food", "travel_plan"]

PREDICATE_CONFIG = {
    "lives_in": {"cardinality": "single"},
    "works_at": {"cardinality": "single"},
    "job_title": {"cardinality": "single"},
    "favorite_food": {"cardinality": "single"},
    "travel_plan": {"cardinality": "single"},
    "pet_name": {"cardinality": "single"},
}

NORMALIZATION_MAP = {
    # lives_in
    "lives_in": "lives_in",
    "resides_in": "lives_in",
    "based_in": "lives_in",
    "current_city": "lives_in",
    "home_city": "lives_in",
    "lives": "lives_in",
    "resides": "lives_in",
    # works_at
    "works_at": "works_at",
    "works_for": "works_at",
    "employed_by": "works_at",
    "employer": "works_at",
    # job_title
    "job_title": "job_title",
    "role": "job_title",
    "title": "job_title",
    "occupation": "job_title",
    # pet_name
    "pet_name": "pet_name",
    "pet": "pet_name",
    # favorite_food
    "favorite_food": "favorite_food",
    "fav_food": "favorite_food",
    # travel_plan
    "travel_plan": "travel_plan",
    "travel": "travel_plan",
}

def normalize_predicate(pred: str) -> str:
    if not pred:
        return ""
    cleaned = pred.strip().lower()
    return NORMALIZATION_MAP.get(cleaned, cleaned)

_MAX_ATTEMPTS = 6


def _with_retry(fn):
    last_err = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - transient network/rate-limit errors, retried below
            last_err = e
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
    raise last_err


def chat_json(system: str, user: str) -> dict:
    def call():
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)

    return _with_retry(call)


def chat_text(system: str, user: str) -> str:
    def call():
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    return _with_retry(call)
