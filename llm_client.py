"""Thin wrapper around Groq's OpenAI-compatible chat API. Shared by ingestion and the API."""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

MODEL = "openai/gpt-oss-120b"

# Fixed predicate vocabulary shared by extraction (ingest) and question parsing (api).
# Keeping this closed rather than open-vocabulary is deliberate: without it, extraction and
# question-parsing could phrase the same concept differently (e.g. "lives_in" vs "current_city"),
# causing a false abstention that looks like a bug rather than genuine "not in memory."
PREDICATES = ["lives_in", "works_at", "job_title", "pet_name", "favorite_food", "travel_plan"]

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
