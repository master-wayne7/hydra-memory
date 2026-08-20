"""Thin wrapper around an OpenAI-compatible chat API. Shared by ingestion and the API.

LLM_PROVIDER selects which backend to use ("groq", "gemini", or "ollama"; default "groq") --
all three are reached through the same `openai` SDK, just with a different base_url/api
key/model, since all three expose an OpenAI-compatible chat completions endpoint. "ollama"
talks to a local Ollama server (no API key, no rate limits) -- see README for setup.
"""

import json
import os
import re
import time
from typing import Optional

import openai
import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        # Proactively space calls out rather than only reacting to 429s after the fact -- observed
        # behavior was ~3 fast calls before the first rate limit, so this stays under that pace.
        "min_interval_s": 2.0,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-3.6-flash",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model": "gpt-oss-120b",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key_env": None,  # local server, no key required
        "model": os.environ.get("LLM_MODEL", "qwen2.5:3b-instruct"),
        # CPU-offloaded local inference can run as slow as ~3-4 tokens/sec on modest hardware,
        # so a cloud-tuned 60s timeout can fire mid-generation on longer sessions and force a
        # retry of the same slow request. Local runs aren't rate-limited or metered, so a long
        # timeout costs nothing but wall-clock time.
        "timeout_s": 300.0,
    },

}

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
if PROVIDER not in _PROVIDERS:
    raise RuntimeError(f"Unknown LLM_PROVIDER {PROVIDER!r} -- expected one of {sorted(_PROVIDERS)}.")

_provider_config = _PROVIDERS[PROVIDER]
if _provider_config["api_key_env"] is None:
    api_key = "ollama"  # OpenAI SDK requires a non-empty string even when unused
else:
    api_key = os.environ.get(_provider_config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{_provider_config['api_key_env']} is not set -- required for LLM_PROVIDER={PROVIDER!r}.")

_REQUEST_TIMEOUT_S = _provider_config.get("timeout_s", 60.0)

# The SDK's defaults (600s timeout, its own internal max_retries=2) let a single stalled
# request block silently for up to ~20-30 minutes before _with_retry ever sees an exception.
# A short timeout plus max_retries=0 means stalls surface quickly and _with_retry (which
# knows about rate limits and non-retryable errors) is the only retry logic in play.
_client = OpenAI(api_key=api_key, base_url=_provider_config["base_url"], timeout=_REQUEST_TIMEOUT_S, max_retries=0)

MODEL = _provider_config["model"]

# Fixed predicate vocabulary shared by extraction (ingest) and question parsing (api).
# Keeping this closed rather than open-vocabulary is deliberate: without it, extraction and
# question-parsing could phrase the same concept differently (e.g. "lives_in" vs "current_city"),
# causing a false abstention that looks like a bug rather than genuine "not in memory."
PREDICATES = ["lives_in", "works_at", "job_title", "pet_name", "favorite_food", "travel_plan"]

# Schema for the fact-extraction contract (see EXTRACT_SYSTEM in ingest.py / eval/engine.py).
# Passed to chat_json's `schema` param to constrain decoding rather than just hoping the model
# free-forms its way to this shape.
FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string", "enum": PREDICATES},
                    "object": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["subject", "predicate", "object", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

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
_MAX_RATE_LIMIT_RETRIES = 6
_MAX_RATE_LIMIT_WAIT_S = 30 * 60  # safety cap on a single wait, in case of a malformed value

# Groq's 429 body reads e.g. "Please try again in 11m32.064s." -- pull the wait time out of it.
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _retry_after_seconds(exc: "openai.RateLimitError") -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        header_val = response.headers.get("retry-after")
        if header_val:
            try:
                return float(header_val)
            except ValueError:
                pass
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return 30.0  # unparseable -- conservative fixed wait rather than failing outright


_MIN_CALL_INTERVAL_S = _provider_config.get("min_interval_s", 0.0)
_last_call_at = [0.0]


def _throttle() -> None:
    if _MIN_CALL_INTERVAL_S <= 0:
        return
    wait = _MIN_CALL_INTERVAL_S - (time.time() - _last_call_at[0])
    if wait > 0:
        time.sleep(wait)
    _last_call_at[0] = time.time()


def _with_retry(fn):
    last_err = None
    rate_limit_retries = 0
    attempt = 0
    while attempt < _MAX_ATTEMPTS:
        _throttle()
        try:
            return fn()
        except openai.RateLimitError as e:
            rate_limit_retries += 1
            last_err = e
            if rate_limit_retries > _MAX_RATE_LIMIT_RETRIES:
                raise
            wait_s = min(_retry_after_seconds(e) + 2.0, _MAX_RATE_LIMIT_WAIT_S)
            print(f"  [rate limit] waiting {wait_s:.0f}s before retrying...", flush=True)
            time.sleep(wait_s)
            # Doesn't consume the generic attempt budget -- a rate limit isn't a flaky failure.
        except openai.APIStatusError as e:
            # A 4xx other than 429 (bad request, request too large, auth, etc.) means the
            # request itself is invalid -- retrying the identical request can't ever succeed.
            if e.status_code < 500:
                raise
            last_err = e
            attempt += 1
            if attempt < _MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
        except Exception as e:  # noqa: BLE001 - other transient errors, retried below
            last_err = e
            attempt += 1
            if attempt < _MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
    raise last_err


def chat_json(system: str, user: str, schema: Optional[dict] = None, max_tokens: int = 1024) -> dict:
    """schema, if given, constrains the response via grammar-based structured output rather than
    just requesting "some valid JSON" -- this is what stops a small model from e.g. returning the
    right JSON syntax but the wrong shape (missing keys, a string where an object was required).
    Ollama enforces this at decode time via its native /api/chat `format` field; other providers
    get the closest OpenAI-compatible equivalent (response_format=json_schema) on a best-effort
    basis since that path isn't exercised by this project's default provider (groq).

    max_tokens should be sized to what the call actually needs -- a large default masks a real
    failure mode where a big candidate list in the prompt (e.g. answering from 80+ facts) can push
    the model into repeating/enumerating content instead of producing the short JSON answer, and
    a generous cap just means burning the full budget every retry instead of failing fast.
    """
    if schema is not None and PROVIDER == "ollama":
        def call():
            resp = requests.post(
                f"{_provider_config['base_url'].rstrip('/').removesuffix('/v1')}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "format": schema,
                    # num_predict bounds a pathological case where temp=0 decoding gets stuck in
                    # a repetition loop (e.g. emitting near-duplicate array entries indefinitely)
                    # -- schema constrains *shape* per-token but not overall output length, so
                    # without a cap a stuck generation just runs until the request timeout.
                    # repeat_penalty makes entering that loop less likely in the first place.
                    "options": {"temperature": 0, "num_predict": max_tokens, "repeat_penalty": 1.05},
                    "stream": False,
                },
                timeout=_REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            return json.loads(resp.json()["message"]["content"])

        return _with_retry(call)

    def call():
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        else:
            response_format = {"type": "json_object"}
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_format,
            temperature=0,
            max_tokens=max_tokens,
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
