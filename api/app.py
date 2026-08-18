"""
POST /ask {"question": "..."} -> {"answer": "...", "source_sessions": [...], "found": bool}

The graph decides abstention, not the LLM: the question-parsing step maps the question to a
(subject, predicate) pair drawn from the same fixed vocabulary used at ingestion time, then
HydraDB is queried for the current fact. If HydraDB returns zero rows -- either because the
predicate isn't recognized or because no fact was ever stated for it -- we return `found: false`
immediately, with no further LLM call. The LLM is never given the chance to guess an answer for
a question the graph doesn't have data for.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request

import hydra_client as hc
import llm_client as llm

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/ask", methods=["OPTIONS"])
def ask_preflight():
    return "", 204

QUESTION_PARSE_SYSTEM = f"""You map a question about a user's remembered facts to a (subject, predicate) pair.
Respond with ONLY JSON: {{"subject": "user", "predicate": "..."}}
The predicate field MUST be exactly one of these six values: {", ".join(llm.PREDICATES)},
OR the literal string "none" if the question does not clearly match any of them."""

ANSWER_SYSTEM = "You answer a question using ONLY the single fact given to you. Respond with one concise, natural sentence. Do not add any information beyond what's given."

NOT_FOUND_ANSWER = "I don't have that in memory."


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True, silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    parsed = llm.chat_json(QUESTION_PARSE_SYSTEM, question)
    subject = parsed.get("subject", "user")
    predicate = parsed.get("predicate")

    if predicate not in llm.PREDICATES:
        return jsonify({"answer": NOT_FOUND_ANSWER, "source_sessions": [], "found": False})

    rows = hc.run(
        "MATCH (f:Fact {subject: $s, predicate: $p, current: true})-[:STATED_IN]->(sess:Session) "
        "RETURN f.text AS text, f.object AS object, sess.key AS session_key, sess.date AS date",
        s=subject, p=predicate,
    )
    if not rows:
        return jsonify({"answer": NOT_FOUND_ANSWER, "source_sessions": [], "found": False})

    fact = rows[0]
    answer = llm.chat_text(
        ANSWER_SYSTEM,
        f"Question: {question}\nFact: {fact['text']} (value: {fact['object']})\n"
        "Answer the question in one natural sentence using only this fact.",
    )
    return jsonify({
        "answer": answer,
        "source_sessions": [fact["session_key"]],
        "found": True,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
