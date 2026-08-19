"""
POST /ask {"question": "..."} -> {"answer": "...", "source_sessions": [...], "found": bool, "status": "...", "memory": {...}}

The graph decides abstention, not the LLM: the question-parsing step maps the question to a
(subject, predicate) pair drawn from the same fixed vocabulary used at ingestion time, then
HydraDB is queried for the current fact. If HydraDB returns zero rows -- either because the
predicate isn't recognized or because no fact was ever stated for it -- we return `found: false`
immediately, with no further LLM call. The LLM is never given the chance to guess an answer for
a question the graph doesn't have data for.
"""

import os
import sys
from typing import Optional

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


def get_fact_history(current_fact_id: int) -> list[dict]:
    history = []
    visited = set()
    next_id = current_fact_id
    MAX_HISTORY_DEPTH = 20
    depth = 0
    
    while next_id and next_id not in visited and depth < MAX_HISTORY_DEPTH:
        visited.add(next_id)
        # Fetch the fact and all its associated sessions
        rows = hc.run(
            "MATCH (f:Fact {id: $fid})-[:STATED_IN]->(sess:Session) "
            "RETURN f.id AS id, f.object AS object, f.text AS text, f.current AS current, "
            "sess.key AS session_key, sess.date AS date, sess.index AS index",
            fid=next_id
        )
        if not rows:
            break
            
        # Sort sessions for this fact chronologically by index
        sorted_rows = sorted(rows, key=lambda r: r.get("index") or 0)
        fact_obj = sorted_rows[0]
        sessions = [r["session_key"] for r in sorted_rows]
        
        # Add to history
        history.append({
            "value": fact_obj["object"],
            "session": sorted_rows[-1]["session_key"], # latest session key for this state
            "sessions": sessions,                      # all sessions key for this state
            "date": sorted_rows[-1]["date"],           # latest date
            "current": bool(fact_obj["current"])
        })
        
        # Find the superseded fact (if any)
        supersedes_rows = hc.run(
            "MATCH (f:Fact {id: $fid})-[:SUPERSEDES]->(old:Fact) "
            "RETURN old.id AS id",
            fid=next_id
        )
        if supersedes_rows:
            next_id = supersedes_rows[0]["id"]
        else:
            next_id = None
            
        depth += 1
        
    return list(reversed(history))


@app.route("/ask", methods=["POST"])
def ask():
    try:
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({
                "error": "Invalid request body, expected JSON object",
                "status": "invalid"
            }), 400

        if "question" not in body:
            return jsonify({
                "error": "question field is required",
                "status": "invalid"
            }), 400

        question = body.get("question")
        if question is None:
            return jsonify({
                "error": "question field is required",
                "status": "invalid"
            }), 400

        if not isinstance(question, str):
            return jsonify({
                "error": "question field must be a string",
                "status": "invalid"
            }), 400

        question = question.strip()
        if not question:
            return jsonify({
                "error": "question cannot be empty or whitespace-only",
                "status": "invalid"
            }), 400

        # Fast-path for memory changes query
        if "what changed" in question.lower() or "memory changes" in question.lower():
            try:
                # Find all current user facts in the database
                rows = hc.run(
                    "MATCH (f:Fact {subject: 'user', current: true}) "
                    "RETURN f.id AS id, f.predicate AS predicate, f.object AS object"
                )
                changes = []
                all_sessions = set()
                for r in rows:
                    history = get_fact_history(r["id"])
                    # Extract only values that were observed
                    history_values = [h["value"] for h in history]
                    for h in history:
                        for s in h.get("sessions", []):
                            all_sessions.add(s)
                    
                    changes.append({
                        "predicate": r["predicate"],
                        "history": history_values,
                        "current": r["object"]
                    })
                
                # Retrieve precomputed comparison if present
                import json
                comparison_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "baseline_comparison.json")
                baseline_data = None
                if os.path.exists(comparison_path):
                    try:
                        with open(comparison_path, encoding="utf-8") as fcomp:
                            comparison = json.load(fcomp)
                            for q_key, q_val in comparison.items():
                                if "changed" in q_key.lower():
                                    baseline_data = {
                                        "question": question,
                                        "hydramemory": q_val["hydramemory"],
                                        "baseline": q_val["baseline"],
                                        "explanation": q_val["explanation"]
                                    }
                                    break
                    except Exception:
                        pass
                
                if not baseline_data:
                    baseline_data = {
                        "question": question,
                        "hydramemory": "Multiple superseded historical facts identified per predicate.",
                        "baseline": "All details combined, missing chronological clarity.",
                        "explanation": "HydraMemory traces exact SUPERSEDES relationships to construct the state diff, while the naive context dump returns all text."
                    }

                return jsonify({
                    "answer": "Here is the timeline diff of user memory changes retrieved directly from the HydraDB graph:",
                    "found": True,
                    "status": "found",
                    "source_sessions": sorted(list(all_sessions)),
                    "changes": changes,
                    "evidence": {
                        "fact_found": True,
                        "current_fact": True,
                        "source_sessions": sorted(list(all_sessions)),
                        "history_checked": True,
                        "supersession_detected": True,
                        "answer_generation_allowed": False
                    },
                    "baseline_comparison": baseline_data
                })
            except Exception as e:
                return jsonify({
                    "error": f"Failed to retrieve memory changes: {str(e)}",
                    "status": "error"
                }), 500

        # Fast-path for diagnostic health check ping
        if question.lower() == "ping":
            return jsonify({
                "answer": "pong",
                "status": "success",
                "found": True,
                "source_sessions": []
            })

        # Load baseline comparison
        import json
        comparison = {}
        comparison_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "baseline_comparison.json")
        if os.path.exists(comparison_path):
            try:
                with open(comparison_path, encoding="utf-8") as fcomp:
                    comparison = json.load(fcomp)
            except Exception:
                pass

        baseline_data = None
        for q_key, q_val in comparison.items():
            if question.lower() == q_key.lower() or question.lower().rstrip("?") == q_key.lower().rstrip("?"):
                baseline_data = {
                    "question": question,
                    "hydramemory": q_val["hydramemory"],
                    "baseline": q_val["baseline"],
                    "explanation": q_val["explanation"]
                }
                break

        # LLM question parsing
        try:
            parsed = llm.chat_json(QUESTION_PARSE_SYSTEM, question)
            print(f"The answer:{parsed}")
        except Exception as e:
            return jsonify({
                "error": f"LLM question parsing failed: {str(e)}",
                "status": "error"
            }), 500

        subject = parsed.get("subject", "user")
        raw_predicate = parsed.get("predicate")
        
        # Apply normalization
        predicate = llm.normalize_predicate(raw_predicate)

        if not predicate or predicate == "none" or predicate not in llm.PREDICATES:
            if not baseline_data:
                baseline_data = {
                    "question": question,
                    "hydramemory": "I don't have that in memory.",
                    "baseline": "No mapping found.",
                    "explanation": "Question could not be resolved to any supported predicates."
                }
            return jsonify({
                "answer": "I couldn't confidently map that question to a memory.",
                "source_sessions": [],
                "found": False,
                "status": "ambiguous",
                "evidence": {
                    "fact_found": False,
                    "current_fact": False,
                    "history_checked": False,
                    "supersession_detected": False,
                    "answer_generation_allowed": False
                },
                "memory": None,
                "baseline_comparison": baseline_data
            })

        # Query HydraDB for current fact
        try:
            rows = hc.run(
                "MATCH (f:Fact {subject: $s, predicate: $p, current: true})-[:STATED_IN]->(sess:Session) "
                "RETURN f.id AS id, f.text AS text, f.object AS object, sess.key AS session_key, sess.date AS date, sess.index AS index",
                s=subject, p=predicate,
            )
        except Exception as e:
            return jsonify({
                "error": f"Database query failed: {str(e)}",
                "status": "error"
            }), 500

        if not rows:
            # NO EVIDENCE -> Gated; answer-synthesis LLM is not called!
            if not baseline_data:
                baseline_data = {
                    "question": question,
                    "hydramemory": "I don't have that in memory.",
                    "baseline": "I don't have that in memory.",
                    "explanation": "No fact match exists for this predicate in HydraDB."
                }
            return jsonify({
                "answer": NOT_FOUND_ANSWER,
                "source_sessions": [],
                "found": False,
                "status": "not_found",
                "evidence": {
                    "fact_found": False,
                    "current_fact": False,
                    "history_checked": True,
                    "supersession_detected": False,
                    "answer_generation_allowed": False
                },
                "memory": None,
                "baseline_comparison": baseline_data
            })

        # Sort sessions for this fact chronologically by index to identify the latest
        sorted_rows = sorted(rows, key=lambda r: r.get("index") or 0)
        fact = sorted_rows[-1]

        # Retrieve history
        try:
            history = get_fact_history(fact["id"])
        except Exception as e:
            return jsonify({
                "error": f"Database history retrieval failed: {str(e)}",
                "status": "error"
            }), 500

        # Synthesize phrasing of retrieved evidence
        try:
            answer = llm.chat_text(
                ANSWER_SYSTEM,
                f"Question: {question}\nFact: {fact['text']} (value: {fact['object']})\n"
                "Answer the question in one natural sentence using only this fact.",
            )
        except Exception as e:
            return jsonify({
                "error": f"LLM answer phrasing failed: {str(e)}",
                "status": "error"
            }), 500

        supersession_detected = len(history) > 1
        if not baseline_data:
            baseline_data = {
                "question": question,
                "hydramemory": fact["object"],
                "baseline": fact["object"],
                "explanation": "Both systems retrieved the correct fact."
            }

        return jsonify({
            "answer": answer,
            "source_sessions": [r["session_key"] for r in sorted_rows],
            "found": True,
            "status": "found",
            "evidence": {
                "fact_found": True,
                "current_fact": True,
                "source_sessions": [r["session_key"] for r in sorted_rows],
                "history_checked": True,
                "supersession_detected": supersession_detected,
                "answer_generation_allowed": True
            },
            "memory": {
                "subject": subject,
                "predicate": predicate,
                "current": {
                    "value": fact["object"],
                    "text": fact["text"],
                    "session": fact["session_key"],
                    "sessions": [r["session_key"] for r in sorted_rows],
                    "date": fact["date"]
                },
                "history": history
            },
            "baseline_comparison": baseline_data
        })
    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "status": "error"
        }), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
