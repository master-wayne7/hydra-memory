"""
Ingest each subset instance's full haystack (up to ~50 real sessions) into HydraDB with
open-vocabulary fact extraction, answer its question via a graph-curated candidate set, and grade
the result -- deterministically for abstention questions, via an LLM judge for knowledge-update
questions. Prints per-instance progress and a final accuracy table; writes results.json.

The actual ingest/answer/grade logic lives in engine.py, shared with the API's live
"ask a LongMemEval instance" endpoints (api/app.py) so the CLI batch run and the interactive
UI path can't drift apart.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def ingest_instance_verbose(instance: dict, instance_id: int) -> None:
    sessions = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]

    def on_progress(idx: int, total: int, note: str) -> None:
        if idx < total:
            print(f"    session {idx + 1}/{total} ({session_ids[idx]})...", end="\r")

    result = engine.ingest_instance(instance, instance_id, on_progress=on_progress)
    print(f"    -> {result['fact_count']} facts written, {result['supersede_count']} supersessions" + " " * 20)


def run() -> None:
    subset = engine.load_subset()

    results = []
    for instance_id, instance in enumerate(subset):
        category = engine.category_for(instance)
        print(f"[{instance_id + 1}/{len(subset)}] {instance['question_id']} ({category})")
        print(f"  question: {instance['question']!r}")
        print(f"  reference answer: {instance['answer']!r}")

        try:
            if engine.is_ingested(instance_id):
                print("  (already ingested -- skipping extraction, reusing existing facts)")
            else:
                ingest_instance_verbose(instance, instance_id)
            response = engine.answer_question(instance_id, instance["question"])
            print(f"  our answer: found={response['found']} answer={response['answer']!r} (from {response['n_candidates']} candidates)")
            passed = engine.grade(instance, response)
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
