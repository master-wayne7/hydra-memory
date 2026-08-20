import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_client as llm
from api.app import app


class TestUnit(unittest.TestCase):
    def setUp(self):
        self.app_client = app.test_client()

    def test_normalize_predicate(self):
        # 1. Test lives_in variants
        self.assertEqual(llm.normalize_predicate("lives_in"), "lives_in")
        self.assertEqual(llm.normalize_predicate("resides_in"), "lives_in")
        self.assertEqual(llm.normalize_predicate("based_in"), "lives_in")
        self.assertEqual(llm.normalize_predicate("current_city"), "lives_in")
        
        # 2. Test works_at variants
        self.assertEqual(llm.normalize_predicate("works_at"), "works_at")
        self.assertEqual(llm.normalize_predicate("works_for"), "works_at")
        self.assertEqual(llm.normalize_predicate("employed_by"), "works_at")
        self.assertEqual(llm.normalize_predicate("employer"), "works_at")
        
        # 3. Test job_title variants
        self.assertEqual(llm.normalize_predicate("job_title"), "job_title")
        self.assertEqual(llm.normalize_predicate("role"), "job_title")
        
        # 4. Test unknown / malformed values
        self.assertEqual(llm.normalize_predicate(""), "")
        self.assertEqual(llm.normalize_predicate("something_else"), "something_else")

    def test_predicate_config(self):
        self.assertEqual(llm.PREDICATE_CONFIG["lives_in"]["cardinality"], "single")
        self.assertEqual(llm.PREDICATE_CONFIG["works_at"]["cardinality"], "single")

    def test_ask_validation_empty_body(self):
        resp = self.app_client.post("/ask", data="not json", content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json)

    def test_ask_validation_missing_question(self):
        resp = self.app_client.post("/ask", json={})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json["status"], "invalid")

    def test_ask_validation_invalid_question_type(self):
        resp = self.app_client.post("/ask", json={"question": 123})
        self.assertEqual(resp.status_code, 400)

        resp = self.app_client.post("/ask", json={"question": ""})
        self.assertEqual(resp.status_code, 400)

        resp = self.app_client.post("/ask", json={"question": "   "})
        self.assertEqual(resp.status_code, 400)

    @patch("api.app.hc.run")
    @patch("api.app.llm.chat_json")
    @patch("api.app.llm.chat_text")
    def test_strict_gating_abstention(self, mock_chat_text, mock_chat_json, mock_hc_run):
        # Setup: LLM parses question successfully, but database query returns empty (no evidence)
        mock_chat_json.return_value = {"subject": "user", "predicate": "lives_in"}
        mock_hc_run.return_value = [] # no rows found
        
        resp = self.app_client.post("/ask", json={"question": "Where does the user live?"})
        
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json["found"])
        self.assertEqual(resp.json["status"], "not_found")
        self.assertEqual(resp.json["answer"], "I don't have that in memory.")
        
        # Verify strict separation: LLM chat_text (answer phrasing) must NOT have been called!
        mock_chat_text.assert_not_called()

    @patch("api.app.hc.run")
    def test_get_fact_history_cycle_safety(self, mock_hc_run):
        from api.app import get_fact_history
        # Setup mock queries to return a cyclic chain: Fact 1 -> Fact 2 -> Fact 1
        # When called for fact 1, return session key 'sess-01'
        # Then next MATCH (supersedes) query returns old fact 2
        # When called for fact 2, return session key 'sess-02'
        # Then next MATCH (supersedes) query returns old fact 1
        def mock_run_side_effect(query, **kwargs):
            if "STATED_IN" in query:
                fid = kwargs.get("fid")
                if fid == 1:
                    return [{"id": 1, "object": "Pune", "text": "lives in Pune", "current": True, "session_key": "sess-01", "date": "2026-06-01", "index": 1}]
                if fid == 2:
                    return [{"id": 2, "object": "Delhi", "text": "lives in Delhi", "current": False, "session_key": "sess-02", "date": "2026-05-01", "index": 2}]
            if "SUPERSEDES" in query:
                fid = kwargs.get("fid")
                if fid == 1:
                    return [{"id": 2}]
                if fid == 2:
                    return [{"id": 1}]
            return []

        mock_hc_run.side_effect = mock_run_side_effect
        history = get_fact_history(1)
        # Verify that it didn't loop infinitely, and safely returned the traversed items
        self.assertTrue(len(history) <= 2)

    @patch("api.app.hc.run")
    @patch("api.app.llm.chat_json")
    def test_ask_database_failure(self, mock_chat_json, mock_hc_run):
        mock_chat_json.return_value = {"subject": "user", "predicate": "lives_in"}
        mock_hc_run.side_effect = RuntimeError("HydraDB is down")
        
        resp = self.app_client.post("/ask", json={"question": "Where does the user live?"})
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json["status"], "error")
        self.assertIn("Database query failed", resp.json["error"])

    @patch("api.app.hc.run")
    @patch("api.app.llm.chat_json")
    @patch("api.app.llm.chat_text")
    def test_ask_llm_phrasing_failure(self, mock_chat_text, mock_chat_json, mock_hc_run):
        mock_chat_json.return_value = {"subject": "user", "predicate": "lives_in"}
        mock_hc_run.return_value = [
            {"id": 1, "text": "lives in Pune", "object": "Pune", "session_key": "sess-01", "date": "2026-06-01", "index": 1, "current": True}
        ]
        mock_chat_text.side_effect = Exception("LLM phrasing failed")
        
        resp = self.app_client.post("/ask", json={"question": "Where does the user live?"})
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json["status"], "error")
        self.assertIn("LLM answer phrasing failed", resp.json["error"])

    @patch("api.app.hc.run")
    @patch("api.app.llm.chat_json")
    @patch("api.app.llm.chat_text")
    def test_ask_explainability_and_baseline(self, mock_chat_text, mock_chat_json, mock_hc_run):
        mock_chat_json.return_value = {"subject": "user", "predicate": "lives_in"}
        mock_hc_run.return_value = [
            {"id": 1, "text": "lives in Pune", "object": "Pune", "session_key": "sess-01", "date": "2026-06-01", "index": 1, "current": True}
        ]
        mock_chat_text.return_value = "The user currently lives in Pune."
        
        resp = self.app_client.post("/ask", json={"question": "Where does the user currently live?"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["status"], "found")
        
        # Verify explainability evidence
        self.assertIn("evidence", resp.json)
        self.assertTrue(resp.json["evidence"]["fact_found"])
        self.assertTrue(resp.json["evidence"]["current_fact"])
        
        # Verify baseline comparison
        self.assertIn("baseline_comparison", resp.json)
        self.assertEqual(resp.json["baseline_comparison"]["hydramemory"], "Pune")

    @patch("api.app.llm.chat_json")
    def test_ask_ambiguous_predicate(self, mock_chat_json):
        mock_chat_json.return_value = {"subject": "user", "predicate": "none"}
        
        resp = self.app_client.post("/ask", json={"question": "Tell me something interesting about the user."})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["status"], "ambiguous")
        self.assertFalse(resp.json["found"])
        self.assertEqual(resp.json["answer"], "I couldn't confidently map that question to a memory.")

    @patch("api.app.hc.run")
    def test_ask_memory_changes_diff(self, mock_hc_run):
        # Setup queries for "what changed"
        # 1. MATCH for user facts returns one fact for lives_in
        # 2. MATCH stated_in for lives_in returns sessions
        # 3. MATCH supersedes returns empty (no supersession)
        def mock_run_side_effect(query, **kwargs):
            if "current: true" in query:
                return [{"id": 1, "predicate": "lives_in", "object": "Pune"}]
            if "STATED_IN" in query:
                return [{"id": 1, "object": "Pune", "text": "lives in Pune", "current": True, "session_key": "sess-01", "date": "2026-06-01", "index": 1}]
            return []
            
        mock_hc_run.side_effect = mock_run_side_effect
        
        resp = self.app_client.post("/ask", json={"question": "What changed about the user recently?"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["status"], "found")
        self.assertIn("changes", resp.json)
        self.assertEqual(resp.json["changes"][0]["predicate"], "lives_in")
        self.assertEqual(resp.json["changes"][0]["current"], "Pune")


if __name__ == "__main__":
    unittest.main()
