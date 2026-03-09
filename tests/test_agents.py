"""Unit tests for all agent node functions."""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage


def _base_state(**overrides) -> dict:
    """Return a minimal valid PipelineState dict for testing."""
    defaults = {
        # Input
        "request": "Create a report on telemedicine for diabetes",
        "content_type": "clinical_evidence",
        # Coordinator orchestration
        "pipeline_phase": "init",
        "coordinator_instructions": {},
        "phase_retry_counts": {},
        "max_retries_per_phase": 2,
        "coordinator_next_action": "",
        "surface_error": False,
        "pipeline_error_message": "",
        # Coordinator gate
        "out_of_scope": False,
        "scope_rejection_reason": "",
        "clarification_needed": False,
        "clarification_question": "",
        # Research
        "research_queries": [],
        "raw_sources": [],
        "search_summary": {},
        "research_summary": "",
        # Analysis
        "key_findings": [],
        "statistical_summary": {},
        "evidence_quality": "unknown",
        "evidence_grade": "C",
        "bias_assessment": "unclear",
        "intervention_categories": {},
        "outcome_measures": {},
        "study_limitations": [],
        "clinical_implications": "",
        # Writing
        "draft_report": "",
        "citations": [],
        # Quality
        "quality_score": 0.0,
        "quality_feedback": [],
        "is_approved": False,
        # Control flow
        "iteration_count": 0,
        "max_iterations": 3,
        "current_phase": "coordinator",
        "errors": [],
        "messages": [],
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Coordinator — init phase
# ---------------------------------------------------------------------------

_MOCK_GATE_RESULT = {
    "in_scope": True,
    "scope_rejection_reason": "",
    "is_clear": True,
    "clarification_question": "",
    "research_brief": "Search PubMed for RCTs on telemedicine diabetes outcomes.",
    "focus_areas": ["HbA1c outcomes", "patient adherence"],
    "key_questions": ["What is the effect size?"],
}


class TestCoordinatorNode(unittest.TestCase):
    """Tests for coordinator_node (init phase) and post-phase handlers."""

    @patch("src.agents.coordinator._llm_init_check", return_value=_MOCK_GATE_RESULT)
    def test_detects_clinical_content_type(self, _mock):
        from src.agents.coordinator import coordinator_node

        state = _base_state(request="Create a clinical evidence synthesis for telemedicine diabetes")
        result = coordinator_node(state)

        self.assertEqual(result["content_type"], "clinical_evidence")

    @patch("src.agents.coordinator._llm_init_check", return_value=_MOCK_GATE_RESULT)
    def test_initialises_all_required_fields(self, _mock):
        from src.agents.coordinator import coordinator_node

        state = _base_state()
        result = coordinator_node(state)

        required_fields = [
            "content_type", "current_phase", "iteration_count", "max_iterations",
            "research_queries", "raw_sources", "research_summary",
            "key_findings", "statistical_summary", "evidence_quality",
            "draft_report", "citations", "quality_score",
            "quality_feedback", "is_approved", "errors",
            "pipeline_phase", "coordinator_instructions",
            "phase_retry_counts", "coordinator_next_action",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_empty_request_sets_error(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(request="")
        result = coordinator_node(state)

        self.assertTrue(result.get("surface_error") or result.get("out_of_scope"))

    @patch("src.agents.coordinator._llm_init_check", return_value=_MOCK_GATE_RESULT)
    def test_preserves_max_iterations(self, _mock):
        from src.agents.coordinator import coordinator_node

        state = _base_state(max_iterations=5)
        result = coordinator_node(state)

        self.assertEqual(result["max_iterations"], 5)

    @patch("src.agents.coordinator._llm_init_check", return_value=_MOCK_GATE_RESULT)
    def test_routes_to_research_on_valid_request(self, _mock):
        from src.agents.coordinator import coordinator_node

        state = _base_state()
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "research")

    def test_routes_out_of_scope_for_off_topic_request(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(request="What is the best Italian pasta recipe?")
        result = coordinator_node(state)

        self.assertTrue(result.get("out_of_scope"))
        self.assertEqual(result.get("coordinator_next_action"), "out_of_scope")

    @patch("src.agents.coordinator._llm_init_check", return_value={
        **_MOCK_GATE_RESULT,
        "is_clear": False,
        "clarification_question": "Which condition and intervention type?",
    })
    def test_routes_clarification_for_vague_request(self, _mock):
        from src.agents.coordinator import coordinator_node

        state = _base_state(request="Tell me about some health stuff")
        result = coordinator_node(state)

        self.assertTrue(result.get("clarification_needed"))
        self.assertEqual(result.get("coordinator_next_action"), "needs_clarification")
        self.assertIn("Which condition", result.get("clarification_question", ""))

    @patch("src.agents.coordinator._llm_init_check", return_value=_MOCK_GATE_RESULT)
    def test_sets_research_coordinator_instructions(self, _mock):
        from src.agents.coordinator import coordinator_node

        state = _base_state()
        result = coordinator_node(state)

        instructions = result.get("coordinator_instructions", {})
        self.assertIn("research", instructions)
        self.assertTrue(len(instructions["research"]) > 0)

    # ------------------------------------------------------------------
    # Post-research routing
    # ------------------------------------------------------------------

    def test_post_research_valid_routes_to_analysis(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_research",
            research_summary="A" * 300,
            raw_sources=[{"title": "Study A", "url": "http://example.com", "snippet": "...", "source_type": "pubmed"}],
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "analysis")

    def test_post_research_insufficient_triggers_retry(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_research",
            research_summary="Too short",
            raw_sources=[],
            phase_retry_counts={},
            max_retries_per_phase=2,
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "research")
        self.assertEqual(result["phase_retry_counts"].get("research"), 1)

    def test_post_research_exhausted_retries_proceeds_degraded(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_research",
            research_summary="Too short",
            raw_sources=[],
            phase_retry_counts={"research": 2},
            max_retries_per_phase=2,
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "analysis")
        self.assertTrue(len(result.get("errors", [])) > 0)

    # ------------------------------------------------------------------
    # Post-analysis routing
    # ------------------------------------------------------------------

    def test_post_analysis_valid_routes_to_writing(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_analysis",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            evidence_quality="moderate",
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "writing")

    def test_post_analysis_sets_writing_instructions(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_analysis",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            evidence_quality="strong",
        )
        result = coordinator_node(state)

        writing_instr = result.get("coordinator_instructions", {}).get("writing", "")
        self.assertGreater(len(writing_instr), 0)

    # ------------------------------------------------------------------
    # Post-quality routing
    # ------------------------------------------------------------------

    def test_post_quality_approved_routes_to_complete(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_quality",
            is_approved=True,
            quality_score=0.85,
            quality_feedback=[],
            iteration_count=1,
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "complete")

    def test_post_quality_max_iterations_forces_complete(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_quality",
            is_approved=False,
            quality_score=0.55,
            quality_feedback=["Report needs improvement."],
            iteration_count=3,
            max_iterations=3,
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "complete")
        self.assertTrue(result.get("is_approved"))

    def test_post_quality_missing_evidence_routes_to_research(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_quality",
            is_approved=False,
            quality_score=0.55,
            quality_feedback=["Report is missing critical evidence on outcomes."],
            iteration_count=1,
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "research")

    def test_post_quality_prose_issue_routes_to_writing(self):
        from src.agents.coordinator import coordinator_node

        state = _base_state(
            pipeline_phase="post_quality",
            is_approved=False,
            quality_score=0.60,
            quality_feedback=["Discussion section lacks depth and clarity."],
            iteration_count=1,
        )
        result = coordinator_node(state)

        self.assertEqual(result.get("coordinator_next_action"), "writing")


# ---------------------------------------------------------------------------
# Coordinator router
# ---------------------------------------------------------------------------


class TestCoordinatorRouter(unittest.TestCase):
    def test_routes_research(self):
        from src.agents.coordinator import coordinator_router
        state = _base_state(coordinator_next_action="research")
        self.assertEqual(coordinator_router(state), "research")

    def test_routes_complete(self):
        from src.agents.coordinator import coordinator_router
        state = _base_state(coordinator_next_action="complete")
        self.assertEqual(coordinator_router(state), "complete")

    def test_routes_quality(self):
        from src.agents.coordinator import coordinator_router
        state = _base_state(coordinator_next_action="quality")
        self.assertEqual(coordinator_router(state), "quality")

    def test_routes_writing(self):
        from src.agents.coordinator import coordinator_router
        state = _base_state(coordinator_next_action="writing")
        self.assertEqual(coordinator_router(state), "writing")

    def test_routes_analysis(self):
        from src.agents.coordinator import coordinator_router
        state = _base_state(coordinator_next_action="analysis")
        self.assertEqual(coordinator_router(state), "analysis")

    def test_defaults_to_surface_error_when_action_missing(self):
        from src.agents.coordinator import coordinator_router
        state = _base_state(coordinator_next_action="")
        result = coordinator_router(state)
        self.assertIn(result, ["surface_error", ""])


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------


class TestResearchNode(unittest.TestCase):

    @patch("src.agents.research_agent._get_llm")
    @patch("src.agents.research_agent.get_research_tools")
    @patch("src.agents.research_agent.run_agent_with_forced_tools")
    def test_successful_research_updates_state(self, mock_runner, mock_tools, mock_llm):
        from src.agents.research_agent import research_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()

        fake_output = json.dumps({
            "research_queries": ["telemedicine diabetes RCT"],
            "raw_sources": [{"title": "Study A", "url": "https://example.com", "snippet": "...", "source_type": "web"}],
            "research_summary": "Telemedicine shows promise for diabetes management.",
        })
        mock_runner.return_value = {"messages": [AIMessage(content=fake_output)]}

        state = _base_state()
        result = research_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_research")
        self.assertEqual(len(result["research_queries"]), 1)
        self.assertEqual(len(result["raw_sources"]), 1)
        self.assertIn("telemedicine", result["research_summary"].lower())

    @patch("src.agents.research_agent._get_llm")
    @patch("src.agents.research_agent.get_research_tools")
    @patch("src.agents.research_agent.run_agent_with_forced_tools")
    def test_agent_error_sets_pipeline_phase_and_error(self, mock_runner, mock_tools, mock_llm):
        from src.agents.research_agent import research_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_runner.side_effect = RuntimeError("LLM API unavailable")

        state = _base_state()
        result = research_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_research")
        self.assertTrue(len(result.get("errors", [])) > 0)
        self.assertEqual(result["research_summary"], "")

    @patch("src.agents.research_agent._get_llm")
    @patch("src.agents.research_agent.get_research_tools")
    @patch("src.agents.research_agent.run_agent_with_forced_tools")
    def test_coordinator_brief_included_in_prompt(self, mock_runner, mock_tools, mock_llm):
        from src.agents.research_agent import research_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_runner.return_value = {"messages": [AIMessage(content=json.dumps({
            "research_queries": ["q1"],
            "raw_sources": [],
            "research_summary": "Summary text.",
        }))]}

        state = _base_state(coordinator_instructions={"research": "Focus on RCT studies only."})
        research_node(state, {})

        # run_agent_with_forced_tools(llm, tools, system_prompt, user_message, agent_name)
        call_args = mock_runner.call_args
        user_message_arg = call_args[0][3]
        self.assertIn("Focus on RCT studies only.", user_message_arg)

    @patch("src.agents.research_agent._get_llm")
    @patch("src.agents.research_agent.get_research_tools")
    @patch("src.agents.research_agent.run_agent_with_forced_tools")
    def test_malformed_json_falls_back_to_raw_text(self, mock_runner, mock_tools, mock_llm):
        from src.agents.research_agent import research_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_runner.return_value = {
            "messages": [AIMessage(content="This is a plain text research summary without JSON.")]
        }

        state = _base_state()
        result = research_node(state, {})

        self.assertIn("plain text research summary", result["research_summary"])


# ---------------------------------------------------------------------------
# Analysis Agent
# ---------------------------------------------------------------------------


class TestAnalysisNode(unittest.TestCase):

    @patch("src.agents.analysis_agent._get_llm")
    @patch("src.agents.analysis_agent.get_analysis_tools")
    @patch("src.agents.analysis_agent.create_react_agent")
    def test_no_research_summary_returns_empty_analysis(self, mock_agent, mock_tools, mock_llm):
        from src.agents.analysis_agent import analysis_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()

        state = _base_state(research_summary="")
        result = analysis_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_analysis")
        self.assertIn("key_findings", result)
        self.assertTrue(len(result.get("errors", [])) > 0)

    @patch("src.agents.analysis_agent._get_llm")
    @patch("src.agents.analysis_agent.get_analysis_tools")
    @patch("src.agents.analysis_agent.create_react_agent")
    def test_successful_analysis_updates_state(self, mock_create_agent, mock_tools, mock_llm):
        from src.agents.analysis_agent import analysis_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()

        fake_output = json.dumps({
            "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
            "statistical_summary": {"sample_size": 500},
            "evidence_quality": "strong",
            "methodology_types": ["RCT"],
            "themes": ["telemedicine"],
        })
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = {"messages": [AIMessage(content=fake_output)]}
        mock_create_agent.return_value = mock_agent_instance

        state = _base_state(research_summary="Telemedicine interventions show promise.")
        result = analysis_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_analysis")
        self.assertEqual(len(result["key_findings"]), 3)
        self.assertEqual(result["evidence_quality"], "strong")

    @patch("src.agents.analysis_agent._get_llm")
    @patch("src.agents.analysis_agent.get_analysis_tools")
    @patch("src.agents.analysis_agent.create_react_agent")
    def test_analysis_error_still_sets_post_analysis_phase(self, mock_create_agent, mock_tools, mock_llm):
        from src.agents.analysis_agent import analysis_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_create_agent.side_effect = RuntimeError("Connection error")

        state = _base_state(research_summary="Some research summary here.")
        result = analysis_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_analysis")
        self.assertTrue(len(result.get("errors", [])) > 0)


# ---------------------------------------------------------------------------
# Writing Agent
# ---------------------------------------------------------------------------


class TestWritingNode(unittest.TestCase):

    @patch("src.agents.writing_agent._get_llm")
    @patch("src.agents.writing_agent.get_writing_tools")
    @patch("src.agents.writing_agent.run_agent_with_forced_tools")
    def test_successful_writing_updates_state(self, mock_runner, mock_tools, mock_llm):
        from src.agents.writing_agent import writing_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()

        report = "# Clinical Evidence Synthesis\n\n## Executive Summary\nSome summary.\n"
        fake_output = json.dumps({
            "draft_report": report,
            "citations": ["Smith, J. (2023). Example citation."],
        })
        mock_runner.return_value = {"messages": [AIMessage(content=fake_output)]}

        state = _base_state(
            key_findings=["Finding 1", "Finding 2"],
            research_summary="Some research findings.",
            raw_sources=[{"title": "Study A", "url": "http://example.com"}],
        )
        result = writing_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_writing")
        self.assertIn("Clinical Evidence Synthesis", result["draft_report"])
        self.assertEqual(len(result["citations"]), 1)

    @patch("src.agents.writing_agent._get_llm")
    @patch("src.agents.writing_agent.get_writing_tools")
    @patch("src.agents.writing_agent.run_agent_with_forced_tools")
    def test_writing_error_generates_fallback_report(self, mock_runner, mock_tools, mock_llm):
        from src.agents.writing_agent import writing_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_runner.side_effect = RuntimeError("LLM unavailable")

        state = _base_state(key_findings=["Finding 1"], research_summary="Some research.")
        result = writing_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_writing")
        self.assertNotEqual(result["draft_report"], "")
        self.assertTrue(len(result.get("errors", [])) > 0)

    @patch("src.agents.writing_agent._get_llm")
    @patch("src.agents.writing_agent.get_writing_tools")
    @patch("src.agents.writing_agent.run_agent_with_forced_tools")
    def test_malformed_json_falls_back_to_raw_text(self, mock_runner, mock_tools, mock_llm):
        from src.agents.writing_agent import writing_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_runner.return_value = {"messages": [AIMessage(content="Plain text report without JSON.")]}

        state = _base_state()
        result = writing_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_writing")
        self.assertIn("Plain text report", result["draft_report"])

    @patch("src.agents.writing_agent._get_llm")
    @patch("src.agents.writing_agent.get_writing_tools")
    @patch("src.agents.writing_agent.run_agent_with_forced_tools")
    def test_iteration_count_incremented(self, mock_runner, mock_tools, mock_llm):
        from src.agents.writing_agent import writing_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_runner.return_value = {"messages": [AIMessage(content=json.dumps({
            "draft_report": "# Report\nContent.",
            "citations": [],
        }))]}

        state = _base_state(iteration_count=1)
        result = writing_node(state, {})

        self.assertEqual(result["iteration_count"], 2)


# ---------------------------------------------------------------------------
# Quality Agent
# ---------------------------------------------------------------------------


class TestQualityNode(unittest.TestCase):

    @patch("src.agents.quality_agent._get_llm")
    @patch("src.agents.quality_agent.get_quality_tools")
    @patch("src.agents.quality_agent.create_react_agent")
    @patch("src.agents.quality_agent.invoke_agent_with_tool_logging")
    def test_approved_report_sets_is_approved_true(self, mock_invoke, mock_create, mock_tools, mock_llm):
        from src.agents.quality_agent import quality_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        quality_output = json.dumps({
            "quality_score": 0.85,
            "is_approved": True,
            "quality_feedback": [],
            "sub_scores": {
                "completeness": 0.90, "relevancy": 0.88,
                "clinical_accuracy": 0.80, "readability": 0.70, "grammar": 0.75,
            },
            "claim_verification": {"verified_count": 10, "unverified_count": 0, "accuracy_score": 1.0},
        })
        mock_invoke.return_value = {"messages": [AIMessage(content=quality_output)]}

        state = _base_state(
            draft_report="# Report\n\nSome clinical content.\n",
            raw_sources=[{"title": "Study A", "snippet": "Some data."}],
        )
        result = quality_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_quality")
        self.assertTrue(result["is_approved"])
        self.assertAlmostEqual(result["quality_score"], 0.85)

    @patch("src.agents.quality_agent._get_llm")
    @patch("src.agents.quality_agent.get_quality_tools")
    @patch("src.agents.quality_agent.create_react_agent")
    @patch("src.agents.quality_agent.invoke_agent_with_tool_logging")
    def test_low_score_report_not_approved(self, mock_invoke, mock_create, mock_tools, mock_llm):
        from src.agents.quality_agent import quality_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        quality_output = json.dumps({
            "quality_score": 0.55,
            "is_approved": False,
            "quality_feedback": ["Report lacks depth in evidence analysis."],
            "sub_scores": {
                "completeness": 0.60, "relevancy": 0.50, "clinical_accuracy": 0.55,
            },
            "claim_verification": {},
        })
        mock_invoke.return_value = {"messages": [AIMessage(content=quality_output)]}

        state = _base_state(draft_report="# Short Report\n\nMinimal content.")
        result = quality_node(state, {})

        self.assertFalse(result["is_approved"])
        self.assertLess(result["quality_score"], 0.70)

    def test_empty_draft_skips_llm_and_returns_zero_score(self):
        from src.agents.quality_agent import quality_node

        state = _base_state(draft_report="")
        result = quality_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_quality")
        self.assertEqual(result["quality_score"], 0.0)
        self.assertFalse(result["is_approved"])

    @patch("src.agents.quality_agent._get_llm")
    @patch("src.agents.quality_agent.get_quality_tools")
    @patch("src.agents.quality_agent.create_react_agent")
    @patch("src.agents.quality_agent.invoke_agent_with_tool_logging")
    def test_malformed_json_uses_default_scores(self, mock_invoke, mock_create, mock_tools, mock_llm):
        from src.agents.quality_agent import quality_node

        mock_tools.return_value = []
        mock_llm.return_value = MagicMock()
        mock_create.return_value = MagicMock()
        mock_invoke.return_value = {"messages": [AIMessage(content="not valid json")]}

        state = _base_state(draft_report="# Report\n\nSome content.")
        result = quality_node(state, {})

        self.assertEqual(result["pipeline_phase"], "post_quality")
        self.assertFalse(result["is_approved"])


# ---------------------------------------------------------------------------
# Quality router (legacy helper — kept for backwards compatibility)
# ---------------------------------------------------------------------------


class TestQualityRouter(unittest.TestCase):
    """Tests for the quality_router function."""

    def test_approved_state_routes_to_end(self):
        from src.agents.quality_agent import quality_router
        state = _base_state(is_approved=True, quality_feedback=[])
        self.assertEqual(quality_router(state), "approved")

    def test_missing_research_routes_to_research(self):
        from src.agents.quality_agent import quality_router
        state = _base_state(
            is_approved=False,
            quality_feedback=["Report is missing critical evidence on outcomes."],
        )
        self.assertEqual(quality_router(state), "needs_more_research")

    def test_prose_issue_routes_to_writing(self):
        from src.agents.quality_agent import quality_router
        state = _base_state(
            is_approved=False,
            quality_feedback=["The discussion section lacks depth and clarity."],
        )
        self.assertEqual(quality_router(state), "needs_revision")

    def test_no_feedback_unapproved_routes_to_revision(self):
        from src.agents.quality_agent import quality_router
        state = _base_state(is_approved=False, quality_feedback=[])
        self.assertEqual(quality_router(state), "needs_revision")


if __name__ == "__main__":
    unittest.main()
