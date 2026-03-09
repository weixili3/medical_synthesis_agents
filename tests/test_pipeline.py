"""Integration tests for the LangGraph pipeline."""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage


def _mock_research_output() -> str:
    return json.dumps({
        "research_queries": ["telemedicine diabetes RCT", "remote monitoring HbA1c"],
        "raw_sources": [
            {
                "title": "Telemedicine for T2D Management",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "snippet": "RCT demonstrating HbA1c reduction via telemedicine.",
                "source_type": "pubmed",
            }
        ],
        "research_summary": (
            "Multiple randomised controlled trials have demonstrated that telemedicine "
            "interventions for Type 2 diabetes management lead to statistically significant "
            "improvements in glycaemic control (HbA1c reduction of 0.5–1.2%). Patient "
            "satisfaction scores are consistently high, and cost analyses suggest telemedicine "
            "reduces overall healthcare expenditure by 15–20%."
        ),
    })


def _mock_analysis_output() -> str:
    return json.dumps({
        "key_findings": [
            "Telemedicine reduces HbA1c by 0.5–1.2% versus standard care.",
            "Patient satisfaction rates exceed 85% across reviewed studies.",
            "Healthcare cost reductions of 15–20% reported.",
            "Adherence to treatment plans improved by 23% with remote monitoring.",
            "No significant adverse events attributable to telemedicine delivery.",
        ],
        "statistical_summary": {"mean_hba1c_reduction": 0.85, "sample_size_total": 2400},
        "evidence_quality": "strong",
        "methodology_types": ["Randomised Controlled Trial", "Systematic Review"],
        "themes": ["telemedicine", "diabetes management", "clinical outcomes"],
    })


def _mock_writing_output() -> str:
    report = """# Clinical Evidence Synthesis: Telemedicine for Type 2 Diabetes

**Date:** January 1, 2025
**Content Type:** clinical_evidence

---

## Executive Summary

This report presents a comprehensive synthesis of clinical evidence on the
effectiveness of telemedicine interventions in managing Type 2 diabetes.
The evidence base, drawn from multiple randomised controlled trials and
systematic reviews, demonstrates that telemedicine significantly improves
glycaemic control and patient outcomes.

---

## Introduction

Type 2 diabetes mellitus represents a major global health burden...

## Methodology

A systematic search of PubMed and Google Scholar was conducted...

## Key Findings

- Telemedicine reduces HbA1c by 0.5-1.2%
- Patient satisfaction exceeds 85%
- Cost reductions of 15-20%

## Evidence Analysis

The body of evidence is robust, comprising multiple RCTs...

## Discussion

These findings suggest telemedicine is an effective adjunct to standard care...

## Conclusions

Telemedicine interventions are effective for Type 2 diabetes management...

## Limitations

Heterogeneity across studies limits direct comparisons...

## References

1. Smith, J. (2023). Telemedicine for diabetes. *NEJM*, 388(1), 1–10.
"""
    return json.dumps({"draft_report": report, "citations": ["Smith, J. (2023). ..."]})


def _mock_quality_output(approved: bool = True) -> str:
    return json.dumps({
        "quality_score": 0.85 if approved else 0.55,
        "is_approved": approved,
        "quality_feedback": [] if approved else ["Discussion section needs more depth."],
        "sub_scores": {
            "completeness": 0.90,
            "relevancy": 0.88,
            "readability": 0.72,
            "grammar": 0.80,
            "clinical_accuracy": 0.85,
        },
        "claim_verification": {"verified_count": 5, "unverified_count": 0, "accuracy_score": 1.0},
    })


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


class TestPipelineBuild(unittest.TestCase):
    """Tests for pipeline graph construction."""

    def test_build_pipeline_returns_compilable_graph(self):
        """build_pipeline should return a compiled LangGraph without errors."""
        from src.pipeline import build_pipeline

        graph = build_pipeline()
        self.assertTrue(hasattr(graph, "invoke"))

    def test_pipeline_graph_has_all_nodes(self):
        """The compiled graph should contain all 5 expected nodes."""
        from src.pipeline import build_pipeline

        graph = build_pipeline()
        node_names = set(graph.nodes.keys())
        expected = {"coordinator", "research", "analysis", "writing", "quality"}
        self.assertTrue(expected.issubset(node_names), f"Missing nodes: {expected - node_names}")


# ---------------------------------------------------------------------------
# End-to-end integration (all agents mocked)
# ---------------------------------------------------------------------------


class TestPipelineRunIntegration(unittest.TestCase):
    """End-to-end pipeline integration tests with all agents mocked."""

    @patch("src.agents.research_agent._get_llm", return_value=MagicMock())
    @patch("src.agents.analysis_agent._get_llm", return_value=MagicMock())
    @patch("src.agents.writing_agent._get_llm", return_value=MagicMock())
    @patch("src.agents.quality_agent._get_llm", return_value=MagicMock())
    @patch("src.agents.research_agent.get_research_tools", return_value=[])
    @patch("src.agents.analysis_agent.get_analysis_tools", return_value=[])
    @patch("src.agents.writing_agent.get_writing_tools", return_value=[])
    @patch("src.agents.quality_agent.get_quality_tools", return_value=[])
    def test_full_pipeline_runs_and_approves(self, *mocks):
        """With mocked agents, the pipeline should run end-to-end and return is_approved=True."""
        analysis_mock = MagicMock()
        analysis_mock.invoke.return_value = {"messages": [AIMessage(content=_mock_analysis_output())]}

        quality_mock = MagicMock()
        quality_mock.invoke.return_value = {"messages": [AIMessage(content=_mock_quality_output(approved=True))]}

        with (
            patch(
                "src.agents.research_agent.run_agent_with_forced_tools",
                return_value={"messages": [AIMessage(content=_mock_research_output())]},
            ),
            patch("src.agents.analysis_agent.create_react_agent", return_value=analysis_mock),
            # Writing agent uses run_agent_with_forced_tools, not create_react_agent
            patch(
                "src.agents.writing_agent.run_agent_with_forced_tools",
                return_value={"messages": [AIMessage(content=_mock_writing_output())]},
            ),
            patch("src.agents.quality_agent.create_react_agent", return_value=quality_mock),
            patch(
                "src.agents.quality_agent.invoke_agent_with_tool_logging",
                return_value={"messages": [AIMessage(content=_mock_quality_output(approved=True))]},
            ),
        ):
            from src.pipeline import run_pipeline

            final_state = run_pipeline(
                request="Create a clinical evidence synthesis for telemedicine and Type 2 diabetes",
                max_iterations=3,
                thread_id="test-thread-001",
            )

        self.assertTrue(final_state.get("is_approved", False))
        self.assertGreater(final_state.get("quality_score", 0), 0.5)
        self.assertNotEqual(final_state.get("draft_report", ""), "")
        self.assertGreater(len(final_state.get("key_findings", [])), 0)

    @patch("src.agents.research_agent._get_llm", return_value=MagicMock())
    @patch("src.agents.research_agent.get_research_tools", return_value=[])
    def test_out_of_scope_request_terminates_early(self, *mocks):
        """An off-topic request should be rejected by the coordinator without calling agents."""
        with patch("src.agents.research_agent.run_agent_with_forced_tools") as mock_research:
            from src.pipeline import run_pipeline

            final_state = run_pipeline(
                request="What is the best pasta recipe?",
                max_iterations=3,
                thread_id="test-oos",
            )

        self.assertTrue(final_state.get("out_of_scope", False))
        mock_research.assert_not_called()


# ---------------------------------------------------------------------------
# State flow
# ---------------------------------------------------------------------------


class TestPipelineStateFlow(unittest.TestCase):
    """Tests that state flows correctly between coordinator and agent nodes."""

    def test_coordinator_output_feeds_research_node(self):
        """Coordinator output should include all fields consumed by research_node."""
        from src.agents.coordinator import coordinator_node

        state = {
            "request": "Telemedicine diabetes clinical evidence synthesis",
            "max_iterations": 3,
            "messages": [],
            "errors": [],
        }
        result = coordinator_node(state)

        self.assertIn("content_type", result)
        self.assertIn("current_phase", result)
        self.assertIn("max_iterations", result)
        self.assertEqual(result["current_phase"], "research")

    def test_coordinator_clears_downstream_state_on_init(self):
        """Coordinator init should zero out stale downstream fields."""
        from src.agents.coordinator import coordinator_node

        state = {
            "request": "Telemedicine diabetes clinical evidence synthesis",
            "max_iterations": 3,
            "messages": [],
            "errors": [],
            # Stale values from a prior run
            "draft_report": "Old report",
            "quality_score": 0.99,
            "is_approved": True,
        }
        result = coordinator_node(state)

        self.assertEqual(result.get("draft_report", ""), "")
        self.assertEqual(result.get("quality_score", 0.0), 0.0)
        self.assertFalse(result.get("is_approved", False))


if __name__ == "__main__":
    unittest.main()
