"""Quality Agent — reviews and validates the generated clinical report.

Tools available:
  - check_readability    : Flesch, FK Grade, Gunning Fog, etc.
  - check_completeness   : Required-section presence validation
  - check_grammar        : LanguageTool / heuristic grammar checking
  - check_relevancy      : Keyword-overlap relevancy scoring
  - check_medical_claims : Cross-reference numerical claims against source snippets
"""

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from ..state.pipeline_state import PipelineState
from ..tools.quality_tools import get_quality_tools
from ..utils.logging_utils import PipelineLogger, ToolLoggingCallback, invoke_agent_with_tool_logging

logger = logging.getLogger(__name__)
plogger = PipelineLogger("quality_agent")

# Composite score thresholds
APPROVAL_THRESHOLD = 0.70   # minimum composite score for auto-approval
MIN_COMPLETENESS   = 0.75   # minimum completeness sub-score

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

QUALITY_SYSTEM_PROMPT = """\
You are a specialist Clinical Quality Agent for a Medical Content Intelligence Pipeline.

Your objective is to rigorously review a clinical evidence synthesis report for
accuracy, completeness, clinical validity, and citation integrity.

## CRITICAL REQUIREMENT — Tool Use Is Mandatory
You MUST call all five quality tools (check_readability, check_completeness,
check_grammar, check_relevancy, check_medical_claims) before computing any scores
or writing any JSON. Do NOT estimate readability scores, completeness, or accuracy
without running the tools — scores will be validated against tool outputs.
Produce the final JSON ONLY after all five tool calls are complete.

## Instructions
1. Call check_readability on the draft report text.
2. Call check_completeness on the draft report using the content_type.
3. Call check_grammar on the draft report.
4. Call check_relevancy with the draft report and the original request.
5. Call check_medical_claims with the report text and the provided raw sources JSON
   to cross-reference numerical claims against original study data.
6. Compute a composite quality score (weighted average):
     - Completeness      : 25%
     - Relevancy         : 25%
     - Clinical accuracy : 25%  (use check_medical_claims accuracy_score)
     - Readability       : 15%  (Flesch ease / 100, capped at 1.0)
     - Grammar           : 10%  (0.0=needs_improvement, 0.5=fair, 0.75=good, 1.0=excellent)
7. List specific, actionable feedback items across all dimensions.

## Clinical Validation Criteria
- **Statistical interpretation**: Verify that p-values, confidence intervals, and
  effect sizes are described correctly (e.g. "statistically significant" requires p < 0.05).
- **Clinical consistency**: Conclusions must logically follow from the stated evidence.
  Flag any overclaiming or unsupported generalisations.
- **Citation integrity**: Every quantitative claim should be traceable to a cited source.
  Flag claims that appear uncited or cite a source not listed in the References section.
- **Guideline alignment**: Check whether clinical recommendations are consistent with
  known major guidelines (e.g. ADA, NICE) or explicitly note where they diverge.

IMPORTANT — your final response MUST be valid JSON with this exact schema:
{
  "quality_score": 0.82,
  "is_approved": true,
  "quality_feedback": [
    "Specific, actionable feedback item with section reference where possible"
  ],
  "sub_scores": {
    "completeness": 0.90,
    "relevancy": 0.85,
    "clinical_accuracy": 0.80,
    "readability": 0.70,
    "grammar": 0.75
  },
  "claim_verification": {
    "verified_count": 10,
    "unverified_count": 2,
    "accuracy_score": 0.83,
    "unverified_claims": ["Claim text that could not be traced to sources..."]
  }
}

Approve (is_approved: true) only when quality_score >= 0.70 AND
completeness >= 0.75 AND clinical_accuracy >= 0.70.
Do NOT wrap the JSON in markdown code fences.
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def quality_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Execute the Quality Agent phase.

    Evaluates the draft report and populates quality_score, quality_feedback,
    and is_approved in the pipeline state.
    """
    plogger.phase_start("Quality Agent")

    draft_report = state.get("draft_report", "")
    request = state["request"]
    content_type = state.get("content_type", "general")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    raw_sources = state.get("raw_sources", [])
    key_findings = state.get("key_findings", [])
    citations = state.get("citations", [])
    evidence_quality = state.get("evidence_quality", "")
    evidence_grade = state.get("evidence_grade", "")

    if not draft_report:
        warning = "No draft report to evaluate."
        logger.warning(warning)
        return _quality_result(state, 0.0, [warning], False)

    # Compact source snippets for claim verification
    sources_for_verification = json.dumps(
        [
            {
                "title": s.get("title", ""),
                "study_type": s.get("study_type", ""),
                "snippet": (s.get("snippet") or "")[:400],
            }
            for s in raw_sources[:20]
        ]
    )

    user_message = (
        f"Original request: {request}\n\n"
        f"Content type: {content_type}\n\n"
        f"Evidence quality from analysis: {evidence_quality} (GRADE {evidence_grade})\n\n"
        f"Key findings from analysis ({len(key_findings)}):\n"
        + "\n".join(f"  - {f}" for f in key_findings[:10])
        + f"\n\nFormatted citations ({len(citations)}):\n"
        + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(citations[:15]))
        + f"\n\nSource snippets for claim verification (JSON):\n{sources_for_verification}\n\n"
        f"Draft report (first 4000 chars):\n{draft_report[:4000]}\n\n"
        f"Full report length: {len(draft_report)} characters\n\n"
        "Evaluate the report quality and clinical accuracy, then return the JSON response."
    )

    try:
        llm = _get_llm()
        tools = get_quality_tools()
        agent = create_react_agent(llm, tools, prompt=QUALITY_SYSTEM_PROMPT)

        agent_config = {
            **config,
            "tags": [*(config.get("tags") or []), "quality"],
            "callbacks": [ToolLoggingCallback("quality")],
        }
        result = invoke_agent_with_tool_logging(agent, {"messages": [HumanMessage(content=user_message)]}, agent_config, "quality")

        raw_output = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                raw_output = msg.content.strip()
                break

        parsed = _parse_quality_output(raw_output)

        plogger.quality_result(
            parsed["quality_score"],
            parsed["is_approved"],
            parsed["quality_feedback"],
        )
        plogger.phase_end("Quality Agent", f"Score: {parsed['quality_score']:.2f}")

        return _quality_result(
            state,
            parsed["quality_score"],
            parsed["quality_feedback"],
            parsed["is_approved"],
        )

    except Exception as exc:
        error_msg = f"Quality Agent error: {exc}"
        logger.error(error_msg, exc_info=True)
        plogger.phase_error("Quality Agent", error_msg)

        # Return unapproved — coordinator decides whether to retry or force-complete
        return _quality_result(
            state,
            0.5,
            [error_msg, "Quality check could not be completed — coordinator will decide next step."],
            False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quality_result(
    state: PipelineState,
    score: float,
    feedback: list[str],
    approved: bool,
) -> dict:
    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "is_approved": approved,
        "pipeline_phase": "post_quality",
        "current_phase": "post_quality",
        "errors": state.get("errors", []),
        "messages": [
            AIMessage(
                content=(
                    f"Quality Agent: score={score:.2f}, "
                    f"approved={'YES' if approved else 'NO'}. "
                    f"Feedback: {'; '.join(feedback[:3])}. "
                    "Reporting to coordinator."
                )
            )
        ],
    }


def _parse_quality_output(raw: str) -> dict[str, Any]:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        data = json.loads(cleaned)
        score = float(data.get("quality_score", 0.5))
        sub_scores = data.get("sub_scores", {})
        completeness = float(sub_scores.get("completeness", 0.5))
        clinical_accuracy = float(sub_scores.get("clinical_accuracy", 0.5))
        # Enforce all three approval conditions
        approved = bool(
            data.get(
                "is_approved",
                score >= APPROVAL_THRESHOLD
                and completeness >= MIN_COMPLETENESS
                and clinical_accuracy >= 0.70,
            )
        )
        return {
            "quality_score": score,
            "is_approved": approved,
            "quality_feedback": data.get("quality_feedback", []),
            "sub_scores": sub_scores,
            "claim_verification": data.get("claim_verification", {}),
        }
    except (json.JSONDecodeError, AttributeError, ValueError):
        logger.warning("Quality Agent: could not parse JSON, using defaults.")
        return {
            "quality_score": 0.5,
            "is_approved": False,
            "quality_feedback": ["Quality assessment parsing failed; manual review recommended."],
            "sub_scores": {},
            "claim_verification": {},
        }


# ---------------------------------------------------------------------------
# Routing helper (used by the LangGraph conditional edge)
# ---------------------------------------------------------------------------


def quality_router(state: PipelineState) -> str:
    """
    Determine the next graph node after quality review.

    Returns:
      "approved"           → END (report is ready)
      "needs_revision"     → writing node (fix prose/structure)
      "needs_more_research"→ research node (missing content)
    """
    if state.get("is_approved", False):
        return "approved"

    feedback = state.get("quality_feedback", [])
    research_signals = ["missing", "insufficient data", "more evidence", "lacks sources", "no citation"]
    needs_research = any(
        signal in fb.lower()
        for fb in feedback
        for signal in research_signals
    )

    return "needs_more_research" if needs_research else "needs_revision"
