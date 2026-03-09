"""Analysis Agent — synthesises research and extracts structured insights.

Tools available:
  - analyze_evidence    : Evidence strength classification and theme detection
  - calculate_statistics: Descriptive + inferential statistics on numerical data
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
from ..tools.analysis_tools import get_analysis_tools
from ..utils.logging_utils import PipelineLogger, ToolLoggingCallback, invoke_agent_with_tool_logging

logger = logging.getLogger(__name__)
plogger = PipelineLogger("analysis_agent")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are a specialist Clinical Analysis Agent for a Medical Content Intelligence Pipeline.

Your objective is to rigorously analyse clinical research evidence and extract
structured, clinically meaningful insights using established evidence frameworks.

## CRITICAL REQUIREMENT — Tool Use Is Mandatory
You MUST call analyze_evidence with the source list before assigning any evidence
grades or writing any JSON. If numerical outcome data is present, you MUST also
call calculate_statistics. Do NOT estimate GRADE ratings, effect sizes, or bias
assessments without running the tools first — tool outputs will be cross-checked.
Produce the final JSON ONLY after all required tool calls are complete.

## Instructions
1. Call analyze_evidence with the full source list (as a JSON string) to classify
   evidence strength and detect clinical themes.
2. For any numerical outcomes reported across studies (HbA1c deltas, adherence rates,
   effect sizes, p-values), call calculate_statistics to derive descriptive statistics
   and 95% confidence intervals.
3. Synthesise findings into the structured JSON output below.

## Clinical Analysis Requirements
- **Categorise by intervention type**: Group study findings under remote monitoring,
  video consultations, mobile apps, and other. Report the number of studies and
  direction of evidence for each category.
- **Outcome measures**: For each primary outcome (e.g. HbA1c, medication adherence,
  quality of life, hospitalisation rates), summarise the direction and magnitude of
  effect, noting any statistically significant results.
- **Study limitations & bias**: Identify methodological limitations (small sample sizes,
  short follow-up, lack of blinding, high attrition, selection bias) present across
  the included studies.
- **GRADE evidence rating**: Assign a GRADE level:
    A = High (further research unlikely to change confidence in estimate)
    B = Moderate (further research likely to affect estimate)
    C = Low (further research very likely to affect estimate)
    D = Very low (estimate very uncertain)
- **Risk of bias**: Rate the aggregate risk across studies as low / moderate / high / unclear.

## Output Schema
IMPORTANT — your final response MUST be valid JSON with this exact schema:
{
  "key_findings": [
    "Specific, evidence-backed finding with study type and direction (min 6 items)"
  ],
  "statistical_summary": {
    "metric_name": "value or object with mean, CI, n_studies"
  },
  "evidence_quality": "strong | moderate-to-strong | moderate | weak-to-moderate | weak",
  "evidence_grade": "A | B | C | D",
  "bias_assessment": "low | moderate | high | unclear",
  "methodology_types": ["RCT", "Systematic Review", "Observational Study", ...],
  "themes": ["theme1", "theme2", ...],
  "intervention_categories": {
    "remote_monitoring": ["Finding or study summary 1", ...],
    "video_consultations": ["Finding or study summary 1", ...],
    "mobile_apps": ["Finding or study summary 1", ...],
    "other": ["Finding or study summary 1", ...]
  },
  "outcome_measures": {
    "hba1c": {
      "summary": "Mean HbA1c reduction of X% (95% CI Y–Z) across N studies.",
      "effect_direction": "favours intervention | favours control | no significant difference",
      "n_studies": 0
    },
    "medication_adherence": { "summary": "...", "effect_direction": "...", "n_studies": 0 },
    "quality_of_life": { "summary": "...", "effect_direction": "...", "n_studies": 0 }
  },
  "study_limitations": [
    "Limitation statement 1 (e.g. most RCTs had follow-up < 12 months)"
  ],
  "clinical_implications": "Prose paragraph on the clinical significance, who benefits most, and conditions under which the evidence applies."
}

Do NOT wrap the JSON in markdown code fences.
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def analysis_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Execute the Analysis Agent phase.

    Receives research outputs from the pipeline state, runs the ReAct agent
    with analysis tools, and returns structured findings.
    """
    plogger.phase_start("Analysis Agent")

    research_summary = state.get("research_summary", "")
    raw_sources = state.get("raw_sources", [])
    search_summary = state.get("search_summary", {})
    request = state["request"]
    coordinator_brief = state.get("coordinator_instructions", {}).get("analysis", "")

    if not research_summary:
        warning = "No research summary available for analysis."
        logger.warning(warning)
        return _empty_analysis(state, warning)

    # Build a compact source metadata block so the agent can reason about study types
    source_metadata = json.dumps(
        [
            {
                "title": s.get("title", ""),
                "source_type": s.get("source_type", ""),
                "study_type": s.get("study_type", "other"),
                "source_quality": s.get("source_quality", ""),
                "quality_rationale": s.get("quality_rationale", ""),
                "snippet": (s.get("snippet") or "")[:300],
            }
            for s in raw_sources[:20]
        ],
        indent=2,
    )

    search_msg = search_summary.get("summary_message", f"{len(raw_sources)} sources gathered.")

    user_message = (
        f"Original request: {request}\n\n"
        + (f"Coordinator guidance:\n{coordinator_brief}\n\n" if coordinator_brief else "")
        + f"Search summary: {search_msg}\n\n"
        f"Research summary:\n{research_summary}\n\n"
        f"Source metadata ({len(raw_sources)} sources):\n{source_metadata}\n\n"
        "Analyse this clinical evidence and return the structured JSON response."
    )

    try:
        llm = _get_llm()
        tools = get_analysis_tools()
        agent = create_react_agent(llm, tools, prompt=ANALYSIS_SYSTEM_PROMPT)

        agent_config = {
            **config,
            "tags": [*(config.get("tags") or []), "analysis"],
            "callbacks": [ToolLoggingCallback("analysis")],
        }
        result = invoke_agent_with_tool_logging(agent, {"messages": [HumanMessage(content=user_message)]}, agent_config, "analysis")

        raw_output = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                raw_output = msg.content.strip()
                break

        parsed = _parse_analysis_output(raw_output)

        plogger.phase_end(
            "Analysis Agent",
            f"Findings: {len(parsed['key_findings'])} | "
            f"Evidence: {parsed['evidence_quality']} (GRADE {parsed['evidence_grade']}) | "
            f"Bias: {parsed['bias_assessment']}",
        )

        return {
            "key_findings": parsed["key_findings"],
            "statistical_summary": parsed["statistical_summary"],
            "evidence_quality": parsed["evidence_quality"],
            "evidence_grade": parsed["evidence_grade"],
            "bias_assessment": parsed["bias_assessment"],
            "intervention_categories": parsed["intervention_categories"],
            "outcome_measures": parsed["outcome_measures"],
            "study_limitations": parsed["study_limitations"],
            "clinical_implications": parsed["clinical_implications"],
            "pipeline_phase": "post_analysis",
            "current_phase": "post_analysis",
            "errors": state.get("errors", []),
            "messages": [
                AIMessage(
                    content=(
                        f"Analysis Agent: {len(parsed['key_findings'])} key findings. "
                        f"Evidence quality: {parsed['evidence_quality']} (GRADE {parsed['evidence_grade']}). "
                        f"Bias risk: {parsed['bias_assessment']}. "
                        "Here's processed clinical evidence with outcome measures, effect sizes, and quality assessments. "
                        "Reporting to coordinator."
                    )
                )
            ],
        }

    except Exception as exc:
        error_msg = f"Analysis Agent error: {exc}"
        logger.error(error_msg, exc_info=True)
        plogger.phase_error("Analysis Agent", error_msg)
        return _empty_analysis(state, error_msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_analysis(state: PipelineState, reason: str) -> dict:
    return {
        "key_findings": [],
        "statistical_summary": {},
        "evidence_quality": "unknown",
        "evidence_grade": "D",
        "bias_assessment": "unclear",
        "intervention_categories": {},
        "outcome_measures": {},
        "study_limitations": [],
        "clinical_implications": "",
        "pipeline_phase": "post_analysis",
        "current_phase": "post_analysis",
        "errors": state.get("errors", []) + [reason],
        "messages": [AIMessage(content=f"Analysis Agent error: {reason}. Reporting to coordinator.")],
    }


def _parse_analysis_output(raw: str) -> dict[str, Any]:
    """Parse JSON from the agent's output; fall back gracefully."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        data = json.loads(cleaned)
        return {
            "key_findings": data.get("key_findings", []),
            "statistical_summary": data.get("statistical_summary", {}),
            "evidence_quality": data.get("evidence_quality", "moderate"),
            "evidence_grade": data.get("evidence_grade", "C"),
            "bias_assessment": data.get("bias_assessment", "unclear"),
            "methodology_types": data.get("methodology_types", []),
            "themes": data.get("themes", []),
            "intervention_categories": data.get("intervention_categories", {}),
            "outcome_measures": data.get("outcome_measures", {}),
            "study_limitations": data.get("study_limitations", []),
            "clinical_implications": data.get("clinical_implications", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Analysis Agent: could not parse JSON, extracting from raw text.")
        lines = [line.strip("- •* ").strip() for line in raw.split("\n") if line.strip().startswith(("-", "•", "*"))]
        return {
            "key_findings": lines if lines else [raw[:300]],
            "statistical_summary": {},
            "evidence_quality": "moderate",
            "evidence_grade": "C",
            "bias_assessment": "unclear",
            "methodology_types": [],
            "themes": [],
            "intervention_categories": {},
            "outcome_measures": {},
            "study_limitations": [],
            "clinical_implications": "",
        }
