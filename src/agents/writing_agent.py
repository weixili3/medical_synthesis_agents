"""Writing Agent — composes the final structured report.

Tools available:
  - generate_report_from_template : Jinja2-based report renderer
  - format_citation               : APA / MLA / Vancouver citation formatter
  - extract_markdown_section      : Pull a specific section from markdown text
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from ..state.pipeline_state import PipelineState
from ..tools.writing_tools import get_writing_tools
from ..utils.agent_runner import run_agent_with_forced_tools
from ..utils.logging_utils import PipelineLogger

logger = logging.getLogger(__name__)
plogger = PipelineLogger("writing_agent")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

WRITING_SYSTEM_PROMPT = """\
You are a specialist Medical Writing Agent for a Clinical Evidence Intelligence Pipeline.

Your objective is to compose a comprehensive, well-structured clinical evidence synthesis
following established medical writing standards.

## CRITICAL REQUIREMENT — Tool Use Is Mandatory
You MUST call generate_report_from_template to render the final report — do NOT write
the report as free text. You MUST call format_citation for each source before adding it
to the citations list. Call create_forest_plot or create_bar_chart when statistical data
is available. Produce the final JSON ONLY after all required tool calls are complete.

## Writing Standards
- Follow PRISMA reporting guidelines where applicable.
- Use precise clinical terminology and evidence-based language throughout.
- Report effect sizes, confidence intervals, and p-values where available.
- Distinguish clearly between statistically significant and non-significant findings.
- Maintain a formal academic register appropriate for a clinical audience.

## Instructions
1. Build the full context object covering all report sections (detailed below).
2. **Chart generation (REQUIRED when statistical data is available)**:
   - If statistical_summary contains outcome data with mean effects and confidence intervals
     (fields like "mean", "CI", or ci_lower/ci_upper), call create_forest_plot.
     Convert each entry in statistical_summary to this format:
       [{"label": "<outcome name>", "effect_size": <mean>, "ci_lower": <CI[0]>, "ci_upper": <CI[1]>}, ...]
     The tool returns a base64 data URI — use this string directly as "chart_path" in the context.
   - If only categorical counts are available (no CIs), call create_bar_chart instead.
   - Do not skip this step when numeric outcome data is present.
3. For each source in raw_sources (up to 15), call format_citation (style="Vancouver")
   and collect the formatted citations — Vancouver style is standard for clinical literature.
4. Call generate_report_from_template with the JSON-encoded context to render the report.
5. Return the final rendered report and citations list.

## Context Object for generate_report_from_template
Required keys:
  - title (str)                          — descriptive clinical report title
  - content_type (str)
  - executive_summary (str, 200-250 words) — include 2-3 key outcome highlights and
                                             the top clinical recommendation
  - introduction (str, 200-300 words)    — background, clinical problem, review scope
  - methodology (str, 150-200 words)     — databases searched, inclusion/exclusion criteria,
                                           study types included; reference PRISMA if applicable
  - prisma_flow (str, optional)          — brief narrative of study selection numbers
  - key_findings (list[str])             — from the analysis, verbatim or lightly edited
  - intervention_analysis (str, optional) — paragraph synthesising by intervention type
  - evidence_analysis (str, 250-350 words) — synthesise across study types; discuss
                                             consistency, heterogeneity, effect sizes.
                                             IMPORTANT: write as plain prose only — do NOT
                                             include "Evidence Quality", "Evidence Grade",
                                             or "Bias Assessment" as headings or sub-sections
                                             here; the template appends those automatically.
  - statistical_summary (png, optional)
  - chart_path (str, optional)           — base64 data URI returned by create_forest_plot or create_bar_chart
  - discussion (str, 300-400 words)      — interpret findings in clinical context,
                                           compare with existing guidelines, address limitations
  - conclusions (str, 150-200 words)     — summary of evidence and overall judgement
  - clinical_recommendations (str, 150-200 words) — numbered, actionable recommendations
                                             for clinicians (e.g. "1. Consider telemedicine
                                             for patients with HbA1c > 8% who cannot attend
                                             in-person appointments.")
  - limitations (str, 100-150 words)     — study limitations and evidence gaps
  - citations (list[str])                — formatted citation strings

IMPORTANT — after calling the tools, your final response MUST be valid JSON:
{
  "draft_report": "<full rendered markdown report>",
  "citations": ["citation 1", "citation 2", ...]
}

Do NOT wrap the JSON in markdown code fences.
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def writing_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Execute the Writing Agent phase.

    Produces a complete markdown report from the research and analysis results.
    """
    plogger.phase_start("Writing Agent")

    request = state["request"]
    key_findings = state.get("key_findings", [])
    research_summary = state.get("research_summary", "")
    evidence_quality = state.get("evidence_quality", "moderate")
    evidence_grade = state.get("evidence_grade", "C")
    bias_assessment = state.get("bias_assessment", "unclear")
    statistical_summary = state.get("statistical_summary", {})
    intervention_categories = state.get("intervention_categories", {})
    outcome_measures = state.get("outcome_measures", {})
    study_limitations = state.get("study_limitations", [])
    clinical_implications = state.get("clinical_implications", "")
    content_type = state.get("content_type", "general")
    raw_sources = state.get("raw_sources", [])
    iteration_count = state.get("iteration_count", 0)
    coordinator_brief = state.get("coordinator_instructions", {}).get("writing", "")

    user_message = (
        f"Content request: {request}\n\n"
        f"Content type: {content_type}\n\n"
        + (f"Coordinator guidance:\n{coordinator_brief}\n\n" if coordinator_brief else "")
        + f"Key findings ({len(key_findings)} items):\n"
        + "\n".join(f"  - {f}" for f in key_findings)
        + f"\n\nEvidence quality: {evidence_quality} | GRADE: {evidence_grade} | Bias: {bias_assessment}\n\n"
        f"Intervention categories:\n{json.dumps(intervention_categories, indent=2)}\n\n"
        f"Outcome measures:\n{json.dumps(outcome_measures, indent=2)}\n\n"
        f"Study limitations: {'; '.join(study_limitations[:5])}\n\n"
        f"Clinical implications: {clinical_implications}\n\n"
        f"Research summary (excerpt):\n{research_summary[:1500]}\n\n"
        f"Statistical summary: {json.dumps(statistical_summary)}\n\n"
        f"Available sources ({len(raw_sources)}): "
        + json.dumps(
            [
                {
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "study_type": s.get("study_type", ""),
                    "source_quality": s.get("source_quality", ""),
                }
                for s in raw_sources[:15]
            ]
        )
        + "\n\nCompose the full clinical report and return the JSON response."
    )

    try:
        llm = _get_llm()
        tools = get_writing_tools()
        result = run_agent_with_forced_tools(llm, tools, WRITING_SYSTEM_PROMPT, user_message, "writing")

        raw_output = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                raw_output = msg.content.strip()
                break

        parsed = _parse_writing_output(raw_output)

        plogger.phase_end(
            "Writing Agent",
            f"Report length: {len(parsed['draft_report'])} chars | "
            f"Citations: {len(parsed['citations'])}",
        )
        plogger.content_preview("Writing Agent — Draft Report Preview", parsed["draft_report"])

        return {
            "draft_report": parsed["draft_report"],
            "citations": parsed["citations"],
            "pipeline_phase": "post_writing",
            "current_phase": "post_writing",
            "iteration_count": iteration_count + 1,
            "errors": state.get("errors", []),
            "messages": [
                AIMessage(
                    content=(
                        f"Writing Agent: report composed "
                        f"({len(parsed['draft_report'])} chars, "
                        f"{len(parsed['citations'])} citations). "
                        "Reporting to coordinator."
                    )
                )
            ],
        }

    except Exception as exc:
        error_msg = f"Writing Agent error: {exc}"
        logger.error(error_msg, exc_info=True)
        plogger.phase_error("Writing Agent", error_msg)

        fallback_report = _fallback_report(request, key_findings, research_summary, evidence_quality)
        return {
            "draft_report": fallback_report,
            "citations": [],
            "pipeline_phase": "post_writing",
            "current_phase": "post_writing",
            "iteration_count": iteration_count + 1,
            "errors": state.get("errors", []) + [error_msg],
            "messages": [AIMessage(content=f"Writing Agent error: {exc}. Fallback report generated. Reporting to coordinator.")],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_writing_output(raw: str) -> dict[str, Any]:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        data = json.loads(cleaned)
        return {
            "draft_report": data.get("draft_report", raw),
            "citations": data.get("citations", []),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Writing Agent: could not parse JSON output, using raw text as report.")
        return {"draft_report": raw, "citations": []}


def _fallback_report(
    request: str,
    key_findings: list[str],
    research_summary: str,
    evidence_quality: str,
) -> str:
    """Minimal markdown report generated without tool use."""
    now = datetime.now().strftime("%B %d, %Y")
    findings_md = "\n".join(f"- {f}" for f in key_findings) if key_findings else "- No findings available."
    return f"""# Report: {request}

**Date:** {now}

---

## Executive Summary

This report presents findings related to the following request: *{request}*.
The research identified {len(key_findings)} key findings with an overall evidence quality of {evidence_quality}.

---

## Key Findings

{findings_md}

---

## Research Summary

{research_summary[:2000] if research_summary else "No research summary available."}

---

*This report was generated automatically by the Content Intelligence Pipeline.*
"""
