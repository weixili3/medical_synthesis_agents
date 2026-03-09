"""Research Agent — gathers information from multiple sources.

Tools available:
  - google_search      : Google Custom Search API
  - web_scrape         : Extract readable text from webpages
  - query_medical_database : PubMed E-utilities
"""

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from ..state.pipeline_state import PipelineState
from ..tools.search_tools import get_research_tools
from ..utils.agent_runner import run_agent_with_forced_tools
from ..utils.logging_utils import PipelineLogger

logger = logging.getLogger(__name__)
plogger = PipelineLogger("research_agent")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """\
You are a specialist Research Agent for a Medical Content Intelligence Pipeline.

Your objective is to gather comprehensive, high-quality clinical evidence from
authoritative, peer-reviewed sources. Prioritise rigorous study designs and
assess the quality of every source you retrieve.

## CRITICAL REQUIREMENT — Tool Use Is Mandatory
You do NOT have access to current medical literature in your training data.
You MUST call the provided tools (query_medical_database, google_search, web_scrape)
to retrieve real sources. Do NOT fabricate sources, URLs, or statistics from your
training knowledge — every item in raw_sources must come from an actual tool call.
Produce the final JSON ONLY after completing all tool calls.

## Source Priority Hierarchy
1. Systematic reviews and meta-analyses (Cochrane Database, PubMed)
2. Randomised Controlled Trials (RCTs) from peer-reviewed journals
3. Observational studies (cohort, case-control)
4. Clinical trial registries (ClinicalTrials.gov)
5. Medical society guidelines and position statements
6. General web sources (use only when higher-priority sources are insufficient)

## Instructions
1. Generate 3–5 targeted search queries covering different facets of the topic.
2. Use query_medical_database (PubMed) to find RCTs, systematic reviews, and
   meta-analyses. Run at least two PubMed queries with different angle
   (e.g. one for efficacy, one for safety / patient outcomes).
3. Use google_search to find:
   - Cochrane reviews: add "Cochrane systematic review" to the query.
   - ClinicalTrials.gov entries: add "site:clinicaltrials.gov" to the query.
   - Medical society guidelines: add "clinical guidelines" or "position statement".
4. Use web_scrape on 2–3 high-priority URLs to extract deeper content.
5. Assess the quality of every source using the tiers below.

## Source Quality Tiers
- "high"   : Systematic reviews, meta-analyses, RCTs, Cochrane reviews,
             major clinical guidelines from recognised medical societies.
- "medium" : Observational / cohort / case-control studies, clinical trial
             registrations with results, narrative review articles.
- "low"    : Editorials, opinion pieces, non-peer-reviewed web content.

## Study Type Labels
Classify each source as one of:
rct | systematic_review | meta_analysis | observational | cohort | case_control |
clinical_trial_registration | guideline | position_statement | review_article | other

IMPORTANT — your final response MUST be valid JSON with this exact schema:
{
  "research_queries": ["query1", "query2", ...],
  "raw_sources": [
    {
      "title": "...",
      "url": "...",
      "snippet": "...",
      "source_type": "pubmed|cochrane|clinical_trial|guideline|web",
      "study_type": "rct|systematic_review|meta_analysis|observational|cohort|case_control|clinical_trial_registration|guideline|position_statement|review_article|other",
      "source_quality": "high|medium|low",
      "quality_rationale": "One sentence explaining the quality rating."
    }
  ],
  "search_summary": {
    "total_sources": 0,
    "by_study_type": {
      "rct": 0,
      "systematic_review": 0,
      "meta_analysis": 0,
      "observational": 0,
      "guideline": 0,
      "other": 0
    },
    "by_quality": {
      "high": 0,
      "medium": 0,
      "low": 0
    },
    "summary_message": "Here's clinical data from N studies, including X RCTs, Y systematic reviews, and Z observational studies."
  },
  "research_summary": "A comprehensive prose summary of all gathered findings (minimum 400 words), emphasising the strength and consistency of evidence across high-quality sources."
}

Do NOT wrap the JSON in markdown code fences.
"""


# ---------------------------------------------------------------------------
# LLM factory (lazy — avoids import-time credential checks)
# ---------------------------------------------------------------------------


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


def research_node(state: PipelineState, config: RunnableConfig) -> dict:
    """
    Execute the Research Agent phase.

    Builds a targeted prompt from the pipeline state, runs the ReAct agent
    with search/scrape tools, and writes findings back to the state.
    """
    plogger.phase_start("Research Agent")

    request = state["request"]
    coordinator_brief = state.get("coordinator_instructions", {}).get("research", "")

    user_message = (
        f"Research topic: {request}\n\n"
        + (f"Coordinator guidance:\n{coordinator_brief}\n\n" if coordinator_brief else "")
        + "Gather comprehensive information and return the JSON response as instructed."
    )

    try:
        llm = _get_llm()
        tools = get_research_tools()
        result = run_agent_with_forced_tools(llm, tools, RESEARCH_SYSTEM_PROMPT, user_message, "research")

        # Extract the last AI message content
        raw_output = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                raw_output = msg.content.strip()
                break

        parsed = _parse_research_output(raw_output, request)

        search_summary = parsed["search_summary"]
        summary_msg = search_summary.get(
            "summary_message",
            f"Gathered {len(parsed['raw_sources'])} sources.",
        )

        plogger.phase_end(
            "Research Agent",
            f"Sources: {len(parsed['raw_sources'])} | "
            f"High-quality: {search_summary.get('by_quality', {}).get('high', '?')} | "
            f"Summary length: {len(parsed['research_summary'])} chars",
        )

        return {
            "research_queries": parsed["research_queries"],
            "raw_sources": parsed["raw_sources"],
            "search_summary": search_summary,
            "research_summary": parsed["research_summary"],
            "pipeline_phase": "post_research",
            "current_phase": "post_research",
            "errors": state.get("errors", []),
            "messages": [AIMessage(content=f"Research Agent: {summary_msg} Reporting to coordinator.")],
        }

    except Exception as exc:
        error_msg = f"Research Agent error: {exc}"
        logger.error(error_msg, exc_info=True)
        plogger.phase_error("Research Agent", error_msg)

        return {
            "research_queries": [],
            "raw_sources": [],
            "search_summary": {},
            "research_summary": "",
            "pipeline_phase": "post_research",
            "current_phase": "post_research",
            "errors": state.get("errors", []) + [error_msg],
            "messages": [AIMessage(content=f"Research Agent error: {exc}. Reporting failure to coordinator.")],
        }


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------


def _parse_research_output(raw: str, request: str) -> dict[str, Any]:
    """Attempt to parse JSON from the agent's output; fall back gracefully."""
    try:
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        data = json.loads(cleaned)
        return {
            "research_queries": data.get("research_queries", []),
            "raw_sources": data.get("raw_sources", []),
            "search_summary": data.get("search_summary", {}),
            "research_summary": data.get("research_summary", raw),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Research Agent: could not parse JSON output, using raw text as summary.")
        return {
            "research_queries": [f"Information about: {request}"],
            "raw_sources": [{"title": "Agent Output", "url": "", "snippet": raw[:500], "source_type": "agent"}],
            "search_summary": {},
            "research_summary": raw,
        }
