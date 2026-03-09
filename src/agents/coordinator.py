"""Coordinator Agent — phase-aware hub that manages all agent interactions.

The coordinator is the only node with conditional outbound edges. Every agent
routes back to coordinator on completion (or error), and coordinator decides
what happens next.

Phase flow:
  coordinator (init)
      ↓ proceed
  research → coordinator (post_research)
      ↓ proceed / retry research
  analysis → coordinator (post_analysis)
      ↓ proceed / retry analysis
  writing  → coordinator (post_writing)
      ↓ proceed
  quality  → coordinator (post_quality)
      ↓ complete / needs_revision / needs_more_research

Error management:
  Each phase handler validates the agent's output with cheap heuristics.
  On failure it either retries (up to max_retries_per_phase) or proceeds
  degraded with a warning appended to state["errors"].
  Unrecoverable failures set surface_error=True → END.

LLM usage (cost discipline):
  • init          — one structured call: scope + clarity + task decomposition
  • post_research — LLM only when heuristic validation fails AND we've exhausted
                    keyword-based diagnosis (rare)
  • post_analysis — heuristic only
  • post_writing  — heuristic only
  • post_quality  — heuristic only (reads quality_score + keyword signals)
"""

import json
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..state.pipeline_state import PipelineState
from ..utils.logging_utils import PipelineLogger
from .scope_gate import keyword_scope_check

logger = logging.getLogger(__name__)
plogger = PipelineLogger("coordinator")

# ---------------------------------------------------------------------------
# Heuristic thresholds (configurable via env)
# ---------------------------------------------------------------------------

MIN_RESEARCH_SUMMARY_CHARS = int(os.getenv("MIN_RESEARCH_SUMMARY_CHARS", "200"))
MIN_SOURCE_COUNT = int(os.getenv("MIN_SOURCE_COUNT", "1"))
MIN_KEY_FINDINGS = int(os.getenv("MIN_KEY_FINDINGS", "3"))
MIN_DRAFT_CHARS = int(os.getenv("MIN_DRAFT_CHARS", "400"))

# ---------------------------------------------------------------------------
# Content-type detection
# ---------------------------------------------------------------------------

CONTENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "clinical_evidence": [
        "clinical", "evidence", "trial", "rct", "systematic review",
        "meta-analysis", "therapy", "treatment", "efficacy", "telemedicine",
        "diabetes", "intervention", "patient", "health",
    ],
    "market_report": [
        "market", "industry", "revenue", "growth", "competitor", "trend",
        "forecast", "segment", "consumer", "sales",
    ],
    "technology_analysis": [
        "technology", "software", "ai", "machine learning", "algorithm",
        "infrastructure", "cloud", "api", "framework", "data",
    ],
    "policy_brief": [
        "policy", "regulation", "government", "legislation", "compliance",
        "governance", "public", "law", "mandate",
    ],
}


def _detect_content_type(request: str) -> str:
    request_lower = request.lower()
    scores = {
        ct: sum(1 for kw in kws if kw in request_lower)
        for ct, kws in CONTENT_TYPE_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# LLM (shared factory)
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Init LLM gate — scope + clarity + task decomposition (single call)
# ---------------------------------------------------------------------------

_INIT_SYSTEM_PROMPT = """\
You are the entry-point coordinator for a medical research pipeline.

For the given research request, do three things in one response:

1. SCOPE — Is this a medical, clinical, or health science research request?
   In-scope: clinical trials, drug efficacy, disease management, telemedicine,
   epidemiology, systematic reviews, public health interventions.
   Out-of-scope: cooking, sports, finance, travel, entertainment.

2. CLARITY — Is the request specific enough for a researcher to proceed?
   A clear request names a topic, condition, intervention, or outcome of interest.
   Unclear examples: "tell me about diabetes", "is telemedicine good?".

3. TASK DECOMPOSITION — If in scope and clear, produce a structured research brief.

Return valid JSON (no markdown fences):
{
  "in_scope": true or false,
  "scope_rejection_reason": "brief explanation (only when in_scope is false)",
  "is_clear": true or false,
  "clarification_question": "one specific follow-up question (only when is_clear is false)",
  "research_brief": "2-3 sentence guidance for the search agent: what to look for, which study types, which databases",
  "focus_areas": ["specific aspect 1", "specific aspect 2", "specific aspect 3"],
  "key_questions": ["Research question 1?", "Research question 2?", "Research question 3?"]
}
"""

_INIT_DEFAULTS = {
    "in_scope": True,
    "scope_rejection_reason": "",
    "is_clear": True,
    "clarification_question": "",
    "research_brief": "",
    "focus_areas": [],
    "key_questions": [],
}


def _llm_init_check(request: str) -> dict:
    """
    Single LLM call: scope gate + clarity check + task decomposition.
    Fails open on any exception.
    """
    try:
        llm = _get_llm()
        response = llm.invoke(
            [HumanMessage(content=f"{_INIT_SYSTEM_PROMPT}\n\nUser request: {request}")]
        )
        raw = response.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(raw)
        return {
            "in_scope": bool(data.get("in_scope", True)),
            "scope_rejection_reason": str(data.get("scope_rejection_reason", "")),
            "is_clear": bool(data.get("is_clear", True)),
            "clarification_question": str(data.get("clarification_question", "")),
            "research_brief": str(data.get("research_brief", "")),
            "focus_areas": list(data.get("focus_areas", [])),
            "key_questions": list(data.get("key_questions", [])),
        }
    except Exception as exc:
        logger.warning("Coordinator init LLM failed (%s); failing open.", exc)
        return _INIT_DEFAULTS.copy()


# ---------------------------------------------------------------------------
# Heuristic validators — no LLM cost on the happy path
# ---------------------------------------------------------------------------

def _validate_research(state: PipelineState) -> tuple[bool, str]:
    summary = state.get("research_summary", "")
    sources = state.get("raw_sources", [])
    errors = state.get("errors", [])
    has_agent_error = any("Research Agent error" in e for e in errors)

    if has_agent_error and len(summary) < MIN_RESEARCH_SUMMARY_CHARS and len(sources) < MIN_SOURCE_COUNT:
        return False, (
            f"Research failed with errors and produced insufficient output "
            f"({len(sources)} sources, {len(summary)} chars summary)."
        )
    if len(summary) < MIN_RESEARCH_SUMMARY_CHARS:
        return False, f"Research summary too short ({len(summary)} chars; minimum {MIN_RESEARCH_SUMMARY_CHARS})."
    return True, ""


def _validate_analysis(state: PipelineState) -> tuple[bool, str]:
    findings = state.get("key_findings", [])
    errors = state.get("errors", [])
    has_agent_error = any("Analysis Agent error" in e for e in errors)

    if has_agent_error and len(findings) < MIN_KEY_FINDINGS:
        return False, f"Analysis failed with errors and produced only {len(findings)} findings."
    if len(findings) < MIN_KEY_FINDINGS:
        return False, f"Too few key findings ({len(findings)}; minimum {MIN_KEY_FINDINGS})."
    return True, ""


def _validate_writing(state: PipelineState) -> tuple[bool, str]:
    draft = state.get("draft_report", "")
    errors = state.get("errors", [])
    has_agent_error = any("Writing Agent error" in e for e in errors)

    if has_agent_error and len(draft) < MIN_DRAFT_CHARS:
        return False, f"Writing agent produced only a fallback report ({len(draft)} chars)."
    if len(draft) < MIN_DRAFT_CHARS:
        return False, f"Draft report too short ({len(draft)} chars; minimum {MIN_DRAFT_CHARS})."
    return True, ""


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _should_retry(phase_name: str, state: PipelineState) -> tuple[bool, dict]:
    """
    Check if we have retry budget for this phase.

    Returns (can_retry, updated_phase_retry_counts).
    """
    counts = dict(state.get("phase_retry_counts", {}))
    max_retries = state.get("max_retries_per_phase", 2)
    used = counts.get(phase_name, 0)
    if used < max_retries:
        counts[phase_name] = used + 1
        return True, counts
    return False, counts


# ---------------------------------------------------------------------------
# Instruction builders (heuristic, no LLM)
# ---------------------------------------------------------------------------

def _build_analysis_instructions(state: PipelineState) -> str:
    sources = state.get("raw_sources", [])
    content_type = state.get("content_type", "general")
    focus_areas = state.get("coordinator_instructions", {}).get("focus_areas", [])
    key_questions = state.get("coordinator_instructions", {}).get("key_questions", [])

    lines = [f"Analyse the {len(sources)} sources gathered by the Research Agent."]
    if content_type == "clinical_evidence":
        lines.append(
            "Prioritise: RCT effect sizes, systematic review conclusions, "
            "evidence quality ratings, patient outcome measures (HbA1c, adherence, QoL)."
        )
    if focus_areas:
        lines.append(f"Key focus areas: {'; '.join(focus_areas)}.")
    if key_questions:
        lines.append("Ensure your findings address these research questions:")
        lines.extend(f"  • {q}" for q in key_questions)
    return "\n".join(lines)


def _build_writing_instructions(state: PipelineState) -> str:
    evidence_quality = state.get("evidence_quality", "unknown")
    content_type = state.get("content_type", "general")
    findings_count = len(state.get("key_findings", []))
    quality_feedback = state.get("quality_feedback", [])

    lines = [
        f"Write a {content_type.replace('_', ' ')} report.",
        f"Base the report on {findings_count} key findings with overall evidence quality: {evidence_quality}.",
    ]
    if content_type == "clinical_evidence":
        lines.append(
            "Structure: Executive Summary, Introduction, Methodology, "
            "Key Findings, Evidence Analysis, Statistical Summary, "
            "Discussion, Conclusions, Limitations, References."
        )
        lines.append("Use PRISMA guidelines where applicable. Maintain clinical terminology throughout.")
    if quality_feedback:
        lines.append("Address these issues from the previous quality review:")
        lines.extend(f"  • {fb}" for fb in quality_feedback)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared blank state (returned on early exits)
# ---------------------------------------------------------------------------

def _blank_downstream(state: PipelineState) -> dict:
    """Return safe defaults for all downstream pipeline fields."""
    return {
        "content_type": state.get("content_type", "unknown"),
        "pipeline_phase": "init",
        "coordinator_instructions": state.get("coordinator_instructions", {}),
        "phase_retry_counts": state.get("phase_retry_counts", {}),
        "max_retries_per_phase": state.get("max_retries_per_phase", 2),
        "coordinator_next_action": "surface_error",
        "surface_error": False,
        "pipeline_error_message": "",
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "research_queries": [],
        "raw_sources": [],
        "research_summary": "",
        "key_findings": [],
        "statistical_summary": {},
        "evidence_quality": "unknown",
        "draft_report": "",
        "citations": [],
        "quality_score": 0.0,
        "quality_feedback": [],
        "is_approved": False,
        "errors": state.get("errors", []),
    }


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------

def _handle_init(state: PipelineState) -> dict:
    """
    First coordinator call: validate input, gate scope/clarity,
    decompose task, dispatch research.
    """
    request = state.get("request", "").strip()

    if not request:
        plogger.phase_error("Coordinator[init]", "Empty request.")
        return {
            **_blank_downstream(state),
            "out_of_scope": True,
            "scope_rejection_reason": "No request was provided.",
            "clarification_needed": False,
            "clarification_question": "",
            "current_phase": "error",
            "coordinator_next_action": "surface_error",
            "surface_error": True,
            "pipeline_error_message": "No request was provided.",
            "messages": [AIMessage(content="Coordinator: no request provided — aborting.")],
        }

    # Tier 1: keyword fast-path for scope
    keyword_verdict = keyword_scope_check(request)

    if keyword_verdict == "out_of_scope":
        reason = (
            "This pipeline is specialised for medical, clinical, and health science "
            "research. Your request does not appear to be within that domain."
        )
        plogger.phase_end("Coordinator[init]", "OUT OF SCOPE (keyword)")
        return {
            **_blank_downstream(state),
            "out_of_scope": True,
            "scope_rejection_reason": reason,
            "clarification_needed": False,
            "clarification_question": "",
            "current_phase": "rejected",
            "coordinator_next_action": "out_of_scope",
            "messages": [
                HumanMessage(content=request),
                AIMessage(content=f"Coordinator: rejected — {reason}"),
            ],
        }

    # Tier 2: LLM — scope (if ambiguous) + clarity + task decomposition
    gate = _llm_init_check(request)

    if not gate["in_scope"]:
        reason = gate["scope_rejection_reason"] or (
            "This pipeline handles medical, clinical, and health science research only."
        )
        plogger.phase_end("Coordinator[init]", "OUT OF SCOPE (LLM)")
        return {
            **_blank_downstream(state),
            "out_of_scope": True,
            "scope_rejection_reason": reason,
            "clarification_needed": False,
            "clarification_question": "",
            "current_phase": "rejected",
            "coordinator_next_action": "out_of_scope",
            "messages": [
                HumanMessage(content=request),
                AIMessage(content=f"Coordinator: rejected — {reason}"),
            ],
        }

    if not gate["is_clear"]:
        question = gate["clarification_question"] or (
            "Could you specify the condition, intervention, population, or outcome "
            "you'd like investigated?"
        )
        plogger.phase_end("Coordinator[init]", "NEEDS CLARIFICATION")
        return {
            **_blank_downstream(state),
            "out_of_scope": False,
            "scope_rejection_reason": "",
            "clarification_needed": True,
            "clarification_question": question,
            "current_phase": "awaiting_clarification",
            "coordinator_next_action": "needs_clarification",
            "messages": [
                HumanMessage(content=request),
                AIMessage(content=f"Coordinator: clarification needed — {question}"),
            ],
        }

    # Proceed: build coordinator instructions for research
    content_type = _detect_content_type(request)
    research_instructions = gate["research_brief"] or (
        f"Research the following request comprehensively: {request}. "
        "Prioritise peer-reviewed sources, clinical databases (PubMed), and "
        "systematic reviews."
    )
    coordinator_instructions = {
        "research": research_instructions,
        "focus_areas": gate["focus_areas"],
        "key_questions": gate["key_questions"],
        # analysis and writing instructions will be built by later phase handlers
    }

    plogger.phase_end(
        "Coordinator[init]",
        f"PROCEED | type={content_type} | focus_areas={len(gate['focus_areas'])}",
    )
    plogger.agent_decision("Coordinator", "dispatch_research", f"type={content_type}")
    plogger.coordinator_dispatch("Research Agent", research_instructions)

    return {
        **_blank_downstream(state),
        "request": request,
        "content_type": content_type,
        "out_of_scope": False,
        "scope_rejection_reason": "",
        "clarification_needed": False,
        "clarification_question": "",
        "pipeline_phase": "init",
        "coordinator_instructions": coordinator_instructions,
        "phase_retry_counts": {},
        "max_retries_per_phase": state.get("max_retries_per_phase", 2),
        "coordinator_next_action": "research",
        "current_phase": "research",
        "messages": [
            HumanMessage(content=request),
            AIMessage(
                content=(
                    f"Coordinator[init]: dispatching Research Agent.\n"
                    f"  Content type : {content_type}\n"
                    f"  Focus areas  : {', '.join(gate['focus_areas']) or 'general'}\n"
                    f"  Key questions: {len(gate['key_questions'])}"
                )
            ),
        ],
    }


def _handle_post_research(state: PipelineState) -> dict:
    """
    Validate research output. Retry research or proceed to analysis.
    On repeated failure, proceed degraded rather than blocking the user.
    """
    ok, reason = _validate_research(state)
    errors = list(state.get("errors", []))

    if ok:
        # Enrich analysis instructions from what research actually found
        analysis_instructions = _build_analysis_instructions(state)
        coordinator_instructions = {
            **state.get("coordinator_instructions", {}),
            "analysis": analysis_instructions,
        }
        plogger.phase_end(
            "Coordinator[post_research]",
            f"Research validated — sources={len(state.get('raw_sources', []))} | dispatching analysis",
        )
        plogger.agent_decision("Coordinator", "dispatch_analysis")
        plogger.coordinator_dispatch("Analysis Agent", analysis_instructions)
        return {
            "coordinator_instructions": coordinator_instructions,
            "coordinator_next_action": "analysis",
            "phase_retry_counts": state.get("phase_retry_counts", {}),
            "current_phase": "analysis",
            "errors": errors,
            "messages": [
                AIMessage(
                    content=(
                        f"Coordinator[post_research]: research validated "
                        f"({len(state.get('raw_sources', []))} sources). "
                        "Dispatching Analysis Agent."
                    )
                )
            ],
        }

    # Research output below threshold
    can_retry, updated_counts = _should_retry("research", state)
    if can_retry:
        retry_num = updated_counts.get("research", 1)
        retry_note = (
            f"Previous attempt produced insufficient results: {reason} "
            "Please search with broader or alternative queries."
        )
        coordinator_instructions = {
            **state.get("coordinator_instructions", {}),
            "research": (
                state.get("coordinator_instructions", {}).get("research", "")
                + f"\n\n[RETRY {retry_num}] {retry_note}"
            ),
        }
        warn = f"Coordinator[post_research]: research insufficient — retrying (attempt {retry_num}). {reason}"
        logger.warning(warn)
        plogger.agent_decision("Coordinator", "retry_research", f"attempt={retry_num}")
        return {
            "coordinator_instructions": coordinator_instructions,
            "coordinator_next_action": "research",
            "phase_retry_counts": updated_counts,
            "current_phase": "research",
            "errors": errors + [warn],
            "messages": [AIMessage(content=warn)],
        }

    # Retry budget exhausted — proceed degraded
    warn = f"Coordinator[post_research]: max retries exhausted. Proceeding degraded. Last issue: {reason}"
    logger.warning(warn)
    analysis_instructions = _build_analysis_instructions(state)
    coordinator_instructions = {
        **state.get("coordinator_instructions", {}),
        "analysis": analysis_instructions,
    }
    plogger.agent_decision("Coordinator", "proceed_degraded_research")
    return {
        "coordinator_instructions": coordinator_instructions,
        "coordinator_next_action": "analysis",
        "phase_retry_counts": state.get("phase_retry_counts", {}),
        "current_phase": "analysis",
        "errors": errors + [warn],
        "messages": [AIMessage(content=warn)],
    }


def _handle_post_analysis(state: PipelineState) -> dict:
    """
    Validate analysis output. Retry analysis, retry research, or proceed to writing.
    """
    ok, reason = _validate_analysis(state)
    errors = list(state.get("errors", []))

    if ok:
        writing_instructions = _build_writing_instructions(state)
        coordinator_instructions = {
            **state.get("coordinator_instructions", {}),
            "writing": writing_instructions,
        }
        plogger.phase_end(
            "Coordinator[post_analysis]",
            f"Analysis validated — findings={len(state.get('key_findings', []))} | dispatching writing",
        )
        plogger.agent_decision("Coordinator", "dispatch_writing")
        plogger.coordinator_dispatch("Writing Agent", writing_instructions)
        return {
            "coordinator_instructions": coordinator_instructions,
            "coordinator_next_action": "writing",
            "phase_retry_counts": state.get("phase_retry_counts", {}),
            "current_phase": "writing",
            "errors": errors,
            "messages": [
                AIMessage(
                    content=(
                        f"Coordinator[post_analysis]: analysis validated "
                        f"({len(state.get('key_findings', []))} findings, "
                        f"quality={state.get('evidence_quality', 'unknown')}). "
                        "Dispatching Writing Agent."
                    )
                )
            ],
        }

    can_retry, updated_counts = _should_retry("analysis", state)
    if can_retry:
        retry_num = updated_counts.get("analysis", 1)
        warn = f"Coordinator[post_analysis]: analysis insufficient — retrying (attempt {retry_num}). {reason}"
        logger.warning(warn)
        plogger.agent_decision("Coordinator", "retry_analysis", f"attempt={retry_num}")
        return {
            "coordinator_next_action": "analysis",
            "phase_retry_counts": updated_counts,
            "current_phase": "analysis",
            "errors": errors + [warn],
            "messages": [AIMessage(content=warn)],
        }

    # Proceed degraded to writing
    warn = f"Coordinator[post_analysis]: max retries exhausted. Proceeding degraded. Last issue: {reason}"
    logger.warning(warn)
    writing_instructions = _build_writing_instructions(state)
    coordinator_instructions = {
        **state.get("coordinator_instructions", {}),
        "writing": writing_instructions,
    }
    plogger.agent_decision("Coordinator", "proceed_degraded_analysis")
    return {
        "coordinator_instructions": coordinator_instructions,
        "coordinator_next_action": "writing",
        "phase_retry_counts": state.get("phase_retry_counts", {}),
        "current_phase": "writing",
        "errors": errors + [warn],
        "messages": [AIMessage(content=warn)],
    }


def _handle_post_writing(state: PipelineState) -> dict:
    """
    Validate writing output. Retry writing or dispatch quality review.
    """
    ok, reason = _validate_writing(state)
    errors = list(state.get("errors", []))

    if ok:
        quality_instructions = (
            f"Review the draft report ({len(state.get('draft_report', ''))} chars, "
            f"{len(state.get('citations', []))} citations). "
            f"Check readability, completeness (required sections), grammar, relevancy to: "
            f"\"{state.get('request', '')}\" and verify medical/clinical claims against sources."
        )
        plogger.phase_end(
            "Coordinator[post_writing]",
            f"Writing validated — {len(state.get('draft_report', ''))} chars | dispatching quality",
        )
        plogger.agent_decision("Coordinator", "dispatch_quality")
        plogger.coordinator_dispatch("Quality Agent", quality_instructions)
        return {
            "coordinator_next_action": "quality",
            "phase_retry_counts": state.get("phase_retry_counts", {}),
            "current_phase": "quality",
            "errors": errors,
            "messages": [
                AIMessage(
                    content=(
                        f"Coordinator[post_writing]: draft validated "
                        f"({len(state.get('draft_report', ''))} chars, "
                        f"{len(state.get('citations', []))} citations). "
                        "Dispatching Quality Agent."
                    )
                )
            ],
        }

    can_retry, updated_counts = _should_retry("writing", state)
    if can_retry:
        retry_num = updated_counts.get("writing", 1)
        # Inject the failure reason into writing instructions
        existing = state.get("coordinator_instructions", {}).get("writing", "")
        coordinator_instructions = {
            **state.get("coordinator_instructions", {}),
            "writing": existing + f"\n\n[RETRY {retry_num}] Previous draft was rejected: {reason}. Produce a complete, detailed report.",
        }
        warn = f"Coordinator[post_writing]: draft insufficient — retrying (attempt {retry_num}). {reason}"
        logger.warning(warn)
        plogger.agent_decision("Coordinator", "retry_writing", f"attempt={retry_num}")
        return {
            "coordinator_instructions": coordinator_instructions,
            "coordinator_next_action": "writing",
            "phase_retry_counts": updated_counts,
            "current_phase": "writing",
            "errors": errors + [warn],
            "messages": [AIMessage(content=warn)],
        }

    warn = f"Coordinator[post_writing]: max retries exhausted. Proceeding to quality with degraded draft. {reason}"
    logger.warning(warn)
    plogger.agent_decision("Coordinator", "proceed_degraded_writing")
    return {
        "coordinator_next_action": "quality",
        "phase_retry_counts": state.get("phase_retry_counts", {}),
        "current_phase": "quality",
        "errors": errors + [warn],
        "messages": [AIMessage(content=warn)],
    }


def _handle_post_quality(state: PipelineState) -> dict:
    """
    Interpret quality review results.
    Route to complete, more research, or writing revision.
    Enforces max_iterations cap.
    """
    is_approved = state.get("is_approved", False)
    quality_score = state.get("quality_score", 0.0)
    quality_feedback = state.get("quality_feedback", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    errors = list(state.get("errors", []))

    # Force-complete if iteration cap reached
    if iteration_count >= max_iterations and not is_approved:
        note = f"Coordinator: force-completing after {max_iterations} revision cycles (score={quality_score:.2f})."
        logger.warning(note)
        plogger.agent_decision("Coordinator", "force_complete", f"iterations={iteration_count}")
        return {
            "is_approved": True,
            "coordinator_next_action": "complete",
            "current_phase": "complete",
            "errors": errors + [note],
            "messages": [AIMessage(content=note)],
        }

    if is_approved:
        plogger.phase_end("Coordinator[post_quality]", f"APPROVED — score={quality_score:.2f}")
        plogger.agent_decision("Coordinator", "complete", f"score={quality_score:.2f}")
        return {
            "coordinator_next_action": "complete",
            "current_phase": "complete",
            "errors": errors,
            "messages": [
                AIMessage(
                    content=(
                        f"Coordinator[post_quality]: report approved "
                        f"(score={quality_score:.2f}). Pipeline complete."
                    )
                )
            ],
        }

    # Determine whether revision needs more research or just rewriting
    research_signals = {
        "missing", "insufficient data", "more evidence",
        "lacks sources", "no citation", "not enough", "inadequate evidence",
    }
    needs_research = any(
        signal in fb.lower()
        for fb in quality_feedback
        for signal in research_signals
    )

    if needs_research:
        # Refresh research instructions with quality feedback context
        existing_research = state.get("coordinator_instructions", {}).get("research", "")
        gap_note = (
            "\n\n[QUALITY REVIEW] The report was rejected. Missing evidence areas:\n"
            + "\n".join(f"  • {fb}" for fb in quality_feedback if any(s in fb.lower() for s in research_signals))
        )
        coordinator_instructions = {
            **state.get("coordinator_instructions", {}),
            "research": existing_research + gap_note,
        }
        plogger.agent_decision("Coordinator", "needs_more_research", f"score={quality_score:.2f}")
        return {
            "coordinator_instructions": coordinator_instructions,
            "coordinator_next_action": "research",
            "current_phase": "research",
            "errors": errors,
            "messages": [
                AIMessage(
                    content=(
                        f"Coordinator[post_quality]: score={quality_score:.2f} — "
                        "needs more research. Re-dispatching Research Agent with gap analysis."
                    )
                )
            ],
        }

    # Needs writing revision — inject quality feedback into writing instructions
    existing_writing = state.get("coordinator_instructions", {}).get("writing", "")
    revision_note = (
        "\n\n[QUALITY REVIEW] Revision required. Address these issues:\n"
        + "\n".join(f"  • {fb}" for fb in quality_feedback)
    )
    coordinator_instructions = {
        **state.get("coordinator_instructions", {}),
        "writing": existing_writing + revision_note,
    }
    plogger.agent_decision("Coordinator", "needs_revision", f"score={quality_score:.2f}")
    return {
        "coordinator_instructions": coordinator_instructions,
        "coordinator_next_action": "writing",
        "current_phase": "writing",
        "errors": errors,
        "messages": [
            AIMessage(
                content=(
                    f"Coordinator[post_quality]: score={quality_score:.2f} — "
                    "needs revision. Re-dispatching Writing Agent with feedback."
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Main coordinator node
# ---------------------------------------------------------------------------

def coordinator_node(state: PipelineState) -> dict:
    """
    Phase-aware coordinator hub. Reads pipeline_phase to determine which
    agent just completed, validates output, and routes accordingly.
    """
    phase = state.get("pipeline_phase", "init")
    plogger.phase_start("Coordinator", f"pipeline_phase={phase}")

    handlers = {
        "init": _handle_init,
        "post_research": _handle_post_research,
        "post_analysis": _handle_post_analysis,
        "post_writing": _handle_post_writing,
        "post_quality": _handle_post_quality,
    }

    handler = handlers.get(phase)
    if handler is None:
        err = f"Coordinator received unknown pipeline_phase='{phase}'."
        logger.error(err)
        return {
            "coordinator_next_action": "surface_error",
            "surface_error": True,
            "pipeline_error_message": err,
            "errors": state.get("errors", []) + [err],
            "messages": [AIMessage(content=f"Coordinator: {err}")],
        }

    return handler(state)


# ---------------------------------------------------------------------------
# Router (used by LangGraph conditional edge)
# ---------------------------------------------------------------------------

def coordinator_router(state: PipelineState) -> str:
    """
    Read coordinator_next_action and return the graph routing target.

    Valid returns (must match pipeline.py conditional edge map):
        'research' | 'analysis' | 'writing' | 'quality' |
        'complete' | 'out_of_scope' | 'needs_clarification' | 'surface_error'
    """
    action = state.get("coordinator_next_action", "surface_error")
    logger.debug("coordinator_router: next_action=%s", action)
    return action
