"""Shared state definition for the Content Intelligence Pipeline."""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class PipelineState(TypedDict):
    """
    Central state object that flows through the entire pipeline.

    The coordinator is the hub. It reads this state after every agent
    completes, validates the output, enriches per-agent instructions, and
    writes coordinator_next_action to tell the router where to go next.
    """

    # ---- Input ----------------------------------------------------------------
    request: str
    """The original user content request."""

    content_type: str
    """Detected content type (e.g. 'clinical_evidence', 'market_report')."""

    # ---- Coordinator Orchestration --------------------------------------------
    pipeline_phase: str
    """
    Tracks which agent just completed so the coordinator knows its context.
    Values: 'init' | 'post_research' | 'post_analysis' | 'post_writing' | 'post_quality'
    Set by each agent node in its return state; read by coordinator_node.
    """

    coordinator_instructions: dict[str, Any]
    """
    Per-agent tailored guidance written by the coordinator.
    Keys: 'research', 'analysis', 'writing', 'focus_areas', 'key_questions'.
    Agents read their key and incorporate it into their LLM prompt.
    """

    phase_retry_counts: dict[str, int]
    """Per-phase retry counter. Keys match pipeline_phase values (e.g. 'research')."""

    max_retries_per_phase: int
    """Maximum retries the coordinator will attempt for any single phase."""

    coordinator_next_action: str
    """
    Routing signal set by coordinator_node and read by coordinator_router.
    Values: 'research' | 'analysis' | 'writing' | 'quality' |
            'complete' | 'out_of_scope' | 'needs_clarification' | 'surface_error'
    """

    surface_error: bool
    """True when the coordinator aborts the pipeline due to an unrecoverable error."""

    pipeline_error_message: str
    """Human-readable description of the fatal error when surface_error is True."""

    # ---- Coordinator Gate -----------------------------------------------------
    out_of_scope: bool
    """True when the coordinator rejected the request as outside the medical domain."""

    scope_rejection_reason: str
    """Human-readable explanation of why the request was considered out of scope."""

    clarification_needed: bool
    """True when the coordinator determined the request is too vague to proceed."""

    clarification_question: str
    """Specific question the coordinator wants the user to answer before retrying."""

    # ---- Research Phase -------------------------------------------------------
    research_queries: list[str]
    """Search queries generated and executed by the Research Agent."""

    raw_sources: list[dict[str, Any]]
    """Raw source data: [{title, url, snippet, full_text, relevance_score}]."""

    search_summary: dict[str, Any]
    """
    Structured summary of the research search results.
    Contains total_sources, by_study_type counts, by_quality counts,
    and a human-readable summary_message (e.g. "47 studies: 12 RCTs, ...").
    """

    research_summary: str
    """Prose summary of all gathered research material."""

    # ---- Analysis Phase -------------------------------------------------------
    key_findings: list[str]
    """Bullet-point key findings extracted from the research."""

    statistical_summary: dict[str, Any]
    """Structured statistics: counts, averages, confidence intervals, etc."""

    evidence_quality: str
    """Overall evidence quality rating: 'strong' | 'moderate' | 'weak'."""

    evidence_grade: str
    """GRADE evidence level: 'A' (high) | 'B' (moderate) | 'C' (low) | 'D' (very low)."""

    bias_assessment: str
    """Aggregate risk-of-bias rating across included studies: 'low' | 'moderate' | 'high' | 'unclear'."""

    intervention_categories: dict[str, Any]
    """
    Studies grouped by intervention type.
    Keys typically include: remote_monitoring, video_consultations, mobile_apps, other.
    Each value is a list of finding strings for that category.
    """

    outcome_measures: dict[str, Any]
    """
    Structured outcome data keyed by measure name (e.g. 'hba1c', 'medication_adherence').
    Each value contains a summary string and, where available, effect size information.
    """

    study_limitations: list[str]
    """Identified methodological limitations and potential biases across included studies."""

    clinical_implications: str
    """Prose paragraph summarising the clinical significance and practical implications."""

    # ---- Writing Phase --------------------------------------------------------
    draft_report: str
    """Full markdown-formatted draft report produced by the Writing Agent."""

    citations: list[str]
    """Formatted citations / references list."""

    # ---- Quality Phase --------------------------------------------------------
    quality_score: float
    """Composite quality score in [0, 1]."""

    quality_feedback: list[str]
    """Specific actionable feedback items from the Quality Agent."""

    is_approved: bool
    """True when the quality threshold has been met and the pipeline can exit."""

    # ---- Control Flow ---------------------------------------------------------
    iteration_count: int
    """Number of writing cycles completed (incremented by the writing agent)."""

    max_iterations: int
    """Hard cap on revision cycles before coordinator forces completion."""

    current_phase: str
    """Human-readable label of the currently executing phase (for observability)."""

    errors: list[str]
    """Accumulated non-fatal errors / warnings from all agents."""

    # ---- Observability --------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]
    """Chronological log of all agent messages for traceability."""
