"""Main LangGraph pipeline — assembles and compiles the Content Intelligence Pipeline.

Graph topology (hub-and-spoke):

  START
    │
    ▼
  coordinator ◄────────────────────────────────────────────────────────┐
    │                                                                    │
    ├─ out_of_scope        ──► END                                       │
    ├─ needs_clarification ──► END                                       │
    ├─ surface_error       ──► END                                       │
    ├─ complete            ──► END                                       │
    │                                                                    │
    ├─ research   ──► research_agent  ── always ──────────────────────► coordinator
    ├─ analysis   ──► analysis_agent  ── always ──────────────────────► coordinator
    ├─ writing    ──► writing_agent   ── always ──────────────────────► coordinator
    └─ quality    ──► quality_agent   ── always ──────────────────────► coordinator

The coordinator is phase-aware: it reads pipeline_phase (set by each agent)
to know which agent just reported, validates the output, enriches instructions
for the next agent, and sets coordinator_next_action to drive routing.
"""

import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .agents.coordinator import coordinator_node, coordinator_router
from .agents.research_agent import research_node
from .agents.analysis_agent import analysis_node
from .agents.writing_agent import writing_node
from .agents.quality_agent import quality_node
from .state.pipeline_state import PipelineState

logger = logging.getLogger(__name__)


def build_pipeline(checkpointer: Any = None):
    """
    Construct and compile the LangGraph StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence /
                      human-in-the-loop support.  Defaults to MemorySaver.

    Returns:
        A compiled LangGraph runnable (CompiledGraph).
    """
    workflow = StateGraph(PipelineState)

    # ------------------------------------------------------------------ nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("writing", writing_node)
    workflow.add_node("quality", quality_node)

    # ------------------------------------------------------------------ edges
    # Entry point
    workflow.add_edge(START, "coordinator")

    # Coordinator is the sole decision-maker — fans out to each agent
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_router,
        {
            "research": "research",
            "analysis": "analysis",
            "writing": "writing",
            "quality": "quality",
            "complete": END,
            "out_of_scope": END,
            "needs_clarification": END,
            "surface_error": END,
        },
    )

    # All agents always report back to coordinator (hub-and-spoke)
    workflow.add_edge("research", "coordinator")
    workflow.add_edge("analysis", "coordinator")
    workflow.add_edge("writing", "coordinator")
    workflow.add_edge("quality", "coordinator")

    cp = checkpointer or MemorySaver()
    graph = workflow.compile(checkpointer=cp)
    logger.info("Pipeline compiled successfully.")

    _save_graph_diagram(graph)

    return graph


def _save_graph_diagram(graph) -> None:
    """Save a PNG diagram of the compiled graph to docs/pipeline_graph.png."""
    try:
        docs_dir = Path(__file__).parent.parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        output_path = docs_dir / "pipeline_graph.png"
        png_bytes = graph.get_graph().draw_mermaid_png()
        output_path.write_bytes(png_bytes)
        logger.info("Pipeline graph saved to %s", output_path)
    except Exception as exc:
        # Non-fatal: diagram generation may fail if Mermaid API is unreachable
        logger.warning("Could not save pipeline graph diagram: %s", exc)


def run_pipeline(
    request: str,
    max_iterations: int = 3,
    max_retries_per_phase: int = 2,
    thread_id: str = "default",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute the full pipeline for a given content request.

    Args:
        request:               The content intelligence request.
        max_iterations:        Maximum writing cycles before coordinator forces completion.
        max_retries_per_phase: How many times coordinator retries a failing phase.
        thread_id:             Checkpointer thread identifier.
        config:                Optional extra LangGraph run config.

    Returns:
        The final pipeline state dict.  Key fields to inspect:
          out_of_scope          — request rejected at scope gate
          clarification_needed  — request was too vague
          clarification_question— follow-up question for the user
          surface_error         — unrecoverable pipeline error
          pipeline_error_message— description of that error
          draft_report          — the final report (when successful)
          quality_score         — composite quality score
          is_approved           — whether the report passed quality review
    """
    graph = build_pipeline()

    initial_state: dict[str, Any] = {
        "request": request,
        "max_iterations": max_iterations,
        "max_retries_per_phase": max_retries_per_phase,
        "pipeline_phase": "init",
        "messages": [],
        # All other fields initialised by coordinator
    }

    run_config = {"configurable": {"thread_id": thread_id}, **(config or {})}

    logger.info("Pipeline started for request: %s", request[:80])
    final_state = graph.invoke(initial_state, config=run_config)

    if final_state.get("out_of_scope"):
        logger.info("Pipeline exited: out of scope — %s", final_state.get("scope_rejection_reason"))
    elif final_state.get("clarification_needed"):
        logger.info("Pipeline exited: clarification needed — %s", final_state.get("clarification_question"))
    elif final_state.get("surface_error"):
        logger.error("Pipeline exited: fatal error — %s", final_state.get("pipeline_error_message"))
    else:
        logger.info(
            "Pipeline completed. Quality score: %.2f | Approved: %s",
            final_state.get("quality_score", 0),
            final_state.get("is_approved", False),
        )

    return final_state


def stream_pipeline(
    request: str,
    max_iterations: int = 3,
    max_retries_per_phase: int = 2,
    thread_id: str = "default",
):
    """
    Stream pipeline events (useful for real-time UIs / monitoring).

    Yields LangGraph event dicts as the pipeline progresses through nodes.
    Because all agents report to coordinator, events alternate between
    coordinator and agent nodes, making the orchestration visible in the stream.

    Args:
        request:               The content intelligence request.
        max_iterations:        Maximum writing cycles.
        max_retries_per_phase: Per-phase retry cap for coordinator.
        thread_id:             Checkpointer thread identifier.

    Yields:
        LangGraph event dicts with keys 'event', 'name', 'data', etc.
    """
    graph = build_pipeline()

    initial_state: dict[str, Any] = {
        "request": request,
        "max_iterations": max_iterations,
        "max_retries_per_phase": max_retries_per_phase,
        "pipeline_phase": "init",
        "messages": [],
    }

    run_config = {"configurable": {"thread_id": thread_id}}

    for event in graph.stream(initial_state, config=run_config, stream_mode="updates"):
        yield event
