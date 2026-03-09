"""API routes for the Research Agents pipeline.

Endpoints:
  POST /api/run              — start a pipeline run, returns thread_id
  GET  /api/stream/{id}      — SSE stream of real-time agent events
  GET  /api/result/{id}      — fetch the final result after completion
  GET  /api/health           — liveness check
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver

from src.pipeline import build_pipeline
from src.utils.logging_utils import flush_log_to_file, setup_log_file
from src.utils.token_tracker import TokenTrackingCallback
from src.utils.streaming_callback import ToolStreamingCallback
from .models import RunRequest, RunResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Singleton graph — shared across all requests so the MemorySaver persists
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_pipeline(checkpointer=_checkpointer)
    return _graph


# ---------------------------------------------------------------------------
# Per-run state
# ---------------------------------------------------------------------------

_queues: dict[str, asyncio.Queue] = {}
_results: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pipeline")


# ---------------------------------------------------------------------------
# Stream event helpers
# ---------------------------------------------------------------------------


def _process_update(update: dict, tracker: TokenTrackingCallback) -> dict | None:
    """Convert a LangGraph stream-update dict into an SSE event payload."""
    if not update:
        return None

    node = next(iter(update))
    data = update[node]
    token_summary = tracker.get_summary()

    if node == "coordinator":
        return {
            "type": "coordinator_decision",
            "node": node,
            "phase": data.get("pipeline_phase", ""),
            "action": data.get("coordinator_next_action", ""),
            "out_of_scope": data.get("out_of_scope", False),
            "scope_rejection_reason": data.get("scope_rejection_reason", ""),
            "clarification_needed": data.get("clarification_needed", False),
            "clarification_question": data.get("clarification_question", ""),
            "surface_error": data.get("surface_error", False),
            "token_summary": token_summary,
        }

    if node == "research":
        return {
            "type": "agent_complete",
            "node": node,
            "agent": "research",
            "research_queries": data.get("research_queries", []),
            "source_count": len(data.get("raw_sources", [])),
            "search_summary": data.get("search_summary", {}),
            "research_summary_excerpt": data.get("research_summary") or "",
            "errors": data.get("errors", []),
            "token_summary": token_summary,
        }

    if node == "analysis":
        return {
            "type": "agent_complete",
            "node": node,
            "agent": "analysis",
            "key_findings": data.get("key_findings", []),
            "evidence_quality": data.get("evidence_quality", ""),
            "evidence_grade": data.get("evidence_grade", ""),
            "bias_assessment": data.get("bias_assessment", ""),
            "statistical_summary": data.get("statistical_summary", {}),
            "errors": data.get("errors", []),
            "token_summary": token_summary,
        }

    if node == "writing":
        report = data.get("draft_report", "")
        return {
            "type": "agent_complete",
            "node": node,
            "agent": "writing",
            "report_chars": len(report),
            "citation_count": len(data.get("citations", [])),
            "errors": data.get("errors", []),
            "token_summary": token_summary,
        }

    if node == "quality":
        return {
            "type": "agent_complete",
            "node": node,
            "agent": "quality",
            "quality_score": data.get("quality_score", 0.0),
            "is_approved": data.get("is_approved", False),
            "quality_feedback": data.get("quality_feedback", []),
            "errors": data.get("errors", []),
            "token_summary": token_summary,
        }

    return {"type": "node_update", "node": node, "token_summary": token_summary}


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------


def _run_pipeline_thread(
    thread_id: str,
    question: str,
    max_iterations: int,
    max_retries_per_phase: int,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run the pipeline in a background thread, pushing events to the SSE queue."""
    # Set up per-run log file under examples/logs/YYYY-MM-DD/
    now = datetime.now()
    _repo_root = Path(__file__).parent.parent.parent
    _log_dir = _repo_root / "examples" / "logs" / now.strftime("%Y-%m-%d")
    _log_path = _log_dir / f"{now.strftime('%H-%M-%S')}-{thread_id[:8]}.log"
    setup_log_file(str(_log_path))

    tracker = TokenTrackingCallback()
    tool_streamer = ToolStreamingCallback(queue=queue, loop=loop)
    graph = _get_graph()

    initial_state = {
        "request": question,
        "max_iterations": max_iterations,
        "max_retries_per_phase": max_retries_per_phase,
        "pipeline_phase": "init",
        "messages": [],
    }
    run_config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracker, tool_streamer],
    }

    def _put(event: dict | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        for update in graph.stream(initial_state, config=run_config, stream_mode="updates"):
            event = _process_update(update, tracker)
            if event:
                _put({"event": "update", "data": event})

        # Retrieve full final state after stream exhausts
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        final: dict[str, Any] = snapshot.values if snapshot else {}

    except Exception as exc:
        logger.error("Pipeline error in thread %s: %s", thread_id, exc, exc_info=True)
        _put({"event": "pipeline_error", "data": {"message": str(exc)}})
        _put(None)  # sentinel
        flush_log_to_file()
        return

    token_summary = tracker.get_summary()
    _results[thread_id] = {**final, "token_summary": token_summary}

    _put({
        "event": "pipeline_complete",
        "data": {
            "draft_report": final.get("draft_report", ""),
            "citations": final.get("citations", []),
            "quality_score": final.get("quality_score", 0.0),
            "is_approved": final.get("is_approved", False),
            "key_findings": final.get("key_findings", []),
            "evidence_quality": final.get("evidence_quality", ""),
            "evidence_grade": final.get("evidence_grade", ""),
            "out_of_scope": final.get("out_of_scope", False),
            "scope_rejection_reason": final.get("scope_rejection_reason", ""),
            "clarification_needed": final.get("clarification_needed", False),
            "clarification_question": final.get("clarification_question", ""),
            "surface_error": final.get("surface_error", False),
            "pipeline_error_message": final.get("pipeline_error_message", ""),
            "errors": final.get("errors", []),
            "token_summary": token_summary,
        },
    })
    _put(None)  # sentinel — closes the SSE stream
    flush_log_to_file()


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------


async def _sse_generator(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=600)
        except asyncio.TimeoutError:
            yield "event: timeout\ndata: {}\n\n"
            break

        if item is None:
            yield "event: stream_end\ndata: {}\n\n"
            break

        yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/run", response_model=RunResponse)
async def start_run(body: RunRequest) -> RunResponse:
    thread_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _queues[thread_id] = queue

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline_thread,
        thread_id,
        body.question,
        body.max_iterations,
        body.max_retries_per_phase,
        queue,
        loop,
    )

    return RunResponse(thread_id=thread_id)


@router.get("/stream/{thread_id}")
async def stream_events(thread_id: str) -> StreamingResponse:
    queue = _queues.get(thread_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return StreamingResponse(
        _sse_generator(queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/result/{thread_id}")
async def get_result(thread_id: str) -> dict:
    result = _results.get(thread_id)
    if result is None:
        if thread_id in _queues:
            return {"status": "pending"}
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "complete", **result}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
