"""LangChain callback handler that streams tool-call events to the SSE queue."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_PIPELINE_AGENTS = {"coordinator", "research", "analysis", "writing", "quality"}


class ToolStreamingCallback(BaseCallbackHandler):
    """Emits tool_call SSE events in real time as tools start and finish."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop
        self._lock = threading.Lock()
        self._tool_runs: dict[str, dict[str, str]] = {}  # run_id -> {agent, tool}

    def _put(self, data: dict) -> None:
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, {"event": "tool_call", "data": data}
        )

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: Any,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = next((t for t in (tags or []) if t in _PIPELINE_AGENTS), "unknown")
        tool_name = serialized.get("name", "unknown")
        rid = str(run_id)
        with self._lock:
            self._tool_runs[rid] = {"agent": agent, "tool": tool_name}
        self._put({
            "agent": agent,
            "tool": tool_name,
            "input": str(input_str)[:800],
            "phase": "start",
            "run_id": rid,
        })

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        with self._lock:
            info = self._tool_runs.pop(rid, {"agent": "unknown", "tool": "unknown"})
        self._put({
            "agent": info["agent"],
            "tool": info["tool"],
            "output": str(output)[:1500] if output else "",
            "phase": "end",
            "run_id": rid,
        })
