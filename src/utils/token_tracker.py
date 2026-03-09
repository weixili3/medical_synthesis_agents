"""LangChain callback handler for tracking token usage and estimating costs."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Gemini 2.0 Flash pricing (USD per token, standard context window)
_INPUT_PRICE_PER_TOKEN = 0.075 / 1_000_000
_OUTPUT_PRICE_PER_TOKEN = 0.30 / 1_000_000

# Node names in the pipeline graph
_PIPELINE_AGENTS = {"coordinator", "research", "analysis", "writing", "quality"}


class TokenTrackingCallback(BaseCallbackHandler):
    """Accumulates token usage per agent and estimates cost."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.by_agent: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0, "calls": 0}
        )
        self._run_agent: dict[str, str] = {}  # run_id -> agent name

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: Any,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = next((t for t in (tags or []) if t in _PIPELINE_AGENTS), "unknown")
        self._run_agent[str(run_id)] = agent

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        agent = self._run_agent.pop(str(run_id), "unknown")
        input_tokens, output_tokens = 0, 0

        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg:
                    usage = getattr(msg, "usage_metadata", None) or {}
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)

        # Fallback: check llm_output dict
        if input_tokens == 0 and output_tokens == 0 and response.llm_output:
            usage = response.llm_output.get("usage_metadata") or response.llm_output.get(
                "token_usage", {}
            )
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

        with self._lock:
            self.by_agent[agent]["input"] += input_tokens
            self.by_agent[agent]["output"] += output_tokens
            self.by_agent[agent]["calls"] += 1

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            agents = {k: dict(v) for k, v in self.by_agent.items()}

        total_input = sum(v["input"] for v in agents.values())
        total_output = sum(v["output"] for v in agents.values())
        cost = total_input * _INPUT_PRICE_PER_TOKEN + total_output * _OUTPUT_PRICE_PER_TOKEN

        return {
            "by_agent": agents,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "estimated_cost_usd": round(cost, 6),
        }
