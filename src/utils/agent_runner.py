"""Custom ReAct agent runner with forced first-turn tool use."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage as _HumanMsg
from langchain_core.messages import SystemMessage as _SystemMsg
from langchain_core.messages import ToolMessage as _ToolMsg

from .logging_utils import console, log_tool_call, log_tool_result


def run_agent_with_forced_tools(
    llm: Any,
    tools: list,
    system_prompt: str,
    user_message: str,
    agent_name: str,
    max_iterations: int = 15,
) -> dict:
    """Custom ReAct loop that guarantees at least one tool call.

    Uses ``tool_choice='any'`` on the first LLM turn so Gemini cannot skip tools
    by returning JSON directly.  Subsequent turns use ``tool_choice='auto'`` so
    the model can decide when to stop calling tools and produce the final answer.

    Returns a dict shaped like ``agent.invoke()``: ``{"messages": [...]}``.
    """
    agent_log = logging.getLogger(f"tool.{agent_name}")
    tool_map = {t.name: t for t in tools}

    llm_forced = llm.bind_tools(tools, tool_choice="any")
    llm_auto = llm.bind_tools(tools)

    user_msg = _HumanMsg(content=user_message)
    messages: list = [_SystemMsg(content=system_prompt), user_msg]
    tool_call_count = 0
    first_call = True

    for _ in range(max_iterations):
        response = (llm_forced if first_call else llm_auto).invoke(messages)
        first_call = False
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for tc in response.tool_calls:
            tool_call_count += 1
            args_preview = log_tool_call(agent_name, tc["name"], tc.get("args", ""))
            agent_log.info("[%s] tool_call: %s | %s", agent_name, tc["name"], args_preview)

            tool_fn = tool_map.get(tc["name"])
            tool_result = tool_fn.invoke(tc["args"]) if tool_fn else f"Tool '{tc['name']}' not found"
            result_str = str(tool_result)

            tool_msg = _ToolMsg(content=result_str, name=tc["name"], tool_call_id=tc["id"])
            messages.append(tool_msg)

            preview = log_tool_result(agent_name, tc["name"], result_str)
            agent_log.info("[%s] tool_result: %s | %s", agent_name, tc["name"], preview)

    if tool_call_count == 0:
        console.print(
            f"  [yellow dim][{agent_name}] No tool calls — completed in single LLM pass[/yellow dim]"
        )

    # Return everything after the SystemMessage (user message + all agent turns)
    return {"messages": messages[1:]}
