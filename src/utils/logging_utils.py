"""Logging and observability utilities for the Content Intelligence Pipeline."""

import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage as _AIMsg
from langchain_core.messages import ToolMessage as _ToolMsg
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text


class _StreamingConsole(Console):
    """Console that simultaneously streams output to an optional plain-text log file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_fh = None
        self._file_console: Console | None = None

    def set_log_file(self, path: str) -> None:
        """Open (or replace) the log file and create a no-colour file console."""
        if self._log_fh:
            self._log_fh.close()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(path, "w", encoding="utf-8")  # noqa: SIM115
        self._file_console = Console(
            file=self._log_fh,
            highlight=False,
            no_color=True,
            markup=True,       # strip markup tags, keep text
            width=120,
        )

    def print(self, *args, **kwargs):  # type: ignore[override]
        super().print(*args, **kwargs)
        if self._file_console is not None:
            self._file_console.print(*args, **kwargs)
            if self._log_fh:
                self._log_fh.flush()

    def close_log_file(self) -> None:
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
            self._file_console = None


console = _StreamingConsole()


def setup_log_file(path: str) -> None:
    """Open a streaming log file. All console.print() output is tee'd there immediately."""
    console.set_log_file(path)


def flush_log_to_file() -> None:
    """Close the log file (all content has already been streamed)."""
    console.close_log_file()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a Rich-enhanced logger for the given module name."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    return logging.getLogger(name)


class PipelineLogger:
    """High-level observability helper used by agent nodes."""

    def __init__(self, name: str = "pipeline"):
        self.logger = get_logger(name)
        self._phase_start: float = 0.0

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def phase_start(self, phase: str, details: str = "") -> None:
        self._phase_start = time.time()
        msg = f"[bold cyan]>> PHASE START:[/bold cyan] {phase}"
        if details:
            msg += f" — {details}"
        console.print(Panel(Text.from_markup(msg), border_style="cyan"))
        self.logger.info("Phase started: %s", phase)

    def phase_end(self, phase: str, summary: str = "") -> None:
        elapsed = time.time() - self._phase_start
        msg = (
            f"[bold green]<< PHASE END:[/bold green] {phase} "
            f"([yellow]{elapsed:.2f}s[/yellow])"
        )
        if summary:
            msg += f"\n   {summary}"
        console.print(Panel(Text.from_markup(msg), border_style="green"))
        self.logger.info("Phase finished: %s (%.2fs)", phase, elapsed)

    def phase_error(self, phase: str, error: str) -> None:
        msg = f"[bold red]!! PHASE ERROR:[/bold red] {phase}\n   {error}"
        console.print(Panel(Text.from_markup(msg), border_style="red"))
        self.logger.error("Phase error in %s: %s", phase, error)

    # ------------------------------------------------------------------
    # Tool call tracing
    # ------------------------------------------------------------------

    def tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        self.logger.debug("Tool called: %s | args=%s", tool_name, args)

    def tool_result(self, tool_name: str, result_preview: str) -> None:
        preview = result_preview[:120].replace("\n", " ")
        self.logger.debug("Tool result: %s | %s...", tool_name, preview)

    # ------------------------------------------------------------------
    # Agent decision logging
    # ------------------------------------------------------------------

    def agent_decision(self, agent: str, decision: str, reason: str = "") -> None:
        self.logger.info("[%s] Decision: %s%s", agent, decision, f" — {reason}" if reason else "")

    def coordinator_dispatch(self, target_agent: str, instructions: str) -> None:
        preview = instructions[:400]
        if len(instructions) > 400:
            preview += f"\n  ... [{len(instructions)} total chars]"
        msg = (
            f"[bold blue]→ DISPATCH → {target_agent}[/bold blue]\n"
            + "\n".join(f"   {line}" for line in preview.splitlines())
        )
        console.print(Panel(Text.from_markup(msg), border_style="blue"))
        self.logger.info("Dispatch to %s: %s", target_agent, instructions[:200].replace("\n", " "))

    def content_preview(self, label: str, content: str, max_chars: int = 600) -> None:
        preview = content[:max_chars]
        suffix = f"\n  ... [{len(content)} total chars]" if len(content) > max_chars else ""
        console.print(Panel(
            f"{preview}{suffix}",
            title=f"[bold yellow]{label}[/bold yellow]",
            border_style="yellow",
        ))

    def quality_result(self, score: float, approved: bool, feedback: list[str]) -> None:
        status = "[bold green]APPROVED[/bold green]" if approved else "[bold yellow]NEEDS REVISION[/bold yellow]"
        lines = [f"Quality Score: {score:.2f} | Status: {status}"]
        for item in feedback:
            lines.append(f"  - {item}")
        console.print(Panel("\n".join(lines), title="Quality Assessment", border_style="magenta"))


class ToolLoggingCallback(BaseCallbackHandler):
    """Logs tool calls to the terminal with Rich formatting (server-side observability)."""

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self._agent = agent_name
        self._logger = logging.getLogger(f"tool.{agent_name}")

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        preview = str(input_str)[:300].replace("\n", " ")
        console.print(
            f"  [cyan]→ TOOL START[/cyan]  [bold magenta][{self._agent}][/bold magenta] "
            f"[bold]{tool_name}[/bold]  [dim]{preview}[/dim]"
        )
        self._logger.info("[%s] tool_start: %s | input: %s", self._agent, tool_name, preview)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        preview = (str(output)[:300].replace("\n", " ")) if output else "(empty)"
        console.print(
            f"  [green]← TOOL END[/green]    [bold magenta][{self._agent}][/bold magenta] "
            f"[dim]{preview}[/dim]"
        )
        self._logger.info("[%s] tool_end: %s", self._agent, preview)

    def on_tool_error(
        self,
        error: Any,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        console.print(
            f"  [bold red]✗ TOOL ERROR[/bold red]  [bold magenta][{self._agent}][/bold magenta] "
            f"{error}"
        )
        self._logger.error("[%s] tool_error: %s", self._agent, error)



def log_tool_call(agent_name: str, tool_name: str, args: Any) -> str:
    """Print a formatted tool-call line and return the args preview string."""
    preview = str(args)[:200].replace("\n", " ")
    console.print(
        f"  [cyan]→ TOOL[/cyan]  [bold magenta][{agent_name}][/bold magenta]  "
        f"[bold]{tool_name}[/bold]  [dim]{preview}[/dim]"
    )
    return preview


def log_tool_result(agent_name: str, tool_name: str, result: str) -> str:
    """Print a formatted tool-result line and return the result preview string."""
    preview = result[:200].replace("\n", " ")
    console.print(
        f"  [green]← RESULT[/green]  [bold magenta][{agent_name}][/bold magenta]  "
        f"[bold]{tool_name}[/bold]  [dim]{preview}[/dim]"
    )
    return preview


def invoke_agent_with_tool_logging(
    agent: Any,
    input_messages: dict,
    config: dict,
    agent_name: str,
) -> dict:
    """Invoke a create_react_agent and log every tool call/result (used by Quality agent)."""
    _log = logging.getLogger(f"tool.{agent_name}")
    result = agent.invoke(input_messages, config=config)

    messages = result.get("messages", [])
    tool_call_count = 0

    for msg in messages:
        if isinstance(msg, _AIMsg) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_call_count += 1
                args_preview = log_tool_call(agent_name, tc["name"], tc.get("args", ""))
                _log.info("[%s] tool_call: %s | %s", agent_name, tc["name"], args_preview)
        elif isinstance(msg, _ToolMsg):
            preview = log_tool_result(agent_name, msg.name, str(msg.content))
            _log.info("[%s] tool_result: %s | %s", agent_name, msg.name, preview)

    if tool_call_count == 0:
        msg_types = ", ".join(type(m).__name__ for m in messages)
        console.print(
            f"  [yellow dim][{agent_name}] No tool calls detected — agent completed in "
            f"a single LLM pass. Message chain: [{msg_types}][/yellow dim]"
        )

    return result
