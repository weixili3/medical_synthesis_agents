"""Entry point for the Content Intelligence Pipeline.

Usage:
    python main.py                              # interactive prompt
    python main.py "Your research request"      # single request from CLI
    python main.py --stream "Your request"      # stream node-by-node progress
    python main.py --output report.md "..."     # save final report to file
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables before importing pipeline modules
load_dotenv()

from src.pipeline import run_pipeline, stream_pipeline
from src.utils.logging_utils import console, flush_log_to_file, get_logger, setup_log_file
logger = get_logger("main", level=os.getenv("LOG_LEVEL", "INFO"))


def _init_log_file() -> str:
    """Create a dated log file under examples/logs/YYYY-MM-DD/ and return its path."""
    now = datetime.now()
    repo_root = Path(__file__).parent
    log_dir = repo_root / "examples" / "logs" / now.strftime("%Y-%m-%d")
    log_path = log_dir / f"{now.strftime('%H-%M-%S')}.log"
    setup_log_file(str(log_path))
    return str(log_path)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-intelligence-pipeline",
        description="Multi-agent content intelligence pipeline powered by Google Gemini + LangGraph.",
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="The content request to process. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream pipeline events as they occur instead of waiting for the final result.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save the final report to this file path (markdown).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        metavar="N",
        help="Maximum quality-revision cycles (default: 3).",
    )
    parser.add_argument(
        "--thread-id",
        default="main",
        help="LangGraph checkpointer thread ID for multi-turn resumption.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print the full final state as JSON instead of rendering the report.",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_env() -> list[str]:
    """Check for required environment variables; return list of missing ones."""
    required = ["GOOGLE_API_KEY"]
    return [var for var in required if not os.getenv(var)]


def _save_report(content: str, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    console.print(f"[green]Report saved to:[/green] {output_path.resolve()}")


def _print_report(final_state: dict) -> None:
    report = final_state.get("draft_report", "")
    score = final_state.get("quality_score", 0.0)
    approved = final_state.get("is_approved", False)
    iterations = final_state.get("iteration_count", 0)
    errors = final_state.get("errors", [])

    console.print(
        Panel(
            f"Quality Score: [bold]{score:.2f}[/bold] | "
            f"Approved: [bold]{'YES' if approved else 'NO'}[/bold] | "
            f"Revisions: {iterations}",
            title="Pipeline Summary",
            border_style="blue",
        )
    )

    if errors:
        console.print("[yellow]Warnings / Errors:[/yellow]")
        for err in errors:
            console.print(f"  [yellow]-[/yellow] {err}")

    if report:
        console.print("\n")
        console.print(Markdown(report))
    else:
        console.print("[red]No report was generated.[/red]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Validate environment
    missing = _validate_env()
    if missing:
        console.print(
            f"[red]Missing required environment variables: {', '.join(missing)}[/red]\n"
            "Copy [bold].env.example[/bold] to [bold].env[/bold] and fill in the values."
        )
        return 1

    # Set up dated log file before anything else runs
    log_path = _init_log_file()
    console.print(f"[dim]Logging to: {log_path}[/dim]")

    # Resolve request
    request = args.request
    if not request:
        console.print("[bold cyan]Content Intelligence Pipeline[/bold cyan]")
        console.print("Enter your content request (or press Ctrl+C to exit):\n")
        try:
            request = input("> ").strip()
        except KeyboardInterrupt:
            console.print("\nAborted.")
            return 0

    if not request:
        console.print("[red]No request provided.[/red]")
        return 1

    console.print(
        Panel(
            f"[bold]{request}[/bold]",
            title="Processing Request",
            border_style="cyan",
        )
    )

    # ---- Streaming mode ----
    if args.stream:
        console.print("[dim]Streaming pipeline events...[/dim]\n")
        final_state: dict = {}
        for event in stream_pipeline(request, max_iterations=args.max_iterations, thread_id=args.thread_id):
            for node_name, node_state in event.items():
                phase = node_state.get("current_phase", "")
                console.print(f"[cyan]Node:[/cyan] [bold]{node_name}[/bold]  phase→{phase}")
            final_state = node_state  # last state

    # ---- Batch mode ----
    else:
        final_state = run_pipeline(
            request,
            max_iterations=args.max_iterations,
            thread_id=args.thread_id,
        )

    # ---- Output ----
    if args.output_json:
        # Exclude messages for cleaner JSON output
        printable = {k: v for k, v in final_state.items() if k != "messages"}
        console.print_json(json.dumps(printable, indent=2, default=str))
    else:
        _print_report(final_state)

    if args.output and final_state.get("draft_report"):
        _save_report(final_state["draft_report"], args.output)

    flush_log_to_file()
    console.print(f"[dim]Log saved: {log_path}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
