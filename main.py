"""CLI for the Agentic Loan Pre-Assessment demo.

  python main.py                      interactive menu
  python main.py list                 show sample applicants
  python main.py assess A002 --amount 12000 --term 36 --purpose "debt consolidation"
  python main.py demo                 run all four preset scenarios
  add --mock to any command to run offline without an OpenAI key
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import ConfigError, load_settings
from app.data import list_applicants
from app.llm import build_llm
from app.logging_setup import setup_logging
from app.orchestrator import AssessmentReport, LoanApplication, LoanOrchestrator
from app.tools import build_registry

console = Console()
log = logging.getLogger("cli")

SCENARIOS = [
    ("A001", 20000, 36, "car purchase", "Strong profile: expect a straight approval"),
    ("A002", 12000, 36, "debt consolidation", "Gambling flag: Underwriter asks the Analyst for more"),
    ("A003", 15000, 48, "personal", "Recent default and overdrafts: expect a decline"),
    ("A004", 15000, 48, "home studio equipment", "Self-employed, irregular income: expect conditions"),
]
COLOURS = {"Analyst": "cyan", "Underwriter": "magenta", "Orchestrator": "yellow"}
DECISION_STYLE = {"APPROVE": "bold green", "CONDITIONAL_APPROVE": "bold yellow",
                  "DECLINE": "bold red", "REFER_TO_HUMAN": "bold blue"}


def _short(value, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def render_event(kind: str, payload: dict, verbose: bool) -> None:
    """Live view of the agentic loop: who is thinking, which tool, what came back."""
    agent = payload.get("agent", "?")
    tag = f"[{COLOURS.get(agent, 'white')}]{agent:<12}[/]"
    if kind == "stage":
        console.rule(f"[yellow]{payload['text']}")
    elif kind == "agent_start":
        console.print(f"{tag} received task")
        if verbose:
            console.print(f"             [dim]{_short(payload['task'], 400)}[/]")
    elif kind == "thought":
        console.print(f"{tag} [italic]reasoning:[/] {payload['text']}")
    elif kind == "tool_call":
        console.print(f"{tag} -> calls [bold]{payload['tool']}[/]({_short(payload['arguments'], 120)})")
    elif kind == "tool_result":
        res = payload["result"]
        style = "red" if isinstance(res, dict) and "error" in res else "green"
        console.print(f"{tag} <- [{style}]{payload['tool']} result[/]: "
                      f"{_short(res, 1000 if verbose else 160)}")
    elif kind == "agent_done":
        out = payload["output"]
        summary = out.get("summary") or out.get("decision")
        console.print(f"{tag} [bold]done[/] ({payload['llm_calls']} LLM calls): {_short(summary, 200)}")


def render_report(r: AssessmentReport) -> None:
    uw, fa = r.underwriter_decision, r.analyst_findings
    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_row("Decision", f"[{DECISION_STYLE.get(r.final_decision, 'bold')}]{r.final_decision}[/]")
    if uw.get("decision") and uw.get("decision") != r.final_decision:
        t.add_row("Underwriter proposed", str(uw.get("decision")))
    if fa:
        t.add_row("Applicant", f"{fa.get('applicant_name', '')} ({fa.get('applicant_id', '')})")
    for label, key, fmt in [("Rate", "interest_rate_pct", "{}%"), ("Monthly payment", "monthly_payment", "{:,.2f}"),
                            ("DTI after loan", "dti_after_pct", "{}%"), ("Confidence", "confidence", "{}")]:
        if uw.get(key) is not None:
            t.add_row(label, fmt.format(uw[key]))
    for label, items in [("Reasons", uw.get("reasons")), ("Conditions", uw.get("conditions")),
                         ("Risk flags", fa.get("risk_flags")), ("Analyst answers", fa.get("answers_to_underwriter")),
                         ("Guardrails", r.guardrail_notes)]:
        if items:
            t.add_row(label, "\n".join(f"- {i}" for i in items))
    t.add_row("Workflow", f"{r.collaboration_rounds} collaboration round(s), {r.llm_calls} LLM calls, "
                          f"{len(r.tool_trace)} tool calls, {r.duration_s}s")
    if r.error:
        t.add_row("Error", f"[red]{r.error}[/]")
    t.add_row("Audit trail", f"runs/{r.run_id}.json")
    console.print(Panel(t, title=f"Assessment {r.run_id} - {r.status}", border_style="bright_white"))


def run_one(orch: LoanOrchestrator, applicant_id: str, amount: float, term: int, purpose: str) -> None:
    try:
        report = orch.assess(LoanApplication(applicant_id.upper(), float(amount), int(term), purpose))
    except ValueError as exc:
        console.print(f"[red]Invalid application:[/] {exc}")
        return
    render_report(report)


def cmd_list() -> None:
    t = Table(title="Sample applicants")
    for col in ("ID", "Name", "Employment"):
        t.add_column(col)
    for a in list_applicants():
        t.add_row(a["applicant_id"], a["name"], a["employment"])
    console.print(t)


def interactive(orch: LoanOrchestrator) -> None:
    while True:
        console.print("\n[bold]1[/] list applicants   [bold]2[/] assess a loan   "
                      "[bold]3[/] run demo scenarios   [bold]q[/] quit")
        choice = console.input("> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return
        if choice == "1":
            cmd_list()
        elif choice == "2":
            
            try:
                aid = console.input("Applicant ID (e.g. A001): ").strip()
                amount = float(console.input("Amount: ").replace(",", ""))
                term = int(console.input("Term in months: "))
                purpose = console.input("Purpose: ").strip() or "personal"
            except ValueError:
                console.print("[red]Amount and term must be numbers.[/]")
                continue
            run_one(orch, aid, amount, term, purpose)
        elif choice == "3":
            run_demo(orch)


def run_demo(orch: LoanOrchestrator) -> None:
    for aid, amount, term, purpose, story in SCENARIOS:
        console.print(Panel(f"{aid}: {amount:,} over {term} months for {purpose}\n[dim]{story}[/]",
                            title="Scenario", border_style="yellow"))
        run_one(orch, aid, amount, term, purpose)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic loan pre-assessment (2 agents, 3 tools, 1 orchestrator)")
    parser.add_argument("--mock", action="store_true", help="offline mode: deterministic mock LLM, no API key")
    parser.add_argument("-v", "--verbose", action="store_true", help="show full tasks and tool results")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list")
    sub.add_parser("demo")
    a = sub.add_parser("assess")
    a.add_argument("applicant_id")
    a.add_argument("--amount", type=float, required=True)
    a.add_argument("--term", type=int, required=True, help="months")
    a.add_argument("--purpose", default="personal")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(use_mock_llm=True if args.mock else None)
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        return 2
    setup_logging(settings.log_level)
    mode = "MOCK (offline)" if settings.use_mock_llm else f"OpenAI {settings.model}"
    console.print(f"[dim]LLM: {mode} | logs: logs/app.log | reports: runs/[/]")

    orch = LoanOrchestrator(build_llm(settings), build_registry(), settings,
                            on_event=lambda k, p: render_event(k, p, args.verbose))
    try:
        if args.command == "list":
            cmd_list()
        elif args.command == "assess":
            run_one(orch, args.applicant_id, args.amount, args.term, args.purpose)
        elif args.command == "demo":
            run_demo(orch)
        else:
            interactive(orch)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        return 130
    except Exception as exc:  # noqa: BLE001 - never show a raw traceback in a demo
        log.exception("unexpected error")
        console.print(f"[red]Unexpected error:[/] {exc} (details in logs/app.log)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
