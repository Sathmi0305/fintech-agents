"""The Orchestrator: owns the workflow, the shared context and the guardrails.

Workflow
  1. Analyst gathers evidence (credit bureau + transactions)          -> findings
  2. Underwriter applies policy (affordability calculator)            -> decision
  3. If Underwriter says NEED_MORE_INFO, route its questions back to
     the Analyst, then re-run the Underwriter (bounded by max_revisions)
  4. Deterministic guardrails check the final decision against the raw
     tool outputs (not the agents' prose) before it is accepted
  5. Persist an auditable report (runs/<run_id>.json)

The agents never call each other directly; everything they share passes
through the orchestrator, which is what makes the flow traceable.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.agents import AgentError, AnalystAgent, UnderwriterAgent
from app.agents.base import EventCallback
from app.config import Settings
from app.llm import LLMClient, LLMError
from app.tools import ToolRegistry

log = logging.getLogger(__name__)
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

FINAL_DECISIONS = {"APPROVE", "CONDITIONAL_APPROVE", "DECLINE", "REFER_TO_HUMAN"}
MAX_DTI_FOR_APPROVAL = 45.0


@dataclass
class LoanApplication:
    applicant_id: str
    amount: float
    term_months: int
    purpose: str = "personal"

    def validate(self) -> None:
        if not self.applicant_id.strip():
            raise ValueError("applicant_id is required")
        if not 500 <= self.amount <= 250_000:
            raise ValueError("amount must be between 500 and 250,000")
        if not 6 <= self.term_months <= 120:
            raise ValueError("term_months must be between 6 and 120")


@dataclass
class AssessmentReport:
    run_id: str
    started_at: str
    application: dict
    status: str = "RUNNING"                 # COMPLETED | FAILED
    final_decision: str = "REFER_TO_HUMAN"
    analyst_findings: dict = field(default_factory=dict)
    underwriter_decision: dict = field(default_factory=dict)
    collaboration_rounds: int = 0           # times the Underwriter sent questions back
    guardrail_notes: list[str] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    duration_s: float = 0.0
    error: str | None = None


class LoanOrchestrator:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, settings: Settings,
                 on_event: EventCallback | None = None) -> None:
        self.settings = settings
        self.on_event = on_event or (lambda kind, payload: None)
        self.analyst = AnalystAgent(llm, registry, settings.max_agent_steps, self.on_event)
        self.underwriter = UnderwriterAgent(llm, registry, settings.max_agent_steps, self.on_event)

    def _stage(self, text: str) -> None:
        log.info("orchestrator: %s", text)
        self.on_event("stage", {"agent": "Orchestrator", "text": text})

    # ------------------------------------------------------------------ workflow
    def assess(self, application: LoanApplication) -> AssessmentReport:
        application.validate()
        start = time.perf_counter()
        report = AssessmentReport(run_id=uuid.uuid4().hex[:8],
                                  started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  application=asdict(application))
        log.info("run %s started: %s", report.run_id, report.application)
        try:
            self._run_workflow(application, report)
            report.status = "COMPLETED"
        except (AgentError, LLMError) as exc:
            # Fail safe: an incomplete automated assessment is never an approval.
            log.error("run %s failed: %s", report.run_id, exc)
            report.status, report.error = "FAILED", str(exc)
            report.final_decision = "REFER_TO_HUMAN"
            report.guardrail_notes.append("Automated assessment failed; routed to a human underwriter.")
        report.duration_s = round(time.perf_counter() - start, 2)
        self._save(report)
        return report

    def _run_workflow(self, app: LoanApplication, report: AssessmentReport) -> None:
        self._stage("Step 1: dispatching applicant to the Analyst for evidence gathering")
        analyst = self.analyst.run(
            f"Build a risk profile for this loan applicant.\napplicant_id: {app.applicant_id}\n"
            f"Loan request: {app.amount:,.0f} over {app.term_months} months for '{app.purpose}'."
        )
        self._absorb(report, analyst)
        findings = analyst.output

        uw_output: dict = {}
        for round_no in range(self.settings.max_revisions + 1):
            label = "Step 2" if round_no == 0 else "Step 3b"
            self._stage(f"{label}: sending application + findings to the Underwriter (round {round_no + 1})")
            uw = self.underwriter.run(
                "Decide on this loan application.\n"
                f"APPLICATION_JSON:\n{json.dumps(asdict(app))}\n"
                f"ANALYST_FINDINGS_JSON:\n{json.dumps(findings)}"
            )
            self._absorb(report, uw)
            uw_output = uw.output
            questions = uw_output.get("questions_for_analyst") or []
            if uw_output.get("decision") != "NEED_MORE_INFO":
                break
            if round_no == self.settings.max_revisions:
                self._stage("Underwriter still needs information but the revision budget is used up")
                break

            report.collaboration_rounds += 1
            self._stage(f"Step 3a: Underwriter asked {len(questions)} question(s); routing back to the Analyst")
            follow_up = self.analyst.run(
                f"FOLLOW-UP from the Underwriter for applicant_id: {app.applicant_id}\n"
                "Questions:\n" + "\n".join(f"- {q}" for q in questions) +
                f"\nYour previous findings:\n{json.dumps(findings)}\n"
                "Investigate with your tools, answer each question in answers_to_underwriter, "
                "and return the complete updated JSON."
            )
            self._absorb(report, follow_up)
            findings = follow_up.output

        report.analyst_findings = findings
        report.underwriter_decision = uw_output
        self._stage("Step 4: verifying the decision against raw tool outputs (guardrails)")
        report.final_decision = self._apply_guardrails(app, uw_output, report)

    @staticmethod
    def _absorb(report: AssessmentReport, result) -> None:
        report.llm_calls += result.llm_calls
        report.tool_trace.extend({"agent": result.agent, **t} for t in result.trace)

    # ---------------------------------------------------------------- guardrails
    def _apply_guardrails(self, app: LoanApplication, uw: dict, report: AssessmentReport) -> str:
        notes = report.guardrail_notes
        decision = str(uw.get("decision", "")).upper()

        if decision == "NEED_MORE_INFO":
            notes.append("Underwriter could not decide within the collaboration budget; referred to a human.")
            return "REFER_TO_HUMAN"
        if decision not in FINAL_DECISIONS:
            notes.append(f"Unrecognised decision '{decision}'; referred to a human.")
            return "REFER_TO_HUMAN"
        if decision == "DECLINE":
            return decision  # declining is always the safe direction

        # Trust tool outputs, not the agent's narration.
        calcs = [t["result"] for t in report.tool_trace
                 if t["tool"] == "loan_affordability_calculator" and "error" not in t["result"]]
        bureau = [t["result"] for t in report.tool_trace
                  if t["tool"] == "credit_bureau_lookup" and "error" not in t["result"]]
        if not calcs:
            notes.append("Approval was not backed by an affordability calculation; referred to a human.")
            return "REFER_TO_HUMAN"
        calc = calcs[-1]
        if round(calc["principal"]) != round(app.amount) or calc["term_months"] != app.term_months:
            notes.append("Affordability was calculated for a different amount/term than requested; referred.")
            return "REFER_TO_HUMAN"
        if calc["dti_after_pct"] > MAX_DTI_FOR_APPROVAL:
            notes.append(f"Verified DTI {calc['dti_after_pct']}% exceeds the {MAX_DTI_FOR_APPROVAL}% hard limit; "
                         "approval overridden.")
            return "REFER_TO_HUMAN"
        if bureau and bureau[-1].get("defaults_last_24m", 0) > 0:
            notes.append("Bureau shows a default in the last 24 months; approval overridden.")
            return "REFER_TO_HUMAN"
        if abs(float(uw.get("monthly_payment") or 0) - calc["monthly_payment"]) > 1:
            notes.append(f"Underwriter's stated payment corrected to the calculator's {calc['monthly_payment']}.")
            uw["monthly_payment"] = calc["monthly_payment"]
        notes.append("Decision consistent with verified tool outputs.")
        return decision

    @staticmethod
    def _save(report: AssessmentReport) -> None:
        try:
            RUNS_DIR.mkdir(exist_ok=True)
            path = RUNS_DIR / f"{report.run_id}.json"
            path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
            log.info("run %s saved to %s", report.run_id, path)
        except OSError:
            log.exception("could not save run report %s", report.run_id)
