"""Offline, deterministic stand-in for the LLM (run with --mock).

It follows the same protocol as the real model - it requests tools, reads their
results from the conversation and returns JSON in the required schema - so the
orchestrator, agents, tools and guardrails all run for real. Only the
"reasoning" is a hand-written policy. Use it to rehearse the demo, to run the
tests without an API key, or as a fallback if the network fails mid-demo.
"""
from __future__ import annotations

import json
import re
from itertools import count

from app.llm import LLMResponse, ToolCall

_ids = count(1)


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"mock_{next(_ids)}", name=name, arguments=json.dumps(args))


def _tool_results(messages: list[dict]) -> dict[str, list[dict]]:
    """Map tool name -> list of results seen so far in this conversation."""
    id_to_name = {tc["id"]: tc["function"]["name"]
                  for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
    out: dict[str, list[dict]] = {}
    for m in messages:
        if m["role"] == "tool":
            out.setdefault(id_to_name[m["tool_call_id"]], []).append(json.loads(m["content"]))
    return out


def _json_after(label: str, text: str) -> dict:
    match = re.search(label + r":\n(\{.*\})", text)
    return json.loads(match.group(1)) if match else {}


class MockLLMClient:
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        system, task = messages[0]["content"], messages[1]["content"]
        results = _tool_results(messages)
        if system.startswith("ROLE: ANALYST"):
            return self._analyst(task, results)
        if system.startswith("ROLE: UNDERWRITER"):
            return self._underwriter(task, results)
        return LLMResponse(content='{"error": "mock does not know this role"}')

    # ------------------------------------------------------------------ analyst
    def _analyst(self, task: str, results: dict) -> LLMResponse:
        applicant_id = re.search(r"applicant_id: (\w+)", task).group(1)

        if task.startswith("FOLLOW-UP"):
            focus = "gambling" if "gambling" in task.lower() else "fees" if "overdraft" in task.lower() else "income"
            if not results:
                return LLMResponse(content=f"Drilling into '{focus}' transactions to answer the Underwriter.",
                                   tool_calls=[_call("analyze_transactions", applicant_id=applicant_id,
                                                     focus_category=focus)])
            detail = results["analyze_transactions"][-1]
            findings = _json_after("Your previous findings", task)
            by_month = list(detail.get("net_by_month", {}).values())
            months = len(by_month) or 1
            net = detail.get("net_total", 0)
            escalating = len(by_month) >= 2 and abs(by_month[-1]) > abs(by_month[0]) * 1.25
            trend = "rising month over month" if escalating else "broadly stable month over month"
            findings["answers_to_underwriter"] = [
                f"{focus.title()}: {detail.get('transaction_count', 0)} transactions over {months} months, "
                f"net {net:,.2f} (about {abs(net) / months:,.0f} per month), monthly nets {by_month}; "
                f"the amount is {trend}."
            ]
            return LLMResponse(content=json.dumps(findings))

        if not results:
            return LLMResponse(content="I need the bureau report and the bank cash-flow picture first.",
                               tool_calls=[_call("credit_bureau_lookup", applicant_id=applicant_id),
                                           _call("analyze_transactions", applicant_id=applicant_id)])

        credit = results["credit_bureau_lookup"][-1]
        cash = results["analyze_transactions"][-1]
        gaps = [f"credit_bureau_lookup: {credit['error']}"] if "error" in credit else []
        gaps += [f"analyze_transactions: {cash['error']}"] if "error" in cash else []
        flags = credit.get("bureau_flags", []) + cash.get("cashflow_flags", [])
        positives = []
        if credit.get("credit_score", 0) >= 740:
            positives.append("excellent credit score")
        if credit.get("defaults_last_24m") == 0 and credit.get("late_payments_last_12m") == 0:
            positives.append("clean repayment history")
        if cash.get("avg_net_monthly_cashflow", 0) > 1000:
            positives.append(f"strong surplus of {cash['avg_net_monthly_cashflow']:,.0f}/month")
        findings = {
            "applicant_id": applicant_id,
            "applicant_name": credit.get("name", "unknown"),
            "key_metrics": {
                "credit_score": credit.get("credit_score"),
                "score_band": credit.get("score_band"),
                "avg_monthly_income": cash.get("avg_monthly_income"),
                "min_monthly_income": cash.get("min_monthly_income"),
                "existing_monthly_debt": cash.get("avg_existing_debt_payments"),
                "avg_net_monthly_cashflow": cash.get("avg_net_monthly_cashflow"),
                "income_volatility_pct": cash.get("income_volatility_pct"),
            },
            "risk_flags": flags,
            "positive_factors": positives,
            "data_gaps": gaps,
            "answers_to_underwriter": [],
            "summary": (f"{credit.get('name', applicant_id)} has a {credit.get('score_band', 'UNKNOWN')} credit "
                        f"profile with {len(flags)} risk flag(s)." if not gaps
                        else "Profile is incomplete: " + "; ".join(gaps)),
        }
        return LLMResponse(content=json.dumps(findings))

    # -------------------------------------------------------------- underwriter
    def _underwriter(self, task: str, results: dict) -> LLMResponse:
        app = _json_after("APPLICATION_JSON", task)
        f = _json_after("ANALYST_FINDINGS_JSON", task)
        m = f.get("key_metrics", {})
        score = m.get("credit_score")
        if score is None or f.get("data_gaps"):
            return LLMResponse(content=json.dumps({
                "decision": "DECLINE", "reasons": ["Insufficient verified data to underwrite."],
                "conditions": [], "questions_for_analyst": [], "confidence": 0.6}))
        rate = 8.5 if score >= 740 else 11.9 if score >= 680 else 15.9 if score >= 620 else 21.9
        volatile = (m.get("income_volatility_pct") or 0) > 40
        income = m["min_monthly_income"] if volatile else m["avg_monthly_income"]

        if not results:
            note = " using minimum monthly income because income is volatile" if volatile else ""
            return LLMResponse(
                content=f"Score {score} prices at {rate}%. Checking affordability{note}.",
                tool_calls=[_call("loan_affordability_calculator", principal=app["amount"], annual_rate_pct=rate,
                                  term_months=app["term_months"], monthly_income=income,
                                  existing_monthly_debt=m["existing_monthly_debt"])])

        calc = results["loan_affordability_calculator"][-1]
        dti = calc["dti_after_pct"]
        flags = f.get("risk_flags", [])
        has_default = any("default" in x for x in flags)
        base = {"interest_rate_pct": rate, "monthly_payment": calc["monthly_payment"], "dti_after_pct": dti,
                "conditions": [], "questions_for_analyst": []}

        if any("gambling" in x for x in flags) and not f.get("answers_to_underwriter"):
            return LLMResponse(content=json.dumps({**base, "decision": "NEED_MORE_INFO",
                "reasons": ["Gambling spend could indicate escalating financial stress."],
                "questions_for_analyst": ["Is the gambling spend stable or increasing month over month, "
                                          "and what is the net monthly amount after winnings?"],
                "confidence": 0.5}))
        if score < 620 or has_default or dti > 45:
            reasons = [r for r, hit in [(f"Credit score {score} is below the 620 minimum", score < 620),
                                        ("Default recorded in the last 24 months", has_default),
                                        (f"DTI after loan {dti}% exceeds 45%", dti > 45)] if hit]
            return LLMResponse(content=json.dumps({**base, "decision": "DECLINE", "reasons": reasons,
                                                   "confidence": 0.9}))
        if score >= 680 and dti <= 36 and not flags:
            return LLMResponse(content=json.dumps({**base, "decision": "APPROVE", "confidence": 0.92,
                "reasons": [f"Score {score} ({m.get('score_band')})", f"DTI after loan {dti}% is comfortable",
                            "No adverse flags"]}))
        conditions = []
        if volatile or any("irregular" in x for x in flags):
            conditions.append("Provide 12 months of bank statements or tax returns to verify income")
        if any("thin" in x for x in flags):
            conditions.append("Limit to a 12-month probation period with automatic repayments")
        if any("gambling" in x for x in flags):
            conditions.append("Set up direct-debit repayments on payday; review account at month 6")
        if any("utilisation" in x for x in flags) or dti > 36:
            conditions.append("Consider a lower amount or longer term to keep DTI under 36%")
        return LLMResponse(content=json.dumps({**base, "decision": "CONDITIONAL_APPROVE", "confidence": 0.75,
            "reasons": [f"Score {score} meets the minimum", f"DTI after loan {dti}% is within policy"]
                       + [f"Mitigable flag: {x}" for x in flags],
            "conditions": conditions or ["Standard verification of employment"]}))
