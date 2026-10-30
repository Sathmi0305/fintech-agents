"""Tool 1 - Credit bureau lookup: score, band and repayment history."""
from __future__ import annotations

from app.data import get_applicant
from app.tools.base import Tool, ToolError


def score_band(score: int) -> str:
    if score >= 740:
        return "EXCELLENT"
    if score >= 680:
        return "GOOD"
    if score >= 620:
        return "FAIR"
    return "POOR"


def credit_bureau_lookup(applicant_id: str) -> dict:
    applicant = get_applicant(applicant_id)
    if applicant is None:
        raise ToolError(f"No credit file found for applicant_id '{applicant_id}'.")
    c = applicant["credit"]
    flags = []
    if c["defaults_last_24m"] > 0:
        flags.append(f"{c['defaults_last_24m']} default(s) in last 24 months")
    if c["late_payments_last_12m"] >= 3:
        flags.append(f"{c['late_payments_last_12m']} late payments in last 12 months")
    if c["credit_utilization_pct"] > 70:
        flags.append(f"high credit utilisation ({c['credit_utilization_pct']}%)")
    if c["hard_inquiries_last_6m"] >= 4:
        flags.append(f"{c['hard_inquiries_last_6m']} hard inquiries in 6 months (credit seeking)")
    if c["credit_history_years"] < 3 or c["open_accounts"] <= 1:
        flags.append("thin credit file")
    return {
        "applicant_id": applicant["applicant_id"],
        "name": applicant["name"],
        "employment": applicant["employment"],
        "credit_score": c["score"],
        "score_band": score_band(c["score"]),
        **{k: v for k, v in c.items() if k != "score"},
        "bureau_flags": flags,
    }


TOOL = Tool(
    name="credit_bureau_lookup",
    description="Fetch the applicant's credit bureau report: credit score, score band, utilisation, "
                "defaults, late payments, inquiries and any bureau risk flags.",
    parameters={
        "type": "object",
        "properties": {"applicant_id": {"type": "string", "description": "Applicant ID, e.g. A001"}},
        "required": ["applicant_id"],
    },
    func=credit_bureau_lookup,
)
