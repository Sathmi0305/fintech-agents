"""Tool 2 - Bank transaction (cash-flow) analyser, as an open-banking feed would provide."""
from __future__ import annotations

from collections import defaultdict

from app.data import get_applicant
from app.tools.base import Tool, ToolError

RISKY_CATEGORIES = {"gambling"}


def analyze_transactions(applicant_id: str, focus_category: str | None = None) -> dict:
    applicant = get_applicant(applicant_id)
    if applicant is None:
        raise ToolError(f"No bank account data found for applicant_id '{applicant_id}'.")
    txns = applicant["transactions"]
    months = sorted({t["date"][:7] for t in txns})
    n = len(months) or 1

    # Drill-down mode: the agent asks for detail on one spending category.
    if focus_category:
        rows = [t for t in txns if t["category"] == focus_category.lower()]
        if not rows:
            return {"applicant_id": applicant["applicant_id"], "focus_category": focus_category,
                    "transactions": [], "note": "No transactions in this category."}
        by_month: dict[str, float] = defaultdict(float)
        for t in rows:
            by_month[t["date"][:7]] += t["amount"]
        return {
            "applicant_id": applicant["applicant_id"],
            "focus_category": focus_category,
            "transaction_count": len(rows),
            "net_by_month": {m: round(v, 2) for m, v in sorted(by_month.items())},
            "net_total": round(sum(t["amount"] for t in rows), 2),
            "transactions": rows,
        }

    income_by_month: dict[str, float] = defaultdict(float)
    spend_by_cat: dict[str, float] = defaultdict(float)
    for t in txns:
        if t["category"] == "income":
            income_by_month[t["date"][:7]] += t["amount"]
        elif t["amount"] < 0:
            spend_by_cat[t["category"]] += -t["amount"]

    incomes = [income_by_month.get(m, 0.0) for m in months]
    avg_income = sum(incomes) / n
    avg_expenses = sum(spend_by_cat.values()) / n
    avg_debt = spend_by_cat.get("debt_repayment", 0.0) / n
    volatility = (max(incomes) - min(incomes)) / avg_income * 100 if avg_income else 0.0
    overdraft_fees = sum(1 for t in txns if "overdraft" in t["description"].lower())

    flags = []
    for cat in RISKY_CATEGORIES:
        share = spend_by_cat.get(cat, 0.0) / n / avg_income * 100 if avg_income else 0
        if share > 2:
            flags.append(f"{cat} spending is {share:.1f}% of monthly income")
    if overdraft_fees:
        flags.append(f"{overdraft_fees} overdraft fee(s) in {n} months")
    if volatility > 40:
        flags.append(f"irregular income (month-to-month swing {volatility:.0f}%)")
    if avg_income - avg_expenses < 0:
        flags.append("negative average monthly cash flow")

    return {
        "applicant_id": applicant["applicant_id"],
        "months_analysed": months,
        "avg_monthly_income": round(avg_income, 2),
        "min_monthly_income": round(min(incomes), 2),
        "income_volatility_pct": round(volatility, 1),
        "avg_monthly_expenses": round(avg_expenses, 2),
        "avg_existing_debt_payments": round(avg_debt, 2),
        "avg_net_monthly_cashflow": round(avg_income - avg_expenses, 2),
        "avg_spend_by_category": {k: round(v / n, 2) for k, v in sorted(spend_by_cat.items())},
        "cashflow_flags": flags,
    }


TOOL = Tool(
    name="analyze_transactions",
    description="Analyse the applicant's last 3 months of bank transactions: average income, expenses, "
                "existing debt payments, net cash flow, income stability and behavioural risk flags. "
                "Pass focus_category (e.g. 'gambling', 'fees', 'income') to get the raw transactions "
                "for one category instead of the summary.",
    parameters={
        "type": "object",
        "properties": {
            "applicant_id": {"type": "string", "description": "Applicant ID, e.g. A001"},
            "focus_category": {"type": "string", "description": "Optional category to drill into"},
        },
        "required": ["applicant_id"],
    },
    func=analyze_transactions,
)
