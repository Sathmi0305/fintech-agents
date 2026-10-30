"""Tool 3 - Loan affordability calculator: monthly instalment (EMI) and debt-to-income."""
from __future__ import annotations

from app.tools.base import Tool, ToolError


def monthly_instalment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / term_months
    return principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)


def loan_affordability_calculator(principal: float, annual_rate_pct: float, term_months: int,
                                  monthly_income: float, existing_monthly_debt: float = 0.0) -> dict:
    if principal <= 0 or term_months <= 0:
        raise ToolError("principal and term_months must be positive.")
    if not 0 <= annual_rate_pct <= 60:
        raise ToolError("annual_rate_pct must be between 0 and 60.")
    if monthly_income <= 0:
        raise ToolError("monthly_income must be positive to compute affordability.")
    if existing_monthly_debt < 0:
        raise ToolError("existing_monthly_debt cannot be negative.")

    emi = monthly_instalment(principal, annual_rate_pct, int(term_months))
    dti_before = existing_monthly_debt / monthly_income * 100
    dti_after = (existing_monthly_debt + emi) / monthly_income * 100
    return {
        "principal": principal,
        "annual_rate_pct": annual_rate_pct,
        "term_months": int(term_months),
        "monthly_payment": round(emi, 2),
        "total_repayable": round(emi * term_months, 2),
        "total_interest": round(emi * term_months - principal, 2),
        "dti_before_pct": round(dti_before, 1),
        "dti_after_pct": round(dti_after, 1),
        "affordability_band": "COMFORTABLE" if dti_after <= 36 else "STRETCHED" if dti_after <= 45 else "UNAFFORDABLE",
    }


TOOL = Tool(
    name="loan_affordability_calculator",
    description="Compute the monthly instalment (amortised EMI), total interest and debt-to-income ratio "
                "before and after the new loan. Bands: COMFORTABLE <=36% DTI, STRETCHED <=45%, else UNAFFORDABLE.",
    parameters={
        "type": "object",
        "properties": {
            "principal": {"type": "number", "description": "Loan amount"},
            "annual_rate_pct": {"type": "number", "description": "Annual interest rate in percent, e.g. 11.9"},
            "term_months": {"type": "integer", "description": "Loan term in months"},
            "monthly_income": {"type": "number", "description": "Verified average monthly income"},
            "existing_monthly_debt": {"type": "number", "description": "Existing monthly debt repayments"},
        },
        "required": ["principal", "annual_rate_pct", "term_months", "monthly_income", "existing_monthly_debt"],
    },
    func=loan_affordability_calculator,
)
