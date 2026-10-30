import json

import pytest

from app.tools import build_registry
from app.tools.affordability import loan_affordability_calculator, monthly_instalment
from app.tools.base import ToolError
from app.tools.credit_bureau import credit_bureau_lookup
from app.tools.transactions import analyze_transactions


def test_credit_bureau_flags_risky_applicant():
    r = credit_bureau_lookup("A003")
    assert r["score_band"] == "POOR"
    assert any("default" in f for f in r["bureau_flags"])


def test_credit_bureau_unknown_applicant():
    with pytest.raises(ToolError):
        credit_bureau_lookup("ZZZ")


def test_transactions_summary_and_flags():
    r = analyze_transactions("A002")
    assert r["avg_monthly_income"] == 4100
    assert any("gambling" in f for f in r["cashflow_flags"])


def test_transactions_drill_down():
    r = analyze_transactions("A002", focus_category="gambling")
    assert r["transaction_count"] == 9 and r["net_total"] < 0


def test_irregular_income_detected():
    r = analyze_transactions("A004")
    assert r["min_monthly_income"] < r["avg_monthly_income"]
    assert any("irregular" in f for f in r["cashflow_flags"])


def test_emi_matches_standard_formula():
    assert monthly_instalment(20000, 8.5, 36) == pytest.approx(631.35, abs=0.01)
    assert monthly_instalment(1200, 0, 12) == 100


def test_affordability_dti_and_band():
    r = loan_affordability_calculator(20000, 8.5, 36, 6200, 350)
    assert r["dti_after_pct"] == pytest.approx(15.8, abs=0.1)
    assert r["affordability_band"] == "COMFORTABLE"


@pytest.mark.parametrize("args", [(0, 10, 12, 1000, 0), (1000, 10, 12, 0, 0), (1000, 99, 12, 1000, 0)])
def test_affordability_rejects_bad_input(args):
    with pytest.raises(ToolError):
        loan_affordability_calculator(*args)


def test_registry_turns_failures_into_error_results():
    reg = build_registry()
    assert "error" in reg.execute("credit_bureau_lookup", '{"applicant_id": "NOPE"}')
    assert "error" in reg.execute("credit_bureau_lookup", "{not json")
    assert "error" in reg.execute("credit_bureau_lookup", json.dumps({"wrong_arg": 1}))
    assert "error" in reg.execute("does_not_exist", "{}")
    assert "not available" in reg.execute("credit_bureau_lookup", '{"applicant_id":"A001"}',
                                          allowed=["loan_affordability_calculator"])["error"]
