import json
from types import SimpleNamespace

import pytest

from app.agents import AgentError, AnalystAgent
from app.config import Settings
from app.llm import LLMError, LLMResponse, OpenAIClient, ToolCall
from app.mock_llm import MockLLMClient
from app.orchestrator import LoanApplication, LoanOrchestrator
from app.tools import build_registry

SETTINGS = Settings(openai_api_key=None, model="test", temperature=None, request_timeout_s=5,
                    max_agent_steps=6, max_revisions=1, log_level="INFO", use_mock_llm=True)


def orchestrator(llm=None):
    return LoanOrchestrator(llm or MockLLMClient(), build_registry(), SETTINGS)


@pytest.mark.parametrize("aid,amount,term,expected,rounds", [
    ("A001", 20000, 36, "APPROVE", 0),
    ("A002", 12000, 36, "CONDITIONAL_APPROVE", 1),   # Underwriter -> Analyst follow-up loop
    ("A003", 15000, 48, "DECLINE", 0),
    ("A004", 15000, 48, "CONDITIONAL_APPROVE", 0),
])
def test_end_to_end_scenarios(aid, amount, term, expected, rounds):
    r = orchestrator().assess(LoanApplication(aid, amount, term))
    assert r.status == "COMPLETED"
    assert r.final_decision == expected
    assert r.collaboration_rounds == rounds
    assert {t["tool"] for t in r.tool_trace} >= {"credit_bureau_lookup", "analyze_transactions",
                                                  "loan_affordability_calculator"}


def test_unknown_applicant_never_approves():
    r = orchestrator().assess(LoanApplication("A999", 5000, 12))
    assert r.final_decision in {"DECLINE", "REFER_TO_HUMAN"}
    assert r.analyst_findings["data_gaps"]


def test_invalid_application_rejected():
    with pytest.raises(ValueError):
        orchestrator().assess(LoanApplication("A001", 10, 12))


class ScriptedLLM:
    """Returns a fixed sequence of responses, regardless of input."""
    def __init__(self, *responses):
        self.responses = list(responses)

    def chat(self, messages, tools=None):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_llm_failure_is_failed_and_referred():
    r = orchestrator(ScriptedLLM(LLMError("network down"))).assess(LoanApplication("A001", 20000, 36))
    assert r.status == "FAILED" and r.final_decision == "REFER_TO_HUMAN"
    assert "network down" in r.error


def test_agent_retries_once_on_bad_json_then_fails():
    agent = AnalystAgent(ScriptedLLM(LLMResponse("not json"), LLMResponse("still not json")), build_registry())
    with pytest.raises(AgentError):
        agent.run("applicant_id: A001")


def test_agent_step_budget():
    loop = [LLMResponse(None, [ToolCall(f"c{i}", "credit_bureau_lookup", '{"applicant_id":"A001"}')])
            for i in range(3)]
    with pytest.raises(AgentError, match="step budget"):
        AnalystAgent(ScriptedLLM(*loop), build_registry(), max_steps=3).run("applicant_id: A001")


def test_guardrail_overrides_unsupported_approval():
    """Underwriter claims APPROVE but the calculator shows DTI > 45%: orchestrator overrides."""
    findings = {"applicant_id": "A003", "key_metrics": {}, "risk_flags": [], "summary": "ok"}
    calc = ToolCall("u1", "loan_affordability_calculator", json.dumps(
        {"principal": 15000, "annual_rate_pct": 8.5, "term_months": 48,
         "monthly_income": 3000, "existing_monthly_debt": 1100}))
    llm = ScriptedLLM(
        LLMResponse(json.dumps(findings)),                       # Analyst answers straight away
        LLMResponse(None, [calc]),                               # Underwriter calls the calculator
        LLMResponse(json.dumps({"decision": "APPROVE", "reasons": ["looks fine"], "monthly_payment": 1})),
    )
    r = orchestrator(llm).assess(LoanApplication("A003", 15000, 48))
    assert r.underwriter_decision["decision"] == "APPROVE"
    assert r.final_decision == "REFER_TO_HUMAN"
    assert any("hard limit" in n for n in r.guardrail_notes)


def test_guardrail_requires_affordability_calculation():
    findings = {"applicant_id": "A001", "key_metrics": {}, "risk_flags": [], "summary": "ok"}
    llm = ScriptedLLM(LLMResponse(json.dumps(findings)),
                      LLMResponse(json.dumps({"decision": "APPROVE", "reasons": ["trust me"]})))
    r = orchestrator(llm).assess(LoanApplication("A001", 20000, 36))
    assert r.final_decision == "REFER_TO_HUMAN"


def test_openai_client_request_and_parsing(monkeypatch):
    """Checks the real client sends tools correctly and normalises tool calls (no network)."""
    settings = Settings(openai_api_key="sk-test", model="gpt-test", temperature=0.1, request_timeout_s=5,
                        max_agent_steps=6, max_revisions=1, log_level="INFO", use_mock_llm=False)
    client = OpenAIClient(settings)
    captured = {}
    fake_msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="call_1", type="function", function=SimpleNamespace(name="credit_bureau_lookup",
                                                               arguments='{"applicant_id":"A001"}'))])
    fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=fake_msg)], usage=None)

    def fake_create(**kwargs):
        captured.update(kwargs)
        return fake_resp

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    tools = build_registry().schemas(["credit_bureau_lookup"])
    out = client.chat([{"role": "user", "content": "hi"}], tools)
    assert captured["model"] == "gpt-test" and captured["tools"] == tools and captured["temperature"] == 0.1
    assert out.tool_calls[0].name == "credit_bureau_lookup"
