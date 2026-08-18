"""Offline tests for LlmAmountResolver request/response handling with the Groq
json_schema structured-output payload. httpx is mocked; no real LLM calls.
"""
import logging
from decimal import Decimal

import httpx

from app.extractors.invoice_amount_extractor import LabelledCandidate
from app.resolvers.amount_resolver import ResolverContext
from app.resolvers.llm_amount_resolver import LlmAmountResolver

KEEP_JSON = '{"selected_amount":null,"confidence":0.0,"evidence":null}'


def _resolver() -> LlmAmountResolver:
    return LlmAmountResolver(api_key="secret-key")


def _ok_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


def _err_response(status: int, body: str) -> httpx.Response:
    return httpx.Response(
        status,
        text=body,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


def test_request_uses_json_schema_structured_output(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _ok_response(KEEP_JSON)

    monkeypatch.setattr("app.resolvers.llm_amount_resolver.httpx.post", fake_post)

    raw = _resolver()._request_completion("a prompt mentioning json")

    # Response contract unchanged: raw JSON string with the four fields.
    assert raw == KEEP_JSON

    rf = captured["json"]["response_format"]
    assert rf["type"] == "json_schema"  # not the failing json_object mode
    schema = rf["json_schema"]["schema"]
    assert set(schema["properties"]) == {"selected_amount", "confidence", "evidence"}
    assert set(schema["required"]) == {"selected_amount", "confidence", "evidence"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["selected_amount"]["type"] == ["string", "null"]

    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["model"]  # model still sent
    assert captured["json"]["max_tokens"] == 500  # reasoning-token budget
    assert captured["json"]["reasoning_effort"] == "low"  # keep reasoning short


def test_response_content_flows_through_arbitrator(monkeypatch):
    override = '{"selected_amount":"1200.00","confidence":0.9,"evidence":"Invoice Amount 1200"}'
    monkeypatch.setattr(
        "app.resolvers.llm_amount_resolver.httpx.post",
        lambda url, json=None, headers=None, timeout=None: _ok_response(override),
    )
    ctx = ResolverContext(
        ocr_text="Invoice Amount 1200",
        candidates=[Decimal("1000.00"), Decimal("1200.00")],
        deterministic_winner=Decimal("1000.00"), winner_label="total", winner_priority=11,
        labelled_candidates=[
            LabelledCandidate("total", Decimal("1000.00"), "Total 1000", 11),
            LabelledCandidate("invoice amount", Decimal("1200.00"), "Invoice Amount 1200", 0),
        ],
        amount_in_words=None,
    )
    assert _resolver().resolve(ctx).amount == Decimal("1200.00")


def test_http_400_degrades_to_keep_and_logs_body_not_key(monkeypatch, caplog):
    monkeypatch.setattr(
        "app.resolvers.llm_amount_resolver.httpx.post",
        lambda url, json=None, headers=None, timeout=None: _err_response(
            400, '{"error":{"code":"json_validate_failed","message":"Failed to validate JSON."}}'
        ),
    )
    ctx = ResolverContext(
        ocr_text="x", candidates=[Decimal("1")], deterministic_winner=Decimal("1"),
        winner_label="total", winner_priority=11, labelled_candidates=[], amount_in_words=None,
    )
    with caplog.at_level(logging.ERROR):
        result = _resolver().resolve(ctx)

    assert result.amount is None  # safe degradation to deterministic keep
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "400" in logged and "json_validate_failed" in logged
    assert "secret-key" not in logged  # key never logged
