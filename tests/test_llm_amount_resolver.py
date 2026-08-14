"""Offline tests for the Groq arbitration resolver. No real API calls: the
provider request is stubbed. Covers the deterministic override gates.

_TOTAL_KEYWORDS priority reference (lower = more authoritative):
  invoice amount=0, invoice total=1, grand total=2, ..., total amount=10, total=11
"""
from decimal import Decimal

import httpx
import pytest

from app.extractors.invoice_amount_extractor import (
    InvoiceAmountExtractor,
    LabelledCandidate,
)
from app.resolvers.amount_resolver import ResolverContext
from app.resolvers.llm_amount_resolver import (
    LlmAmountResolver,
    build_default_resolver,
)
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeOcrClient

PDF = "application/pdf"


class StubResolver(LlmAmountResolver):
    """LlmAmountResolver with the network call stubbed (real gates run)."""

    def __init__(self, raw: str = "", exc: Exception | None = None):
        super().__init__(api_key="test-key")
        self._raw = raw
        self._exc = exc
        self.calls = 0

    def _request_completion(self, user_prompt: str) -> str:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._raw


def ctx(
    *,
    winner=None,
    winner_label=None,
    winner_priority=None,
    labelled=None,
    candidates=None,
    words=None,
    ocr="OCR TEXT",
) -> ResolverContext:
    return ResolverContext(
        ocr_text=ocr,
        candidates=candidates or [],
        deterministic_winner=winner,
        winner_label=winner_label,
        winner_priority=winner_priority,
        labelled_candidates=labelled or [],
        amount_in_words=words,
    )


def lc(label, amount, priority, line="line"):
    return LabelledCandidate(label=label, amount=Decimal(amount), line=line, priority=priority)


# --- Test 1: 8.jpg regression (lower-priority override rejected) -----------

def test_lower_priority_override_is_rejected():
    context = ctx(
        winner=Decimal("47925.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "47925.00", 0), lc("total amount", "47924.68", 10)],
        candidates=[Decimal("47925.00"), Decimal("47924.68")],
    )
    raw = '{"decision":"override","amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}'
    assert StubResolver(raw).resolve(context).amount is None


# --- Test 2: keep -----------------------------------------------------------

def test_keep_decision_returns_none():
    raw = '{"decision":"keep","amount":null,"confidence":0.0,"evidence":null}'
    assert StubResolver(raw).resolve(ctx(winner=Decimal("100"))).amount is None


# --- Test 3: override with same amount == keep ------------------------------

def test_override_same_amount_is_treated_as_keep():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total amount", winner_priority=10,
        labelled=[lc("total amount", "1000.00", 10)],
        candidates=[Decimal("1000.00")],
    )
    raw = '{"decision":"override","amount":"1000.00","confidence":0.99,"evidence":"Total 1000"}'
    assert StubResolver(raw).resolve(context).amount is None


# --- Test 4: valid override (equal/higher priority) -------------------------

def test_valid_override_is_accepted():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "1000.00", 0), lc("invoice amount", "1200.00", 0)],
        candidates=[Decimal("1000.00"), Decimal("1200.00")],
    )
    raw = '{"decision":"override","amount":"1200.00","confidence":0.9,"evidence":"Invoice Amount 1200"}'
    result = StubResolver(raw).resolve(context)
    assert result.amount == Decimal("1200.00")


# --- Test 5: low confidence -------------------------------------------------

def test_override_below_threshold_is_rejected():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "1000.00", 0), lc("invoice amount", "1200.00", 0)],
        candidates=[Decimal("1000.00"), Decimal("1200.00")],
    )
    raw = '{"decision":"override","amount":"1200.00","confidence":0.6,"evidence":"Invoice Amount 1200"}'
    assert StubResolver(raw).resolve(context).amount is None


# --- Test 6: ungrounded amount ----------------------------------------------

def test_ungrounded_override_is_rejected():
    context = ctx(
        winner=Decimal("20000"), winner_label="total", winner_priority=11,
        labelled=[lc("total", "20000", 11)],
        candidates=[Decimal("20000")],
    )
    raw = '{"decision":"override","amount":"68230.50","confidence":0.99,"evidence":"made up"}'
    assert StubResolver(raw).resolve(context).amount is None


# --- Test 7: lower-priority override vs winner authority --------------------

def test_lower_priority_override_rejected_when_winner_authoritative():
    # Winner carries an explicit invoice-level label -> lower-priority refused.
    context = ctx(
        winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "1000.00", 0), lc("total", "900.00", 11)],
        candidates=[Decimal("1000.00"), Decimal("900.00")],
    )
    raw = '{"decision":"override","amount":"900.00","confidence":0.99,"evidence":"Total 900"}'
    assert StubResolver(raw).resolve(context).amount is None


def test_lower_priority_override_allowed_when_winner_not_authoritative():
    # Winner is only a weak "total amount" label -> a lower-priority candidate
    # may override (this is the Case-4 relaxation).
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total amount", winner_priority=10,
        labelled=[lc("total amount", "1000.00", 10), lc("total", "900.00", 11)],
        candidates=[Decimal("1000.00"), Decimal("900.00")],
    )
    raw = '{"decision":"override","amount":"900.00","confidence":0.9,"evidence":"Total 900"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("900.00")


# --- Test 8: missing winner, bare token -------------------------------------

def test_missing_winner_bare_token_is_rejected():
    context = ctx(
        winner=None, labelled=[], candidates=[Decimal("500.00")], words=None,
    )
    raw = '{"decision":"override","amount":"500.00","confidence":0.99,"evidence":"500.00"}'
    assert StubResolver(raw).resolve(context).amount is None


# --- Test 9: missing winner with labelled evidence --------------------------

def test_missing_winner_with_labelled_candidate_is_accepted():
    context = ctx(
        winner=None,
        labelled=[lc("invoice total", "1200.00", 1, line="Invoice Total 1200.00")],
        candidates=[Decimal("1200.00")], words=None,
    )
    raw = '{"decision":"override","amount":"1200.00","confidence":0.9,"evidence":"Invoice Total 1200.00"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("1200.00")


def test_missing_winner_with_amount_in_words_is_accepted():
    context = ctx(
        winner=None, labelled=[], candidates=[Decimal("4490.00")],
        words=Decimal("4490.00"),
    )
    raw = '{"decision":"override","amount":"4490.00","confidence":0.9,"evidence":"FOUR THOUSAND ..."}'
    assert StubResolver(raw).resolve(context).amount == Decimal("4490.00")


# --- Test 10: timeout / exception -------------------------------------------

def test_timeout_degrades_to_keep():
    assert StubResolver(exc=httpx.TimeoutException("slow")).resolve(ctx()).amount is None


def test_exception_degrades_to_keep():
    assert StubResolver(exc=RuntimeError("boom")).resolve(ctx()).amount is None


# --- extra adapter validation ----------------------------------------------

def test_malformed_json_degrades_to_keep():
    assert StubResolver("not json").resolve(ctx(winner=Decimal("1"))).amount is None


def test_override_missing_evidence_is_rejected():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "1200.00", 0)], candidates=[Decimal("1200.00")],
    )
    raw = '{"decision":"override","amount":"1200.00","confidence":0.9,"evidence":""}'
    assert StubResolver(raw).resolve(context).amount is None


def test_override_invalid_confidence_is_rejected():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "1200.00", 0)], candidates=[Decimal("1200.00")],
    )
    raw = '{"decision":"override","amount":"1200.00","confidence":2.0,"evidence":"Invoice Amount 1200"}'
    assert StubResolver(raw).resolve(context).amount is None


def test_override_amount_is_normalized_before_grounding():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "1200.00", 0)], candidates=[Decimal("1200.00")],
    )
    raw = '{"decision":"override","amount":"1,200.00","confidence":0.9,"evidence":"Invoice Amount 1,200.00"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("1200.00")


# --- build_default_resolver key gating -------------------------------------

def test_build_default_resolver_none_without_key(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "GROQ_API_KEY", None, raising=False)
    assert build_default_resolver() is None


def test_build_default_resolver_instance_with_key(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "GROQ_API_KEY", "test-key", raising=False)
    assert isinstance(build_default_resolver(), LlmAmountResolver)


# --- service integration (real extractor, stubbed provider) ----------------

def _service(ocr_text: str, resolver) -> InvoiceVerificationService:
    extractor = InvoiceAmountExtractor(FakeOcrClient(text=ocr_text), AmountNormalizer())
    return InvoiceVerificationService(extractor, AmountValidator(), resolver=resolver)


def test_confident_result_never_calls_provider():
    stub = StubResolver('{"decision":"override","amount":"9999","confidence":1.0,"evidence":"x"}')
    service = _service("Grand Total Rs. 18,750.00", stub)

    result = service.verify(b"%PDF", PDF, Decimal("18750.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("18750.00")
    assert stub.calls == 0


def test_8jpg_style_regression_is_prevented_end_to_end():
    ocr = (
        "Invoice Amount: INR 47,925.00\n"
        "Total Amounts (INR) 38,991.00 8,933.68 47,924.68\n"
        "Invoice Total (in figures): INR 47,925.00"
    )
    stub = StubResolver('{"decision":"override","amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}')
    service = _service(ocr, stub)

    result = service.verify(b"%PDF", PDF, Decimal("47925.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("47925.00")  # deterministic winner kept
    assert stub.calls == 1


def test_valid_override_applied_end_to_end_same_priority_tie():
    # Two equal-priority "Grand Total" lines disagree -> ambiguous; the LLM
    # resolves to the grounded, equal-priority alternative.
    ocr = "Grand Total 1,000.00\nGrand Total 1,200.00"
    stub = StubResolver('{"decision":"override","amount":"1200.00","confidence":0.9,"evidence":"Grand Total 1,200.00"}')
    service = _service(ocr, stub)

    result = service.verify(b"%PDF", PDF, Decimal("1200.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("1200.00")
    assert stub.calls == 1
