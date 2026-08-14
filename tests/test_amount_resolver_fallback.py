"""Component 7 - optional LLM amount-resolution fallback (plumbing only).

Tests the ResolvedAmount model invariants and the service's fallback
orchestration using a fake resolver (no LLM / no network). Deterministic
confidence is exercised through the REAL extractor with a fake OCR client.
"""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.core.exceptions import AmountNotFoundException
from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.models.resolved_amount import ResolvedAmount
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeAmountResolver, FakeOcrClient

PDF = "application/pdf"


def build_service(ocr_text: str, resolver=None) -> InvoiceVerificationService:
    extractor = InvoiceAmountExtractor(FakeOcrClient(text=ocr_text), AmountNormalizer())
    return InvoiceVerificationService(extractor, AmountValidator(), resolver=resolver)


# --- ResolvedAmount model invariants --------------------------------------

def test_null_amount_is_valid():
    r = ResolvedAmount(amount=None, confidence=0.0, evidence=None)
    assert r.amount is None


def test_amount_with_evidence_is_valid():
    r = ResolvedAmount(amount=Decimal("10.00"), confidence=0.9, evidence="Total 10.00")
    assert r.amount == Decimal("10.00")


@pytest.mark.parametrize("evidence", [None, "", "   "])
def test_amount_without_evidence_is_rejected(evidence):
    with pytest.raises(ValidationError):
        ResolvedAmount(amount=Decimal("10.00"), confidence=0.9, evidence=evidence)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_out_of_range_is_rejected(confidence):
    with pytest.raises(ValidationError):
        ResolvedAmount(amount=None, confidence=confidence, evidence=None)


# --- Service fallback orchestration ---------------------------------------

def test_confident_result_bypasses_resolver():
    # Keyword-labelled total => confident => resolver must never run.
    resolver = FakeAmountResolver(
        result=ResolvedAmount(amount=Decimal("9999"), confidence=1.0, evidence="x")
    )
    service = build_service("Grand Total Rs. 5,000.00", resolver=resolver)

    result = service.verify(b"%PDF", PDF, Decimal("5000.00"))

    assert result.matched is True
    assert result.actual_amount == Decimal("5000.00")
    assert resolver.calls == []


def test_low_confidence_calls_resolver_and_uses_its_amount():
    # No keyword / no words => strict-fallback guess (3450) => not confident.
    resolver = FakeAmountResolver(
        result=ResolvedAmount(
            amount=Decimal("9999.00"), confidence=0.9, evidence="Widget B 3,450.00"
        )
    )
    service = build_service("Widget A 1,200.00\nWidget B 3,450.00", resolver=resolver)

    # Expected matches the RESOLVER amount, not the deterministic guess.
    result = service.verify(b"%PDF", PDF, Decimal("9999.00"))

    assert result.matched is True
    assert result.actual_amount == Decimal("9999.00")
    assert len(resolver.calls) == 1


def test_resolver_returns_none_yields_amount_not_found():
    resolver = FakeAmountResolver(
        result=ResolvedAmount(amount=None, confidence=0.0, evidence=None)
    )
    service = build_service("no monetary values in this document at all", resolver=resolver)

    with pytest.raises(AmountNotFoundException):
        service.verify(b"%PDF", PDF, Decimal("1.00"))
    assert len(resolver.calls) == 1


def test_resolver_error_degrades_to_deterministic_guess():
    # Low-confidence deterministic guess exists; resolver raises -> keep guess.
    resolver = FakeAmountResolver(raises=TimeoutError("slow"))
    service = build_service("Widget A 1,200.00\nWidget B 3,450.00", resolver=resolver)

    result = service.verify(b"%PDF", PDF, Decimal("3450.00"))

    assert result.matched is True
    assert result.actual_amount == Decimal("3450.00")
    assert len(resolver.calls) == 1


def test_resolver_error_with_no_deterministic_result_is_amount_not_found():
    resolver = FakeAmountResolver(raises=RuntimeError("boom"))
    service = build_service("no monetary values here", resolver=resolver)

    with pytest.raises(AmountNotFoundException):
        service.verify(b"%PDF", PDF, Decimal("1.00"))


def test_no_resolver_preserves_deterministic_behavior():
    # Without a resolver, the low-confidence guess is used exactly as before.
    service = build_service("Widget A 1,200.00\nWidget B 3,450.00", resolver=None)
    result = service.verify(b"%PDF", PDF, Decimal("3450.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("3450.00")


def test_no_resolver_confident_path_unchanged():
    service = build_service("Grand Total Rs. 5,000.00", resolver=None)
    result = service.verify(b"%PDF", PDF, Decimal("5000.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("5000.00")
