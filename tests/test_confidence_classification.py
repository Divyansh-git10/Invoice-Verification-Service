"""Deterministic confidence classification (Component 7 refinement).

Explicit-but-conflicting labelled totals must be classified NOT confident
(so they can escalate to the resolver) WITHOUT changing the extracted amount.
Clean single-total / amount-in-words results stay confident (bypass).
"""
from decimal import Decimal

import pytest

from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.models.resolved_amount import ResolvedAmount
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeAmountResolver, FakeOcrClient

PDF = "application/pdf"


def outcome_for(text: str):
    extractor = InvoiceAmountExtractor(FakeOcrClient(text=text), AmountNormalizer())
    return extractor.resolve(b"%PDF", PDF)


def service_for(text: str, resolver=None):
    extractor = InvoiceAmountExtractor(FakeOcrClient(text=text), AmountNormalizer())
    return InvoiceVerificationService(extractor, AmountValidator(), resolver=resolver)


# --- classification (via extractor.resolve) -------------------------------

def test_single_labelled_total_is_confident():
    o = outcome_for("Grand Total Rs. 18,750.00")
    assert o.amount == Decimal("18750.00")
    assert o.confident is True


def test_same_value_on_two_labels_is_confident():
    o = outcome_for("Total Amount 5,000.00\nGrand Total 5,000.00")
    assert o.amount == Decimal("5000.00")
    assert o.confident is True


def test_conflicting_labelled_totals_are_ambiguous_amount_unchanged():
    # Mirrors 4.jpg: "Total Amount After Tax" (winner, priority) vs a generic
    # "Total" line with a different value.
    o = outcome_for("Total Amount After Tax 3,988.00\nTotal 3,988.40")
    assert o.amount == Decimal("3988.00")   # winner unchanged
    assert o.confident is False             # but now escalates


def test_amount_in_words_is_confident():
    o = outcome_for("Total in words : FOUR THOUSAND FOUR HUNDRED AND NINETY RUPEES .")
    assert o.amount == Decimal("4490.00")
    assert o.confident is True


def test_keyword_less_guess_is_not_confident():
    o = outcome_for("Widget A 1,200.00\nWidget B 3,450.00")
    assert o.amount == Decimal("3450.00")
    assert o.confident is False


def test_no_result_is_not_confident():
    o = outcome_for("no monetary values in this document at all")
    assert o.amount is None
    assert o.confident is False


# --- service escalation for ambiguous results -----------------------------

def test_ambiguous_result_escalates_to_resolver():
    resolver = FakeAmountResolver(
        result=ResolvedAmount(
            amount=Decimal("3988.40"), confidence=0.95, evidence="Total 3,988.40"
        )
    )
    service = service_for("Total Amount After Tax 3,988.00\nTotal 3,988.40", resolver=resolver)

    # Resolver's amount is used for the ambiguous case.
    result = service.verify(b"%PDF", PDF, Decimal("3988.40"))
    assert result.matched is True
    assert result.actual_amount == Decimal("3988.40")
    assert len(resolver.calls) == 1


def test_ambiguous_result_degrades_to_deterministic_winner_when_resolver_declines():
    resolver = FakeAmountResolver(
        result=ResolvedAmount(amount=None, confidence=0.0, evidence=None)
    )
    service = service_for("Total Amount After Tax 3,988.00\nTotal 3,988.40", resolver=resolver)

    # Resolver returns None -> keep the deterministic winner (unchanged amount).
    result = service.verify(b"%PDF", PDF, Decimal("3988.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("3988.00")
    assert len(resolver.calls) == 1


def test_confident_single_total_still_bypasses_resolver():
    resolver = FakeAmountResolver(
        result=ResolvedAmount(amount=Decimal("9999"), confidence=1.0, evidence="x")
    )
    service = service_for("Grand Total Rs. 18,750.00", resolver=resolver)

    result = service.verify(b"%PDF", PDF, Decimal("18750.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("18750.00")
    assert resolver.calls == []


def test_extract_amount_unchanged_for_conflicting_totals():
    # extract() (used by the deterministic regression runner) is unaffected.
    extractor = InvoiceAmountExtractor(
        FakeOcrClient(text="Total Amount After Tax 3,988.00\nTotal 3,988.40"),
        AmountNormalizer(),
    )
    assert extractor.extract(b"%PDF", PDF).amount == Decimal("3988.00")
