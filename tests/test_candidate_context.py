"""Component: enriched candidate context for arbitration.

Verifies that labelled candidates now carry position + surrounding OCR context,
that the context reaches the resolver, that it distinguishes an invoice-level
total from a tax-section total, and that deterministic amounts / 8.jpg
protection are unchanged. No real LLM calls.
"""
from decimal import Decimal

import pytest

from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeAmountResolver, FakeOcrClient
from tests.test_llm_amount_resolver import StubResolver

PDF = "application/pdf"

# 13.jpg-style: an invoice-level TOTAL followed by a CGST/SGST tax summary that
# also contains a "Total" row with a different value.
INVOICE_VS_TAX_OCR = (
    "1 Apple 808 5KG Rs. 100.00 Rs. 5.00 (5%) Rs. 525.00\n"
    "Discount = Rs. 50.00\n"
    "TOTAL 15KG Rs. 80.00 Rs. 1525.00\n"
    "CGST SGST\n"
    "HSN Taxable Amount Total Tax Amount\n"
    "808 Rs. 500.00 2.5% Rs. 12.50 2.5% Rs. 12.50 Rs. 25.00\n"
    "Total Rs. 1500.00 Rs. 37.50 Rs. 37.50 Rs. 75.00\n"
)


def _extractor(ocr: str) -> InvoiceAmountExtractor:
    return InvoiceAmountExtractor(FakeOcrClient(text=ocr), AmountNormalizer())


def _service(ocr: str, resolver) -> InvoiceVerificationService:
    return InvoiceVerificationService(_extractor(ocr), AmountValidator(), resolver=resolver)


def test_labelled_candidates_carry_position_and_context():
    outcome = _extractor(INVOICE_VS_TAX_OCR).resolve(b"%PDF", PDF)
    by_amount = {c.amount: c for c in outcome.labelled_candidates}

    assert Decimal("1525.00") in by_amount
    assert Decimal("1500.00") in by_amount

    # position is recorded and ordered
    assert by_amount[Decimal("1525.00")].line_index < by_amount[Decimal("1500.00")].line_index

    # the tax-section total's context exposes tax markers; the invoice total's
    # own line does not sit inside the tax breakdown.
    tax_ctx = by_amount[Decimal("1500.00")].context.lower()
    assert "cgst" in tax_ctx or "tax" in tax_ctx or "hsn" in tax_ctx
    assert by_amount[Decimal("1525.00")].context != by_amount[Decimal("1500.00")].context


def test_deterministic_winner_unchanged():
    outcome = _extractor(INVOICE_VS_TAX_OCR).resolve(b"%PDF", PDF)
    assert outcome.amount == Decimal("1525.00")
    assert outcome.confident is False  # two competing "total" values -> ambiguous
    # extract() (used by the deterministic runner) is unaffected.
    assert _extractor(INVOICE_VS_TAX_OCR).extract(b"%PDF", PDF).amount == Decimal("1525.00")


def test_context_is_passed_to_resolver():
    spy = FakeAmountResolver()  # returns keep (amount=None)
    _service(INVOICE_VS_TAX_OCR, spy).verify(b"%PDF", PDF, Decimal("1525.00"))

    assert len(spy.calls) == 1
    ctx = spy.calls[0]
    contexts = {c.amount: c.context for c in ctx.labelled_candidates}
    assert contexts[Decimal("1500.00")]  # non-empty
    assert "tax" in contexts[Decimal("1500.00")].lower() or "cgst" in contexts[Decimal("1500.00")].lower()


def test_13jpg_style_keep_retains_invoice_total():
    # With the richer context the model should keep; keep -> deterministic 1525.
    stub = StubResolver('{"decision":"keep","amount":null,"confidence":0.0,"evidence":null}')
    result = _service(INVOICE_VS_TAX_OCR, stub).verify(b"%PDF", PDF, Decimal("1525.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("1525.00")


def test_13jpg_override_to_winner_is_kept():
    # Even if the model 'overrides' to the winner value, gate 5 keeps 1525.
    stub = StubResolver('{"decision":"override","amount":"1525.00","confidence":0.9,"evidence":"TOTAL ... 1525.00"}')
    result = _service(INVOICE_VS_TAX_OCR, stub).verify(b"%PDF", PDF, Decimal("1525.00"))
    assert result.actual_amount == Decimal("1525.00")


def test_8jpg_protection_still_intact_with_context():
    ocr = (
        "Invoice Amount: INR 47,925.00\n"
        "Total Amounts (INR) 38,991.00 8,933.68 47,924.68\n"
        "Invoice Total (in figures): INR 47,925.00"
    )
    stub = StubResolver('{"decision":"override","amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}')
    result = _service(ocr, stub).verify(b"%PDF", PDF, Decimal("47925.00"))
    assert result.actual_amount == Decimal("47925.00")  # lower-priority override rejected


def test_context_default_backward_compatible():
    # LabelledCandidate still constructs with the original fields only.
    from app.extractors.invoice_amount_extractor import LabelledCandidate

    c = LabelledCandidate(label="total", amount=Decimal("10"), line="Total 10", priority=11)
    assert c.line_index == -1
    assert c.context == ""
