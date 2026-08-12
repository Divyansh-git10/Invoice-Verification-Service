"""Regression tests for the two approved deterministic fixes (post-Component 6).

FIX 1: currency-anchored amount-in-words ("Rs./Rupees/INR ... Only") used as a
       fallback when no labelled numeric total is found (1.jpg -> 105200.00).
FIX 2: "total sale"/"total sales" treated as subordinate so a pre-tax
       "Total Sale" cannot beat the real "Total" (5.jpg -> 30180.00).

Deterministic, fake-OCR text (no Tesseract).
"""
from decimal import Decimal

import pytest

from app.extractors.invoice_amount_extractor import (
    InvoiceAmountExtractor,
    words_to_amount,
)
from app.utils.amount_normalizer import AmountNormalizer
from tests.fakes import FakeOcrClient

PDF = "application/pdf"


def make_extractor(text: str):
    return InvoiceAmountExtractor(
        ocr_client=FakeOcrClient(text=text),
        normalizer=AmountNormalizer(),
    )


# --- FIX 1: currency-anchored amount-in-words -----------------------------

def test_currency_anchored_words_when_no_numeric_total():
    # Mirrors 1.jpg: no surviving total-keyword line; the words phrase carries
    # the real total. (The 106200.00 line-item must not be picked.)
    text = (
        "Item Description 3923 1020 100000.00 10.00 (%) 18.00 106200.00\n"
        "Discount - 1000.00\n"
        "1,05,200.00\n"
        "Rs. One Lakh Five Thousand Two Hundred Only\n"
        "Sale @18% = 90000.00, CGST = 8100.00, SGST = 8100.00"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("105200.00")


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("Rs. One Lakh Five Thousand Two Hundred Only", Decimal("105200.00")),
        ("Rupees Thirty Eight Thousand Only", Decimal("38000.00")),
        ("INR Ninety Nine Thousand Nine Hundred Ninety Nine Only", Decimal("99999.00")),
    ],
)
def test_currency_anchored_words_phrases(phrase, expected):
    # Fallback path: no numeric total, just the anchored words line.
    assert make_extractor(phrase).extract(b"x", PDF).amount == expected


def test_currency_anchor_does_not_fire_on_numeric_only():
    # "Rs. ... Only" with digits (no number-words) must NOT be treated as a
    # worded total; falls through to the strict numeric fallback.
    text = "Received Rs. 1,000.00 Only\nGrand Total Rs. 5,000.00"
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("5000.00")


def test_words_still_do_not_override_a_valid_numeric_total():
    # A reliable labelled numeric total wins; the anchored words are ignored.
    text = (
        "Grand Total Rs. 5,000.00\n"
        "Rs. Ten Thousand Only"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("5000.00")


def test_existing_in_words_label_still_works():
    text = "Total in words : FOUR THOUSAND FOUR HUNDRED AND NINETY RUPEES ."
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("4490.00")


# --- FIX 2: Total Sale suppression ----------------------------------------

def test_total_sale_does_not_beat_real_total():
    # Mirrors 5.jpg ordering: "Total Sale = 28000.00" appears before
    # "Total 30,180.00" but must not win.
    text = (
        "Total Sale = 28000.00, Tax = 2180.00, Cess = 0.00\n"
        "Total 30,180.00"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("30180.00")


def test_total_sales_plural_is_also_subordinate():
    text = (
        "Total Sales 28000.00\n"
        "Grand Total 30,180.00"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("30180.00")


def test_grand_total_and_subordinate_suppression_unchanged():
    # Existing behavior preserved: sub-total/tax still suppressed.
    text = "Sub Total 15,000.00\nGST @18% 2,700.00\nGrand Total 17,700.00"
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("17700.00")
