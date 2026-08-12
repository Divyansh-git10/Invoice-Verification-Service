"""Component 6 - Extractor Hardening regression tests.

Focused, deterministic tests (fake OCR text, no Tesseract) for:
  - explicit final-total label priority
  - the amount-in-words fallback
Includes regressions built from the REAL invoice OCR text investigated
(Case A: 8.png; Case B: 3.jpg and 9.png).
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


# --- Part 1: final-total label priority -----------------------------------

def test_invoice_amount_beats_lower_level_total():
    text = (
        "Invoice Amount: INR 47,925.00\n"
        "Total Amounts (INR) 38,991.00 8,933.68 47,924.68\n"
        "Rounding 0.32"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("47925.00")


def test_invoice_total_beats_total_amounts_rollup():
    text = (
        "Total Amounts (INR) 38,991.00 8,933.68 47,924.68\n"
        "Invoice Total (in figures): INR 47,925.00"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("47925.00")


def test_grand_total_behavior_still_works():
    text = "ACME Traders\nGrand Total : Rs. 18,750.00\nThank you"
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("18750.00")


def test_subtotal_does_not_win_via_generic_total():
    # "Sub-Total" must not be treated as the final total; the amount-in-words
    # phrase carries the real value.
    text = (
        "Sub-Total: 512784 | 3802584\n"
        "Invoice Total in Word Total Amount\n"
        "Rupees ThirtyEight Thousand TwentySix Only"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("38026.00")


# --- Part 2: amount-in-words fallback -------------------------------------

def test_total_in_words_extracts_4490():
    text = "Total in words : FOUR THOUSAND FOUR HUNDRED AND NINETY RUPEES ."
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("4490.00")


def test_invoice_total_in_word_extracts_38026():
    text = (
        "Invoice Total in Word Total Amount\n"
        "Rupees ThirtyEight Thousand TwentySix Only"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("38026.00")


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("FOUR THOUSAND FOUR HUNDRED AND NINETY RUPEES", Decimal("4490.00")),
        ("Rupees ThirtyEight Thousand TwentySix Only", Decimal("38026.00")),
        ("Forty seven thousand, nine hundred and twenty five", Decimal("47925.00")),
        ("One Lakh Twenty Three Thousand Four Hundred Fifty", Decimal("123450.00")),
        ("Nine Hundred Sixty Eight Only", Decimal("968.00")),
    ],
)
def test_words_to_amount_parses_common_phrases(phrase, expected):
    assert words_to_amount(phrase) == expected


@pytest.mark.parametrize(
    "phrase",
    ["", "Rupees Only", "some garbled text here", "TAX INVOICE ORIGINAL"],
)
def test_words_to_amount_returns_none_when_unparseable(phrase):
    assert words_to_amount(phrase) is None


def test_unparseable_amount_in_words_does_not_invent_a_value():
    # An "in words" label with no valid number words, and no numeric total,
    # must NOT invent an amount.
    from app.core.exceptions import AmountNotFoundException

    text = "Total in words : SOME UNREADABLE OCR NOISE ONLY"
    with pytest.raises(AmountNotFoundException):
        make_extractor(text).extract(b"x", PDF)


# --- Part 3: existing behavior preserved ----------------------------------

def test_words_fallback_does_not_override_valid_numeric_total():
    # A clean labelled numeric total wins; the (garbled) words are ignored.
    text = (
        "Grand Total Rs. 5,000.00\n"
        "Total in words : FIVE THOUSAND RUPEES ONLY"
    )
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("5000.00")


def test_normalization_still_intact_for_labelled_total():
    text = "Grand Total Rs. 1,00,000/-"
    assert make_extractor(text).extract(b"x", PDF).amount == Decimal("100000")
