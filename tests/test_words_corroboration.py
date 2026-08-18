"""7.jpg-class recovery: a cross-grounded amount-in-words value that disagrees
with a single labelled total escalates the case and may be accepted by the
arbitrator. No real LLM calls (provider stubbed).

Uses synthetic OCR that mirrors 7.jpg (this sandbox's Tesseract drops the words
line, so the real image can't demonstrate it here).
"""
from decimal import Decimal

from app.extractors.invoice_amount_extractor import (
    InvoiceAmountExtractor,
    words_to_amount,
)
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeAmountResolver, FakeOcrClient
from tests.test_llm_amount_resolver import StubResolver

PDF = "application/pdf"

# 7.jpg-style: labelled Total is corrupted (965) while the amount-in-words
# ("Nine Hundred Sixty-eight ... Only" = 968) is corroborated numerically
# (Amount: 968.00).
SEVEN_STYLE_OCR = (
    "Sleek Bill\n"
    "Total: € 965.00\n"
    "Sub Total: € 900.00\n"
    "Tax Amount: 68.00\n"
    "Amount: € 968.00\n"
    "Nine Hundred Sixty-eight And Zero Paisa Only\n"
)

# 11.jpg-style: labelled Total Amount After Tax is corrupted (24490); the
# correct 4490 is only an unlabelled token and there is NO words corroboration.
ELEVEN_STYLE_OCR = (
    "Taxable Amount 3,805.00\n"
    "Add : IGST 684.90\n"
    "Total Tax 684.90\n"
    "Total Amount After Tax 24,490.00\n"
    "4,490.00\n"
)


def _extractor(ocr: str) -> InvoiceAmountExtractor:
    return InvoiceAmountExtractor(FakeOcrClient(text=ocr), AmountNormalizer())


def _service(ocr: str, resolver) -> InvoiceVerificationService:
    return InvoiceVerificationService(_extractor(ocr), AmountValidator(), resolver=resolver)


# --- extractor: routing / corroboration -----------------------------------

def test_words_phrase_parses_correctly():
    assert words_to_amount("Nine Hundred Sixty-eight And Zero Paisa") == Decimal("968.00")


def test_single_labelled_plus_corroborated_words_conflict_is_ambiguous():
    outcome = _extractor(SEVEN_STYLE_OCR).resolve(b"%PDF", PDF)
    assert outcome.amount == Decimal("965.00")          # winner UNCHANGED
    assert outcome.confident is False                    # now escalates
    assert outcome.amount_in_words == Decimal("968.00")  # cross-grounded


def test_deterministic_winner_unchanged_for_seven_style():
    assert _extractor(SEVEN_STYLE_OCR).extract(b"%PDF", PDF).amount == Decimal("965.00")


def test_words_not_present_numerically_is_not_corroborated():
    # Words parse to 968 but 968 never appears as a numeric token -> not trusted.
    ocr = "Total: € 965.00\nNine Hundred Sixty-eight Only\n"
    outcome = _extractor(ocr).resolve(b"%PDF", PDF)
    assert outcome.amount == Decimal("965.00")
    assert outcome.amount_in_words is None
    assert outcome.confident is True  # single labelled, no corroborated conflict


def test_words_agree_with_labelled_winner_stays_confident():
    ocr = "Total: € 968.00\nNine Hundred Sixty-eight Only\n"
    outcome = _extractor(ocr).resolve(b"%PDF", PDF)
    assert outcome.amount == Decimal("968.00")
    assert outcome.amount_in_words == Decimal("968.00")
    assert outcome.confident is True  # words agree -> no escalation


# --- service: escalation + arbitration ------------------------------------

def test_corroborated_words_override_is_accepted():
    stub = StubResolver(
        '{"selected_amount":"968.00","confidence":0.9,'
        '"evidence":"Nine Hundred Sixty-eight And Zero Paisa Only"}'
    )
    result = _service(SEVEN_STYLE_OCR, stub).verify(b"%PDF", PDF, Decimal("968.00"))
    assert result.matched is True
    assert result.actual_amount == Decimal("968.00")
    assert stub.calls == 1


def test_words_agree_confident_resolver_not_called():
    spy = FakeAmountResolver(
        result=None  # would be a keep, but must not even be called
    )
    ocr = "Total: € 968.00\nNine Hundred Sixty-eight Only\n"
    result = _service(ocr, spy).verify(b"%PDF", PDF, Decimal("968.00"))
    assert result.matched is True
    assert spy.calls == []  # confident -> resolver bypassed


# --- 11.jpg-style must stay failing (no corroboration) --------------------

def test_eleven_style_no_words_stays_confident_and_no_override():
    outcome = _extractor(ELEVEN_STYLE_OCR).resolve(b"%PDF", PDF)
    assert outcome.amount == Decimal("24490.00")  # winner unchanged
    assert outcome.amount_in_words is None
    assert outcome.confident is True              # not escalated


def test_eleven_style_resolver_not_called():
    stub = StubResolver(
        '{"selected_amount":"4490.00","confidence":1.0,"evidence":"4,490.00"}'
    )
    result = _service(ELEVEN_STYLE_OCR, stub).verify(b"%PDF", PDF, Decimal("4490.00"))
    assert result.matched is False                       # honest failure
    assert result.actual_amount == Decimal("24490.00")   # deterministic kept
    assert stub.calls == 0                               # confident -> bypass


# --- 8.jpg protection unaffected by the new corroboration branch -----------

def test_eight_style_protection_intact_even_with_agreeing_words():
    ocr = (
        "Invoice Amount: INR 47,925.00\n"
        "Total Amounts (INR) 38,991.00 8,933.68 47,924.68\n"
        "Invoice Total (in figures): INR 47,925.00\n"
        "Invoice Total amount in words: Forty seven thousand nine hundred and twenty five Only\n"
    )
    # LLM tries the roll-up 47924.68 (which is NOT the corroborated words value).
    stub = StubResolver(
        '{"selected_amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}'
    )
    result = _service(ocr, stub).verify(b"%PDF", PDF, Decimal("47925.00"))
    assert result.actual_amount == Decimal("47925.00")  # authority veto still wins
