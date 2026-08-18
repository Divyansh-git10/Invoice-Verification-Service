"""Prompt/context alignment for the corroborated amount-in-words signal.

Verifies the prompt now exposes `corroborated_amount_in_words` with evidence and
that grounding / arbitration gates / protections are unchanged. No LLM calls.
"""
from decimal import Decimal

from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.resolvers.amount_resolver import ResolverContext
from app.resolvers.llm_amount_resolver import LlmAmountResolver, _SYSTEM_PROMPT
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeOcrClient
from tests.test_llm_amount_resolver import StubResolver
from tests.test_words_corroboration import ELEVEN_STYLE_OCR, SEVEN_STYLE_OCR

PDF = "application/pdf"


def _ctx(ocr: str):
    o = InvoiceAmountExtractor(FakeOcrClient(text=ocr), AmountNormalizer()).resolve(b"x", PDF)
    return o, ResolverContext(
        ocr_text=o.ocr_text, candidates=o.candidates, deterministic_winner=o.amount,
        winner_label=o.winner_label, winner_priority=o.winner_priority,
        labelled_candidates=o.labelled_candidates, amount_in_words=o.amount_in_words,
    )


def _prompt(ocr: str) -> str:
    _, ctx = _ctx(ocr)
    return LlmAmountResolver(api_key="x")._build_prompt(ctx)


# --- prompt exposure -------------------------------------------------------

def test_system_prompt_documents_corroborated_and_amends_bare_rule():
    assert "corroborated_amount_in_words" in _SYSTEM_PROMPT
    assert "cross-grounded, not bare" in _SYSTEM_PROMPT
    assert "NOT a bare number" in _SYSTEM_PROMPT


def test_system_prompt_requires_active_candidate_comparison():
    # Selection must compare all grounded candidates, not default to the winner.
    assert "Compare EVERY grounded candidate" in _SYSTEM_PROMPT
    assert "Do NOT pick the deterministic_winner merely because it is the default" in _SYSTEM_PROMPT
    # the LLM SELECTS the best-supported candidate (or null); it does not decide.
    assert "SELECT the single best-supported invoice-level FINAL PAYABLE" in _SYSTEM_PROMPT
    assert "select that candidate" in _SYSTEM_PROMPT
    # but do not force a change (protects 8.jpg / 13.jpg keep behaviour).
    assert "do not force a change" in _SYSTEM_PROMPT


def test_seven_style_prompt_exposes_corroborated_block_with_evidence():
    p = _prompt(SEVEN_STYLE_OCR)
    assert "corroborated_amount_in_words:" in p
    assert '"value": "968.00"' in p
    assert '"cross_grounded": true' in p
    assert "Amount: € 968.00" in p                              # numeric occurrence
    assert "Nine Hundred Sixty-eight And Zero Paisa Only" in p  # words phrase


def test_absent_corroboration_renders_none():
    assert "corroborated_amount_in_words: none" in _prompt(ELEVEN_STYLE_OCR)


# --- gates / grounding unchanged ------------------------------------------

def test_grounding_unchanged_ungrounded_override_still_rejected():
    _, ctx = _ctx(SEVEN_STYLE_OCR)
    raw = '{"selected_amount":"12345.00","confidence":1.0,"evidence":"x"}'
    assert StubResolver(raw).resolve(ctx).amount is None


def test_corroborated_value_still_subject_to_confidence_gate():
    _, ctx = _ctx(SEVEN_STYLE_OCR)
    raw = '{"selected_amount":"968.00","confidence":0.5,"evidence":"Nine Hundred ... Only"}'
    assert StubResolver(raw).resolve(ctx).amount is None  # below threshold -> keep


def test_corroborated_value_accepted_when_all_gates_pass():
    _, ctx = _ctx(SEVEN_STYLE_OCR)
    raw = '{"selected_amount":"968.00","confidence":0.9,"evidence":"Nine Hundred Sixty-eight And Zero Paisa Only"}'
    assert StubResolver(raw).resolve(ctx).amount == Decimal("968.00")


# --- protections unchanged -------------------------------------------------

def test_deterministic_amounts_unchanged():
    o7, _ = _ctx(SEVEN_STYLE_OCR)
    o11, _ = _ctx(ELEVEN_STYLE_OCR)
    assert o7.amount == Decimal("965.00")
    assert o11.amount == Decimal("24490.00")


def test_8jpg_protection_unchanged():
    ocr = (
        "Invoice Amount: INR 47,925.00\n"
        "Total Amounts (INR) 38,991.00 8,933.68 47,924.68\n"
        "Invoice Total (in figures): INR 47,925.00\n"
    )
    stub = StubResolver('{"selected_amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}')
    ex = InvoiceAmountExtractor(FakeOcrClient(text=ocr), AmountNormalizer())
    svc = InvoiceVerificationService(ex, AmountValidator(), resolver=stub)
    assert svc.verify(b"x", PDF, Decimal("47925.00")).actual_amount == Decimal("47925.00")


def test_13jpg_protection_unchanged():
    ocr = (
        "TOTAL 15KG Rs. 1525.00\n"
        "CGST SGST\n"
        "HSN Taxable Amount Total Tax Amount\n"
        "Total Rs. 1500.00 Rs. 37.50 Rs. 37.50 Rs. 75.00\n"
    )
    o, ctx = _ctx(ocr)
    assert o.amount == Decimal("1525.00")
    assert "corroborated_amount_in_words: none" in LlmAmountResolver(api_key="x")._build_prompt(ctx)
    stub = StubResolver('{"selected_amount":"1500.00","confidence":0.99,"evidence":"Total Rs. 1500.00"}')
    assert stub.resolve(ctx).amount is None  # equal-priority tax-context still rejected
