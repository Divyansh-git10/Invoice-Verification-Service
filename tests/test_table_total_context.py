"""Structural table-total context detection + its use in arbitration.

`in_table_total` flags a candidate that is the Total-column value of a line-item
table's totals row (columnar line + a table header above it). It is metadata
only: the deterministic winner is unchanged and all gates still apply. No
arithmetic, no digit correction, no invoice-specific rules. No real LLM calls.
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

PDF = "application/pdf"

# 4.jpg-style: item table with a columnar totals row (3,988.40) plus a footer
# "Total Amount After Tax" (3,988.00). No amount-in-words here.
TABLE_OCR = (
    "Sr No Name HSN / SAC Qty Rate Taxable Value IGST Amount Total\n"
    "1 Stanley Hammer 82052000 3.00 PCS 499.00 1,497.00 18.00 269.46 1,766.46\n"
    "2 Automatic Saw 8202 1.00 PCS 1,883.00 1,883.00 18.00 338.94 2,221.94\n"
    "Total 4.00 3,380.00 608.40 3,988.40\n"
    "Bank Details Total Amount After Tax 3,988.00\n"
)


def _extractor(ocr: str) -> InvoiceAmountExtractor:
    return InvoiceAmountExtractor(FakeOcrClient(text=ocr), AmountNormalizer())


def _ctx(ocr: str):
    o = _extractor(ocr).resolve(b"x", PDF)
    return o, ResolverContext(
        ocr_text=o.ocr_text, candidates=o.candidates, deterministic_winner=o.amount,
        winner_label=o.winner_label, winner_priority=o.winner_priority,
        labelled_candidates=o.labelled_candidates, amount_in_words=o.amount_in_words,
    )


def _service(ocr: str, resolver):
    return InvoiceVerificationService(_extractor(ocr), AmountValidator(), resolver=resolver)


def _by_amount(o):
    return {c.amount: c for c in o.labelled_candidates}


# --- detection: positive + negative ---------------------------------------

def test_positive_table_total_row_is_flagged():
    o, _ = _ctx(TABLE_OCR)
    cands = _by_amount(o)
    assert cands[Decimal("3988.40")].in_table_total is True   # columnar totals row under a header
    assert cands[Decimal("3988.00")].in_table_total is False  # single-value footer line


def test_deterministic_winner_unchanged_with_flag():
    o, _ = _ctx(TABLE_OCR)
    assert o.amount == Decimal("3988.00")          # winner selection unchanged
    assert o.winner_label == "total amount"
    assert _extractor(TABLE_OCR).extract(b"x", PDF).amount == Decimal("3988.00")


def test_negative_footer_only_total_not_flagged():
    o, _ = _ctx("Grand Total 5,000.00\n")
    assert _by_amount(o)[Decimal("5000.00")].in_table_total is False


def test_negative_columnar_without_header_not_flagged():
    # Columnar line but no table header above -> not a table total.
    o, _ = _ctx("Total 1,000.00 200.00 1,200.00\n")
    assert _by_amount(o)[Decimal("1200.00")].in_table_total is False


# --- prompt exposure -------------------------------------------------------

def test_flag_is_passed_to_prompt():
    _, ctx = _ctx(TABLE_OCR)
    prompt = LlmAmountResolver(api_key="x")._build_prompt(ctx)
    assert '"in_item_table_total": true' in prompt
    assert '"in_item_table_total": false' in prompt


def test_system_prompt_documents_flag():
    assert "in_item_table_total" in _SYSTEM_PROMPT
    assert "Total-column value of a line-item table" in _SYSTEM_PROMPT


# --- arbitration: table-total can be accepted; protections intact ---------

def test_table_total_override_accepted_when_winner_weak():
    # 4.jpg-style: weak winner label -> a grounded table-total override is allowed.
    stub = StubResolver(
        '{"selected_amount":"3988.40","confidence":0.9,'
        '"evidence":"Total 4.00 3,380.00 608.40 3,988.40"}'
    )
    result = _service(TABLE_OCR, stub).verify(b"x", PDF, Decimal("3988.40"))
    assert result.matched is True
    assert result.actual_amount == Decimal("3988.40")
    assert stub.calls == 1


def test_table_total_flag_does_not_weaken_8jpg_authority():
    ocr = (
        "No Name HSN SAC Qty Rate Taxable Value Total\n"
        "1 Widget 111 1.00 PCS 1000.00 1000.00 18.00 180.00 1180.00\n"
        "Total Amounts (INR) 38,991.00 8,933.68 47,924.68\n"
        "Invoice Amount: INR 47,925.00\n"
    )
    o, _ = _ctx(ocr)
    assert _by_amount(o)[Decimal("47924.68")].in_table_total is True  # it IS a table total
    stub = StubResolver('{"selected_amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}')
    result = _service(ocr, stub).verify(b"x", PDF, Decimal("47925.00"))
    assert result.actual_amount == Decimal("47925.00")  # authority veto still wins


def test_table_total_flag_does_not_weaken_13jpg_tax_protection():
    ocr = (
        "TOTAL 15KG Rs. 1525.00\n"
        "HSN Taxable Amount Total Tax Amount Rate\n"
        "808 Rs. 500.00 2.5% Rs. 12.50 2.5% Rs. 12.50 Rs. 25.00\n"
        "Total Rs. 1500.00 Rs. 37.50 Rs. 37.50 Rs. 75.00\n"
    )
    o, _ = _ctx(ocr)
    assert o.amount == Decimal("1525.00")
    assert _by_amount(o)[Decimal("1500.00")].in_table_total is True  # tax-table total
    stub = StubResolver('{"selected_amount":"1500.00","confidence":0.99,"evidence":"Total Rs. 1500.00"}')
    result = _service(ocr, stub).verify(b"x", PDF, Decimal("1525.00"))
    assert result.actual_amount == Decimal("1525.00")  # tax-context veto still wins
