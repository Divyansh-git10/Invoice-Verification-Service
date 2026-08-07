from decimal import Decimal

import pytest

from app.core.exceptions import (
    AmountNotFoundException,
    ExtractionException,
    FileTooLargeException,
    OcrExecutionException,
    UnsupportedFileTypeException,
)
from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.models.extracted_amount import ExtractedAmount
from app.utils.amount_normalizer import AmountNormalizer
from tests.fakes import FakeOcrClient

PDF = "application/pdf"


def make_extractor(text: str = "", raises: Exception | None = None, **kwargs):
    return InvoiceAmountExtractor(
        ocr_client=FakeOcrClient(text=text, raises=raises),
        normalizer=AmountNormalizer(),
        **kwargs,
    )


def test_extracts_grand_total():
    text = "ACME Traders\nGrand Total : Rs. 18,750.00\nThank you"
    extractor = make_extractor(text)

    result = extractor.extract(b"%PDF-fake", PDF)

    assert isinstance(result, ExtractedAmount)
    assert result.amount == Decimal("18750.00")


def test_prefers_grand_total_over_subtotal_and_tax():
    text = "Sub Total 15,000.00\nGST @18% 2,700.00\nGrand Total 17,700.00"
    extractor = make_extractor(text)

    assert extractor.extract(b"x", PDF).amount == Decimal("17700.00")


def test_picks_highest_priority_keyword():
    # "Total" appears before the more specific "Amount Payable" line;
    # the more specific keyword must win regardless of order.
    text = "Total 100.00\nAmount Payable 118.00"
    extractor = make_extractor(text)

    assert extractor.extract(b"x", PDF).amount == Decimal("118.00")


def test_takes_largest_figure_on_a_total_line():
    text = "Grand Total  1 x 250.00   5,000.00"
    extractor = make_extractor(text)

    assert extractor.extract(b"x", PDF).amount == Decimal("5000.00")


def test_fallback_to_largest_monetary_value_when_no_keyword():
    text = "Widget A 1,200.00\nWidget B 3,450.00"
    extractor = make_extractor(text)

    assert extractor.extract(b"x", PDF).amount == Decimal("3450.00")


def test_amount_not_found_raises():
    extractor = make_extractor("No monetary values in this document at all")

    with pytest.raises(AmountNotFoundException):
        extractor.extract(b"x", PDF)


def test_unsupported_mime_type_raises():
    extractor = make_extractor("Grand Total 10.00")

    with pytest.raises(UnsupportedFileTypeException):
        extractor.extract(b"x", "text/plain")


def test_empty_file_raises():
    extractor = make_extractor("Grand Total 10.00")

    with pytest.raises(ExtractionException):
        extractor.extract(b"", PDF)


def test_file_too_large_raises():
    extractor = make_extractor("Grand Total 10.00", max_file_size_mb=1)
    oversized = b"0" * (1 * 1024 * 1024 + 1)

    with pytest.raises(FileTooLargeException):
        extractor.extract(oversized, PDF)


def test_ocr_failure_is_wrapped_as_ocr_execution_exception():
    extractor = make_extractor(raises=RuntimeError("engine crashed"))

    with pytest.raises(OcrExecutionException):
        extractor.extract(b"x", PDF)


def test_domain_extraction_error_from_ocr_propagates():
    extractor = make_extractor(raises=OcrExecutionException("boom"))

    with pytest.raises(OcrExecutionException):
        extractor.extract(b"x", PDF)


def test_ignores_dates_and_ids_via_keyword_targeting():
    text = "Invoice No 123456\nDate 01.02.2024\nGrand Total 2,499.00"
    extractor = make_extractor(text)

    assert extractor.extract(b"x", PDF).amount == Decimal("2499.00")
