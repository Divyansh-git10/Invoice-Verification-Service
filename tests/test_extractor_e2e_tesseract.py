"""Extraction end-to-end tests with the REAL Tesseract engine.

These are EXTRACTION end-to-end tests only, not HTTP end-to-end tests. A
rendered invoice image is passed through the real `TesseractOcrClient` and
the real extractor/parser/normalizer, asserting the outcome. No fakes.

The extractor is not yet wired into the API (the endpoint is still `501`);
HTTP end-to-end tests are part of Component 5. These tests exercise only
what the extraction component owns, end to end, against real OCR.

Skipped automatically if the `tesseract` binary is not installed.
"""
import io
import shutil
from decimal import Decimal

import pytest

pytest.importorskip("PIL")
pytest.importorskip("pytesseract")

if shutil.which("tesseract") is None:
    pytest.skip("tesseract binary not installed", allow_module_level=True)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.core.exceptions import AmountNotFoundException  # noqa: E402
from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor  # noqa: E402
from app.extractors.tesseract_ocr_client import TesseractOcrClient  # noqa: E402
from app.utils.amount_normalizer import AmountNormalizer  # noqa: E402


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _render_invoice_png(lines: list[str]) -> bytes:
    width, height = 900, 90 + 60 * len(lines)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(34)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 60
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _real_extractor() -> InvoiceAmountExtractor:
    return InvoiceAmountExtractor(
        ocr_client=TesseractOcrClient(),
        normalizer=AmountNormalizer(),
    )


def test_end_to_end_real_ocr_reads_grand_total():
    png = _render_invoice_png(
        [
            "ACME Traders Pvt Ltd",
            "Invoice No: INV-2024-0091",
            "Sub Total: 15000.00",
            "GST 18%: 2700.00",
            "Grand Total: 17700.00",
        ]
    )

    result = _real_extractor().extract(png, "image/png")

    assert result.amount == Decimal("17700.00")


def test_end_to_end_real_ocr_simple_total():
    png = _render_invoice_png(["Total Amount: 18750.00"])

    result = _real_extractor().extract(png, "image/png")

    assert result.amount == Decimal("18750.00")


def test_extraction_failure_path_no_identifiable_total():
    """OCR succeeds but no invoice total exists -> AmountNotFoundException.

    Validates the architectural decision that an EXTRACTION failure
    (document read fine, but no total could be identified) is a distinct
    category from a validation failure. The document contains readable
    prose and no monetary figures at all.
    """
    png = _render_invoice_png(
        [
            "ACME Traders Pvt Ltd",
            "Thank you for your business",
            "Please retain this note for your records",
        ]
    )

    # Confirm OCR itself succeeded (produced readable text)...
    recognized = TesseractOcrClient().extract_text(png, "image/png")
    assert recognized.strip() != ""

    # ...yet the extractor reports no identifiable total.
    with pytest.raises(AmountNotFoundException):
        _real_extractor().extract(png, "image/png")
