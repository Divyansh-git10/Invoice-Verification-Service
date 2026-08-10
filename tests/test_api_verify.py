import shutil
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_verification_service
from app.core.exceptions import AmountNotFoundException
from app.main import app
from app.models.extracted_amount import ExtractedAmount
from app.services.invoice_verification_service import InvoiceVerificationService
from app.validators.amount_validator import AmountValidator

PDF = "application/pdf"


class FakeExtractor:
    """Fake extractor so API tests don't depend on Tesseract (except the
    explicit real-upload test). Returns a canned amount or raises."""

    def __init__(self, amount: Decimal | None = None, raises: Exception | None = None):
        self._amount = amount
        self._raises = raises

    def extract(self, file_bytes: bytes, mime_type: str) -> ExtractedAmount:
        if self._raises is not None:
            raise self._raises
        return ExtractedAmount(amount=self._amount)


@pytest.fixture
def client():
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _use_extractor(extractor) -> None:
    """Override the endpoint's service with one wired to a fake extractor
    and the real validator."""
    app.dependency_overrides[get_verification_service] = (
        lambda: InvoiceVerificationService(extractor, AmountValidator())
    )


def _post(client, expected_amount, file_bytes=b"%PDF-data", filename="invoice.pdf", content_type=PDF):
    return client.post(
        "/verify",
        files={"invoice_file": (filename, file_bytes, content_type)},
        data={"expected_amount": str(expected_amount)},
    )


def test_matching_amount_returns_200_matched_true(client):
    _use_extractor(FakeExtractor(amount=Decimal("18750.00")))

    r = _post(client, "18750.00")

    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is True
    assert Decimal(str(body["expected_amount"])) == Decimal("18750.00")
    assert Decimal(str(body["actual_amount"])) == Decimal("18750.00")


def test_mismatch_returns_200_matched_false(client):
    # Business mismatch is a successful verification, not an error.
    _use_extractor(FakeExtractor(amount=Decimal("18750.00")))

    r = _post(client, "12750.00")

    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is False
    assert Decimal(str(body["expected_amount"])) == Decimal("12750.00")
    assert Decimal(str(body["actual_amount"])) == Decimal("18750.00")


def test_extraction_failure_no_total_returns_422(client):
    _use_extractor(FakeExtractor(raises=AmountNotFoundException("no total found")))

    r = _post(client, "18750.00")

    assert r.status_code == 422
    assert r.json()["detail"]["type"] == "amount_not_found"


def test_unsupported_mime_type_returns_415(client):
    # No override: the real default extractor rejects the MIME type before OCR.
    r = _post(client, "18750.00", file_bytes=b"hello", filename="note.txt", content_type="text/plain")

    assert r.status_code == 415
    assert r.json()["detail"]["type"] == "unsupported_file_type"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_invoice_upload_multipart(client):
    # End-to-end through the real default service (Tesseract) via multipart.
    path = "tests/fixtures/invoices/invoice_01.png"  # clean Grand Total 30798.00
    with open(path, "rb") as fh:
        data = fh.read()

    r = client.post(
        "/verify",
        files={"invoice_file": ("invoice_01.png", data, "image/png")},
        data={"expected_amount": "30798.00"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is True
    assert Decimal(str(body["actual_amount"])) == Decimal("30798.00")
