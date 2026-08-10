from decimal import Decimal

import pytest

from app.core.exceptions import AmountNotFoundException
from app.models.extracted_amount import ExtractedAmount
from app.models.validation_result import ValidationResult
from app.services.invoice_verification_service import InvoiceVerificationService

PDF = "application/pdf"


class FakeExtractor:
    """Test double for InvoiceAmountExtractor.

    Returns a canned ExtractedAmount, or raises a canned exception. Records
    every call so the service's orchestration can be asserted.
    """

    def __init__(self, amount: Decimal | None = None, raises: Exception | None = None):
        self._amount = amount
        self._raises = raises
        self.calls: list[tuple[bytes, str]] = []

    def extract(self, file_bytes: bytes, mime_type: str) -> ExtractedAmount:
        self.calls.append((file_bytes, mime_type))
        if self._raises is not None:
            raise self._raises
        return ExtractedAmount(amount=self._amount)


class SpyValidator:
    """Test double for AmountValidator.

    Records the (expected, actual) pair it is called with. Returns a preset
    ValidationResult if one is provided; otherwise computes exact equality so
    matching/mismatching flows can be exercised without the real validator.
    """

    def __init__(self, result: ValidationResult | None = None):
        self._result = result
        self.calls: list[tuple[Decimal, Decimal]] = []

    def validate(self, expected_amount: Decimal, actual_amount: Decimal) -> ValidationResult:
        self.calls.append((expected_amount, actual_amount))
        if self._result is not None:
            return self._result
        return ValidationResult(
            matched=expected_amount == actual_amount,
            expected_amount=expected_amount,
            actual_amount=actual_amount,
        )


def test_matching_amounts_return_matched_result():
    extractor = FakeExtractor(amount=Decimal("18750.00"))
    validator = SpyValidator()
    service = InvoiceVerificationService(extractor, validator)

    result = service.verify(b"%PDF-bytes", PDF, Decimal("18750.00"))

    assert isinstance(result, ValidationResult)
    assert result.matched is True
    assert result.expected_amount == Decimal("18750.00")
    assert result.actual_amount == Decimal("18750.00")
    # Orchestration: extractor was called with the exact inputs.
    assert extractor.calls == [(b"%PDF-bytes", PDF)]


def test_mismatching_amounts_fraud_scenario():
    # Invoice actually reads 18750; user claims 12750.
    extractor = FakeExtractor(amount=Decimal("18750.00"))
    validator = SpyValidator()
    service = InvoiceVerificationService(extractor, validator)

    result = service.verify(b"img", "image/png", Decimal("12750.00"))

    assert result.matched is False
    assert result.expected_amount == Decimal("12750.00")
    assert result.actual_amount == Decimal("18750.00")
    assert validator.calls == [(Decimal("12750.00"), Decimal("18750.00"))]


def test_extraction_failure_prevents_validator_execution():
    extractor = FakeExtractor(raises=AmountNotFoundException("no total"))
    validator = SpyValidator()
    service = InvoiceVerificationService(extractor, validator)

    with pytest.raises(AmountNotFoundException):
        service.verify(b"x", PDF, Decimal("18750.00"))

    # The validator must never run when extraction fails.
    assert validator.calls == []


def test_service_passes_extracted_amount_to_validator():
    # Distinct sentinel result to prove the service returns exactly what the
    # validator produced, and passes the extracted amount as `actual_amount`.
    sentinel = ValidationResult(
        matched=True,
        expected_amount=Decimal("999.00"),
        actual_amount=Decimal("42000.75"),
    )
    extractor = FakeExtractor(amount=Decimal("42000.75"))
    validator = SpyValidator(result=sentinel)
    service = InvoiceVerificationService(extractor, validator)

    result = service.verify(b"data", PDF, Decimal("999.00"))

    # Extracted amount (42000.75) was forwarded as the validator's actual_amount.
    assert validator.calls == [(Decimal("999.00"), Decimal("42000.75"))]
    # Service returns the validator's result unchanged.
    assert result is sentinel
