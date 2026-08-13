from decimal import Decimal

from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.models.validation_result import ValidationResult
from app.validators.amount_validator import AmountValidator


class InvoiceVerificationService:
    """Orchestrates extraction then validation. If extraction raises, the
    exception propagates and the validator is not called."""

    def __init__(
        self,
        extractor: InvoiceAmountExtractor,
        validator: AmountValidator,
    ):
        self._extractor = extractor
        self._validator = validator

    def verify(
        self,
        file_bytes: bytes,
        mime_type: str,
        expected_amount: Decimal,
    ) -> ValidationResult:
        extracted = self._extractor.extract(file_bytes, mime_type)
        return self._validator.validate(expected_amount, extracted.amount)
