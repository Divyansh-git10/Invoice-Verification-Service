from decimal import Decimal

from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.models.validation_result import ValidationResult
from app.validators.amount_validator import AmountValidator


class InvoiceVerificationService:
    """Application orchestrator for the verification workflow.

    Coordinates the existing extractor and validator and nothing else:
    it contains no OCR, no validation, no HTTP, and no response mapping.
    Dependencies are supplied via constructor injection (no framework).

    The service is deliberately independent of FastAPI's `UploadFile`; the
    API layer converts the upload into raw bytes + MIME type before calling
    `verify`. This keeps the orchestrator pure and unit-testable.

    Flow:
        file bytes + MIME type + expected amount
            -> extractor.extract(...)  -> actual amount
            -> validator.validate(...) -> ValidationResult

    Error handling: if extraction raises, the exception propagates and the
    validator is never called (extraction failures are kept separate from
    business validation).
    """

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
        """Extract the invoice total and validate it against the expected amount.

        Args:
            file_bytes: Raw bytes of the uploaded invoice.
            mime_type: MIME type of the invoice.
            expected_amount: The user-entered amount to verify against.

        Returns:
            The `ValidationResult` produced by the validator.

        Raises:
            Any extraction exception raised by the extractor. When this
            happens the validator is not invoked.
        """
        extracted = self._extractor.extract(file_bytes, mime_type)
        return self._validator.validate(expected_amount, extracted.amount)
