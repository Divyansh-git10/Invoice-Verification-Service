class InvoiceVerificationException(Exception):
    """Base exception for the Invoice Verification Service."""


class ExtractionException(InvoiceVerificationException):
    """Raised when invoice amount extraction fails."""


class UnsupportedFileTypeException(ExtractionException):
    """Raised when an unsupported file type is uploaded."""


class FileTooLargeException(ExtractionException):
    """Raised when uploaded file exceeds configured size."""


class OcrExecutionException(ExtractionException):
    """Raised when the OCR engine fails to process the document."""


class AmountNotFoundException(ExtractionException):
    """Raised when invoice amount cannot be identified."""
