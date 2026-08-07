import re
from decimal import Decimal
from typing import Optional

from app.core.config import settings
from app.core.exceptions import (
    AmountNotFoundException,
    ExtractionException,
    FileTooLargeException,
    OcrExecutionException,
    UnsupportedFileTypeException,
)
from app.core.logger import get_logger
from app.extractors.ocr_client import OcrClient
from app.extractors.tesseract_ocr_client import TesseractOcrClient
from app.models.extracted_amount import ExtractedAmount
from app.utils.amount_normalizer import AmountNormalizer

logger = get_logger(__name__)


# Keywords that mark the invoice total, in priority order (most specific
# first). The extractor prefers the highest-priority keyword present.
_TOTAL_KEYWORDS = (
    "grand total",
    "total amount payable",
    "total payable",
    "amount payable",
    "net payable",
    "balance due",
    "amount due",
    "net amount",
    "total amount",
    "invoice total",
    "total",
)

# Loose money token: used on lines already identified by a total keyword,
# where surrounding context implies the number is monetary. Allows plain
# integers (e.g. "Total 18750"). The comma-grouped alternative requires at
# least one comma so an un-grouped integer falls through to the plain
# alternative and is matched in full (not truncated to its first 3 digits).
_LOOSE_AMOUNT = re.compile(
    r"(?:₹|rs\.?|inr)?\s*"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Strict money token: used for the keyword-less fallback. Requires a
# grouping comma or exactly two decimals so bare identifiers (phone
# numbers, invoice numbers) are not mistaken for amounts.
_STRICT_AMOUNT = re.compile(
    r"(?:₹|rs\.?|inr)?\s*"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{2})",
    re.IGNORECASE,
)


class InvoiceAmountExtractor:
    """Owns the complete invoice-amount extraction pipeline.

    Responsibilities: input validation, OCR (delegated to the injected
    `OcrClient` seam), identifying the invoice total in the recognized
    text, and normalizing it into a canonical `Decimal`.

    Dependencies are supplied via constructor injection (no framework),
    so tests can pass a fake OCR client and the Tesseract implementation
    can later be swapped for a cloud provider without touching this class.
    """

    def __init__(
        self,
        ocr_client: OcrClient,
        normalizer: AmountNormalizer,
        supported_mime_types: Optional[tuple[str, ...]] = None,
        max_file_size_mb: Optional[int] = None,
    ):
        self._ocr_client = ocr_client
        self._normalizer = normalizer
        self._supported_mime_types = (
            tuple(supported_mime_types)
            if supported_mime_types is not None
            else settings.SUPPORTED_MIME_TYPES
        )
        self._max_file_size_mb = (
            max_file_size_mb
            if max_file_size_mb is not None
            else settings.MAX_FILE_SIZE_MB
        )

    def extract(self, file_bytes: bytes, mime_type: str) -> ExtractedAmount:
        """Extract the invoice total from a document.

        Raises:
            UnsupportedFileTypeException: MIME type not allowed.
            FileTooLargeException: Document exceeds the configured size.
            ExtractionException: Empty/unreadable document.
            OcrExecutionException: OCR engine failure.
            AmountNotFoundException: No invoice total could be identified.
        """
        self._validate_mime(mime_type)
        self._validate_size(file_bytes)

        text = self._run_ocr(file_bytes, mime_type)

        amount = self._identify_total(text)
        if amount is None:
            raise AmountNotFoundException(
                "Unable to identify the invoice total in the document"
            )

        logger.info("Identified invoice total: %s", amount)
        return ExtractedAmount(amount=amount)

    # -- pipeline steps -------------------------------------------------

    def _validate_mime(self, mime_type: str) -> None:
        if mime_type not in self._supported_mime_types:
            raise UnsupportedFileTypeException(
                f"Unsupported file type: {mime_type!r}. "
                f"Supported: {', '.join(self._supported_mime_types)}"
            )

    def _validate_size(self, file_bytes: bytes) -> None:
        if not file_bytes:
            raise ExtractionException("Uploaded document is empty")

        max_bytes = self._max_file_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise FileTooLargeException(
                f"File is {len(file_bytes)} bytes; "
                f"limit is {self._max_file_size_mb} MB"
            )

    def _run_ocr(self, file_bytes: bytes, mime_type: str) -> str:
        try:
            text = self._ocr_client.extract_text(file_bytes, mime_type)
        except ExtractionException:
            # Already a domain extraction failure (e.g. OcrExecutionException).
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any client error
            logger.exception("OCR client raised an unexpected error")
            raise OcrExecutionException("OCR execution failed") from exc

        return text or ""

    def _identify_total(self, text: str) -> Optional[Decimal]:
        best_priority: Optional[int] = None
        best_amount: Optional[Decimal] = None

        for line in text.splitlines():
            lowered = line.lower()
            for priority, keyword in enumerate(_TOTAL_KEYWORDS):
                if keyword in lowered:
                    amounts = self._amounts_in(line, _LOOSE_AMOUNT)
                    if amounts and (
                        best_priority is None or priority < best_priority
                    ):
                        # On a total line, the total is the largest figure
                        # (e.g. it exceeds any per-item value on that line).
                        best_priority = priority
                        best_amount = max(amounts)
                    break  # highest-priority keyword on this line wins

        if best_amount is not None:
            return best_amount

        # Keyword-less fallback: the largest clearly-monetary figure found.
        fallback = self._amounts_in(text, _STRICT_AMOUNT)
        if fallback:
            logger.info("No total keyword found; using largest monetary value")
            return max(fallback)

        return None

    def _amounts_in(self, text: str, pattern: re.Pattern) -> list[Decimal]:
        amounts: list[Decimal] = []
        for match in pattern.finditer(text):
            try:
                amounts.append(self._normalizer.normalize(match.group(1)))
            except ValueError:
                continue
        return amounts


def build_default_extractor() -> InvoiceAmountExtractor:
    """Wire the default V1 extractor (manual dependency injection).

    Uses the local Tesseract OCR client. This is the single place that
    knows which concrete OCR implementation V1 ships; swapping providers
    later means changing this wiring only.
    """
    return InvoiceAmountExtractor(
        ocr_client=TesseractOcrClient(),
        normalizer=AmountNormalizer(),
    )
