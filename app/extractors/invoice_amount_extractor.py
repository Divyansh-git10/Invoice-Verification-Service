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


# Final-total labels in priority order (most explicit final total first). The
# extractor prefers the highest-priority label present. Explicit final-amount
# labels ("invoice amount", "invoice total") intentionally outrank calculated
# roll-ups such as "total amount(s)", so a printed final amount beats a
# lower-level computed total (see Case A in Component 6).
_TOTAL_KEYWORDS = (
    "invoice amount",
    "invoice total",
    "grand total",
    "total amount payable",
    "total payable",
    "amount payable",
    "net payable",
    "balance due",
    "amount due",
    "net amount",
    "total amount",
    "total",
)

# Lower-level financial labels. A line carrying one of these must NOT win the
# total just because it also contains the generic word "total" (e.g.
# "Sub-Total", "Total Tax"). Such lines are skipped when the only matching
# keyword is the generic "total".
_SUBORDINATE_MARKERS = (
    "sub total",
    "sub-total",
    "subtotal",
    "taxable",
    "total tax",
    "tax amount",
    "total sale",
    "total sales",
    "cgst",
    "sgst",
    "igst",
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

# --- Amount-in-words support ----------------------------------------------
# Deterministic, local English number-word parsing for invoice totals printed
# in words (e.g. "Total in words: FOUR THOUSAND ... NINETY RUPEES"). No LLM,
# no external calls. Supports units/tens/hundred and Indian + Western scales.
_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_WORD_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_SCALES = {
    "thousand": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000, "crores": 10_000_000,
    "billion": 1_000_000_000,
}

# Trigger phrase: "... in words" / "... in word".
_WORDS_TRIGGER = re.compile(r"in\s+words?\b", re.IGNORECASE)
# Strong conventional total-in-words phrase without an explicit "in words"
# label: a currency token, then number words, terminated by "only"
# (e.g. "Rs. One Lakh Five Thousand Two Hundred Only", "INR ... Only").
_WORDS_CURRENCY_ANCHOR = re.compile(
    r"(?:₹|rs\.?|rupees|inr)\s+(.*?)\bonly\b", re.IGNORECASE
)
# Split camelCase / PascalCase runs like "ThirtyEight" -> "Thirty Eight".
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")


def words_to_amount(phrase: str) -> Optional[Decimal]:
    """Parse an English amount-in-words phrase into a Decimal, or None.

    Deterministic and conservative: returns None (rather than guessing) when
    the phrase contains no recognizable number words. Non-number tokens
    ("rupees", "and", "only", ...) are ignored. Reads up to the word "only"
    if present (a common terminator).

    Examples:
        "FOUR THOUSAND FOUR HUNDRED AND NINETY RUPEES" -> Decimal("4490.00")
        "Rupees ThirtyEight Thousand TwentySix Only"   -> Decimal("38026.00")
    """
    if not phrase:
        return None

    spaced = _CAMEL_SPLIT.sub(" ", phrase)
    tokens = [t.lower() for t in re.split(r"[^A-Za-z]+", spaced) if t]
    if "only" in tokens:
        tokens = tokens[: tokens.index("only")]

    total = 0
    current = 0
    found = False
    for token in tokens:
        if token in _WORD_UNITS:
            current += _WORD_UNITS[token]
            found = True
        elif token in _WORD_TENS:
            current += _WORD_TENS[token]
            found = True
        elif token == "hundred":
            current = (current or 1) * 100
            found = True
        elif token in _WORD_SCALES:
            current = (current or 1) * _WORD_SCALES[token]
            total += current
            current = 0
            found = True
        # Any other token (rupees, and, of, etc.) is ignored.

    total += current
    if not found or total <= 0:
        return None
    return Decimal(total).quantize(Decimal("0.01"))


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
                    # A subordinate line (sub-total, tax, cgst...) must not win
                    # via the generic "total" keyword.
                    if keyword == "total" and any(
                        marker in lowered for marker in _SUBORDINATE_MARKERS
                    ):
                        break

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

        # Amount-in-words fallback: used only when no labelled numeric total
        # was found, so it never overrides a reliable printed figure.
        worded = self._amount_from_words(text)
        if worded is not None:
            return worded

        # Keyword-less fallback: the largest clearly-monetary figure found.
        fallback = self._amounts_in(text, _STRICT_AMOUNT)
        if fallback:
            logger.info("No total keyword found; using largest monetary value")
            return max(fallback)

        return None

    def _amount_from_words(self, text: str) -> Optional[Decimal]:
        """Find an amount written in words. Fallback only; returns None if
        nothing parses.

        Two deterministic detectors, tried in order:
          1. an explicit "... in words" label (words on the same line, or a
             following line for a label-only line);
          2. a strong currency-anchored phrase terminated by "only"
             (e.g. "Rs. One Lakh Five Thousand Two Hundred Only").
        """
        lines = text.splitlines()

        # 1) Explicit "... in words" label.
        for i, line in enumerate(lines):
            match = _WORDS_TRIGGER.search(line)
            if not match:
                continue

            segment = line[match.end():]
            value = words_to_amount(segment)
            if value is not None:
                logger.info("Identified invoice total from amount-in-words: %s", value)
                return value

            # Words may sit on a following line (possibly after blank lines).
            # Append the next few non-empty lines and retry.
            appended = 0
            for nxt in lines[i + 1:]:
                nxt = nxt.strip()
                if not nxt:
                    continue
                segment = f"{segment} {nxt}"
                appended += 1
                value = words_to_amount(segment)
                if value is not None:
                    logger.info(
                        "Identified invoice total from amount-in-words: %s", value
                    )
                    return value
                if appended >= 2:
                    break

        # 2) Currency-anchored "... Only" phrase (no explicit label needed).
        for line in lines:
            anchor = _WORDS_CURRENCY_ANCHOR.search(line)
            if not anchor:
                continue
            value = words_to_amount(anchor.group(1))
            if value is not None:
                logger.info(
                    "Identified invoice total from currency-anchored "
                    "amount-in-words: %s",
                    value,
                )
                return value

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
