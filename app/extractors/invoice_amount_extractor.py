import re
from dataclasses import dataclass
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


# Final-total labels in priority order. Explicit final-amount labels
# ("invoice amount", "invoice total") outrank calculated roll-ups like
# "total amount(s)", so a printed final amount beats a computed total.
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

# Lower-level labels that must not win the total via the generic word
# "total" (e.g. "Sub-Total", "Total Tax", "Total Sale").
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

# Column-name keywords that mark a line-item table's header row. Used only to
# flag whether a candidate sits in such a table's totals row (structural
# metadata; no arithmetic, no digit correction).
_TABLE_HEADER_KEYWORDS = (
    "hsn",
    "sac",
    "qty",
    "quantity",
    "rate",
    "taxable",
    "description",
    "unit",
    "amount",
)

# Money token for lines that already matched a total keyword. The comma-grouped
# alternative requires >=1 comma so an un-grouped integer falls through to the
# plain alternative and is matched in full (not truncated to its first 3 digits).
_LOOSE_AMOUNT = re.compile(
    r"(?:₹|rs\.?|inr)?\s*"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Strict money token for the keyword-less fallback: requires a grouping comma
# or two decimals so bare IDs (phone/invoice numbers) aren't read as amounts.
_STRICT_AMOUNT = re.compile(
    r"(?:₹|rs\.?|inr)?\s*"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{2})",
    re.IGNORECASE,
)

# Deterministic English number-word parsing for totals printed in words.
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
# Generic amount-in-words terminated by "only"/"paisa only", with no currency
# or "in words" prefix (e.g. "Nine Hundred Sixty-eight And Zero Paisa Only").
# Used only as corroboration and always cross-grounded before it is trusted.
_WORDS_ONLY_PHRASE = re.compile(r"([A-Za-z][A-Za-z \-]*?)\bonly\b", re.IGNORECASE)
# Split camelCase / PascalCase runs like "ThirtyEight" -> "Thirty Eight".
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")


def words_to_amount(phrase: str) -> Optional[Decimal]:
    """Parse an English amount-in-words phrase to a Decimal, or None if it has
    no recognizable number words. Reads up to "only" if present.
    e.g. "FOUR THOUSAND FOUR HUNDRED AND NINETY" -> Decimal("4490.00")."""
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


@dataclass(frozen=True)
class LabelledCandidate:
    """A total-keyword line: its label (matched keyword), the amount on that
    line, the source line, the keyword priority (lower = more authoritative,
    per _TOTAL_KEYWORDS order), the line's position in the OCR, and a small
    window of surrounding OCR lines for section context."""

    label: str
    amount: Decimal
    line: str
    priority: int
    line_index: int = -1
    context: str = ""
    in_table_total: bool = False


@dataclass(frozen=True)
class DeterministicOutcome:
    """Result of deterministic extraction. `confident` is True only for
    keyword-labelled or amount-in-words totals; the keyword-less fallback and
    the no-result case are not confident. `ocr_text`, `candidates`, and the
    labelled metadata are carried so a downstream resolver need not re-run OCR."""

    amount: Optional[Decimal]
    confident: bool
    ocr_text: str
    candidates: list[Decimal]
    labelled_candidates: list[LabelledCandidate]
    winner_label: Optional[str]
    winner_priority: Optional[int]
    amount_in_words: Optional[Decimal]


class InvoiceAmountExtractor:
    """Extraction pipeline: validate input, OCR (via the injected OcrClient),
    identify the invoice total, and normalize it to a Decimal."""

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

    def resolve(self, file_bytes: bytes, mime_type: str) -> DeterministicOutcome:
        """Validate, OCR (once), and deterministically identify the total,
        reporting whether the result is confident along with the OCR text and
        monetary candidates. Does not raise when no total is found.

        Raises:
            UnsupportedFileTypeException / FileTooLargeException /
            ExtractionException / OcrExecutionException on input/OCR failure.
        """
        self._validate_mime(mime_type)
        self._validate_size(file_bytes)

        text = self._run_ocr(file_bytes, mime_type)
        # words_winner drives the deterministic winner (unchanged behaviour);
        # words_corroborated is a cross-grounded value used only for routing and
        # arbitration - it never changes the winner.
        words_winner = self._amount_from_words(text)
        words_corroborated = self._corroborated_words(text)
        amount, confident, labelled, winner_label, winner_priority = (
            self._identify_total(text, words_winner, words_corroborated)
        )
        candidates = self._amounts_in(text, _STRICT_AMOUNT)

        if amount is not None:
            logger.info("Deterministic total: %s (confident=%s)", amount, confident)
        return DeterministicOutcome(
            amount=amount,
            confident=confident,
            ocr_text=text,
            candidates=candidates,
            labelled_candidates=labelled,
            winner_label=winner_label,
            winner_priority=winner_priority,
            amount_in_words=words_corroborated,
        )

    def extract(self, file_bytes: bytes, mime_type: str) -> ExtractedAmount:
        """Extract the invoice total.

        Raises:
            UnsupportedFileTypeException / FileTooLargeException /
            ExtractionException / OcrExecutionException on input/OCR failure;
            AmountNotFoundException when no total is identified.
        """
        outcome = self.resolve(file_bytes, mime_type)
        if outcome.amount is None:
            raise AmountNotFoundException(
                "Unable to identify the invoice total in the document"
            )
        return ExtractedAmount(amount=outcome.amount)

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

    def _identify_total(
        self,
        text: str,
        words_winner: Optional[Decimal],
        words_corroborated: Optional[Decimal],
    ) -> tuple[
        Optional[Decimal], bool, list["LabelledCandidate"], Optional[str], Optional[int]
    ]:
        best_priority: Optional[int] = None
        best_amount: Optional[Decimal] = None
        best_label: Optional[str] = None
        labelled: list[LabelledCandidate] = []

        lines = text.splitlines()
        for idx, line in enumerate(lines):
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
                    if amounts:
                        # On a total line, the total is the largest figure
                        # (e.g. it exceeds any per-item value on that line).
                        line_total = max(amounts)
                        labelled.append(
                            LabelledCandidate(
                                label=keyword,
                                amount=line_total,
                                line=line.strip(),
                                priority=priority,
                                line_index=idx,
                                in_table_total=self._is_table_total_row(lines, idx, line),
                                context=self._context_window(lines, idx),
                            )
                        )
                        if best_priority is None or priority < best_priority:
                            best_priority = priority
                            best_amount = line_total
                            best_label = keyword
                    break  # highest-priority keyword on this line wins

        if best_amount is not None:
            # Confident only when the explicit labelled totals agree. Competing
            # labelled totals with different values are ambiguous and escalate
            # to the resolver. The extracted amount (the winner) is unchanged;
            # only the confidence flag differs. Decimal equality is exact here
            # (no fuzzy/approximate comparison, no arithmetic).
            confident = len({c.amount for c in labelled}) <= 1
            # A cross-grounded amount-in-words value that disagrees with the sole
            # labelled winner makes the outcome ambiguous (routing only; the
            # winner amount is unchanged).
            if (
                confident
                and words_corroborated is not None
                and words_corroborated != best_amount
            ):
                confident = False
                logger.info(
                    "Corroborated amount-in-words %s conflicts with labelled "
                    "winner %s; escalating",
                    words_corroborated,
                    best_amount,
                )
            if not confident:
                logger.info(
                    "Ambiguous total; labelled=%s winner=%s",
                    [str(c.amount) for c in labelled],
                    best_amount,
                )
            return best_amount, confident, labelled, best_label, best_priority

        # Amount-in-words fallback: used only when no labelled numeric total
        # was found, so it never overrides a reliable printed figure.
        if words_winner is not None:
            return words_winner, True, labelled, None, None

        # Keyword-less fallback: the largest clearly-monetary figure found.
        # This is a low-confidence guess (not a labelled/worded total).
        fallback = self._amounts_in(text, _STRICT_AMOUNT)
        if fallback:
            logger.info("No total keyword found; using largest monetary value")
            return max(fallback), False, labelled, None, None

        return None, False, labelled, None, None

    def _amount_from_words(self, text: str) -> Optional[Decimal]:
        """Amount-in-words fallback (None if nothing parses). Detector 1: an
        explicit "... in words" label. Detector 2: a currency-anchored phrase
        terminated by "only" (e.g. "Rs. One Lakh ... Only")."""
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

    def _words_only_phrase(self, text: str) -> Optional[Decimal]:
        """Detector 3: a number-words phrase terminated by 'only'/'paisa only'
        with no currency or 'in words' prefix. Value is not trusted until it is
        cross-grounded (see `_corroborated_words`)."""
        for line in text.splitlines():
            match = _WORDS_ONLY_PHRASE.search(line)
            if not match:
                continue
            value = words_to_amount(match.group(1))
            if value is not None:
                return value
        return None

    def _corroborated_words(self, text: str) -> Optional[Decimal]:
        """A cross-grounded amount-in-words value: parsed from any words detector
        AND also present as a numeric monetary token in the OCR. Used only for
        routing/arbitration; it never changes the deterministic winner. Returns
        None when the parsed value does not appear numerically (mandatory
        cross-grounding, guards against a hallucinated word parse)."""
        value = self._amount_from_words(text)
        if value is None:
            value = self._words_only_phrase(text)
        if value is None:
            return None
        if value in set(self._amounts_in(text, _STRICT_AMOUNT)):
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

    def _is_table_total_row(self, lines: list[str], index: int, line: str) -> bool:
        """True when the candidate is the Total-column value of a line-item
        table's totals row: the line is columnar (>=2 monetary tokens) and a
        table header (a line with >=2 column-name keywords) appears above it.
        Structural only - no arithmetic, no digit correction."""
        if len(self._amounts_in(line, _LOOSE_AMOUNT)) < 2:
            return False
        for j in range(index):
            lowered = lines[j].lower()
            if sum(1 for kw in _TABLE_HEADER_KEYWORDS if kw in lowered) >= 2:
                return True
        return False

    @staticmethod
    def _context_window(lines: list[str], index: int, radius: int = 2) -> str:
        """A small window around `index`: the line plus up to `radius` non-empty
        OCR lines on each side, so a candidate carries its local section context
        (e.g. nearby CGST/SGST/HSN headers that mark a tax-summary section)."""
        before: list[str] = []
        j = index - 1
        while j >= 0 and len(before) < radius:
            stripped = lines[j].strip()
            if stripped:
                before.append(stripped)
            j -= 1
        before.reverse()

        after: list[str] = []
        j = index + 1
        while j < len(lines) and len(after) < radius:
            stripped = lines[j].strip()
            if stripped:
                after.append(stripped)
            j += 1

        return " | ".join(before + [lines[index].strip()] + after)


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
