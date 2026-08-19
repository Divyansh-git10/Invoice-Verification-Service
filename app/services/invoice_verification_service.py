from decimal import Decimal
from typing import Optional

from app.core.exceptions import AmountNotFoundException
from app.core.logger import get_logger
from app.extractors.invoice_amount_extractor import (
    DeterministicOutcome,
    InvoiceAmountExtractor,
)
from app.models.resolved_amount import ResolvedAmount
from app.models.validation_result import ValidationResult
from app.resolvers.amount_resolver import AmountResolver, ResolverContext
from app.validators.amount_validator import AmountValidator

logger = get_logger(__name__)

# The V1 pipeline uses local Tesseract OCR. Recorded for audit/persistence only.
_OCR_METHOD = "tesseract"


class InvoiceVerificationService:
    """Orchestrates extraction then validation. An optional AmountResolver is
    consulted only when deterministic extraction is not confident; a confident
    deterministic result always bypasses it. The resolver arbitrates: it either
    keeps the deterministic winner (amount=None) or returns an evidenced
    override. With no resolver configured, the behaviour is exactly the
    deterministic-only pipeline."""

    def __init__(
        self,
        extractor: InvoiceAmountExtractor,
        validator: AmountValidator,
        resolver: Optional[AmountResolver] = None,
    ):
        self._extractor = extractor
        self._validator = validator
        self._resolver = resolver

    def verify(
        self,
        file_bytes: bytes,
        mime_type: str,
        expected_amount: Decimal,
    ) -> ValidationResult:
        if self._resolver is None:
            extracted = self._extractor.extract(file_bytes, mime_type)
            result = self._validator.validate(expected_amount, extracted.amount)
            # Deterministic-only path: confidence flag isn't surfaced by extract().
            result.confident = None
            result.llm_used = False
            result.ocr_method = _OCR_METHOD
            result.llm_confidence = None
            return result

        outcome = self._extractor.resolve(file_bytes, mime_type)
        actual, llm_used, llm_confidence = self._resolve_amount(outcome)
        if actual is None:
            raise AmountNotFoundException(
                "Unable to identify the invoice total in the document"
            )
        result = self._validator.validate(expected_amount, actual)
        result.confident = outcome.confident
        result.llm_used = llm_used
        result.ocr_method = _OCR_METHOD
        result.llm_confidence = llm_confidence
        return result

    def _resolve_amount(
        self, outcome: DeterministicOutcome
    ) -> tuple[Optional[Decimal], bool, Optional[float]]:
        """Returns (amount, llm_used, llm_confidence). Decision logic is
        unchanged; the extra fields are audit metadata only. `llm_used` is True
        whenever the resolver is consulted (i.e. the deterministic result was not
        confident); `llm_confidence` is set only when an override was accepted."""
        if outcome.confident and outcome.amount is not None:
            return outcome.amount, False, None

        resolved = self._safe_resolve(outcome)  # LLM fallback consulted here
        if resolved is not None and resolved.amount is not None:
            return resolved.amount, True, resolved.confidence

        # Resolver kept / failed: retain the deterministic winner, which may be
        # a low-confidence figure or None (-> AmountNotFound).
        return outcome.amount, True, None

    def _safe_resolve(self, outcome: DeterministicOutcome) -> Optional[ResolvedAmount]:
        try:
            return self._resolver.resolve(self._build_context(outcome))
        except Exception:  # noqa: BLE001 - resolver failures degrade to deterministic
            logger.exception("Amount resolver failed; using deterministic result")
            return None

    @staticmethod
    def _build_context(outcome: DeterministicOutcome) -> ResolverContext:
        return ResolverContext(
            ocr_text=outcome.ocr_text,
            candidates=outcome.candidates,
            deterministic_winner=outcome.amount,
            winner_label=outcome.winner_label,
            winner_priority=outcome.winner_priority,
            labelled_candidates=outcome.labelled_candidates,
            amount_in_words=outcome.amount_in_words,
        )
