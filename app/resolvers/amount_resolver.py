from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from app.extractors.invoice_amount_extractor import LabelledCandidate
from app.models.resolved_amount import ResolvedAmount


@dataclass(frozen=True)
class ResolverContext:
    """Everything an arbitrator needs to audit the deterministic winner
    without re-running OCR."""

    ocr_text: str
    candidates: list[Decimal]
    deterministic_winner: Optional[Decimal]
    winner_label: Optional[str]
    winner_priority: Optional[int]
    labelled_candidates: list[LabelledCandidate]
    amount_in_words: Optional[Decimal]


class AmountResolver(ABC):
    """Seam for arbitrating the invoice total when deterministic extraction is
    not confident. Implementations must not guess: return a ResolvedAmount with
    amount=None to keep the deterministic winner, or a grounded amount only when
    OCR evidence clearly supports overriding it."""

    @abstractmethod
    def resolve(self, context: ResolverContext) -> ResolvedAmount:
        raise NotImplementedError
