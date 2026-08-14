from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ResolvedAmount(BaseModel):
    """Structured output of an AmountResolver. `amount` is None when evidence
    is insufficient; when `amount` is set, `evidence` must be non-empty."""

    amount: Optional[Decimal] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = None

    @model_validator(mode="after")
    def _evidence_required_with_amount(self) -> "ResolvedAmount":
        if self.amount is not None and not (self.evidence or "").strip():
            raise ValueError("evidence must be non-empty when amount is provided")
        return self
