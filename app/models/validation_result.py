from decimal import Decimal
from typing import Optional

from pydantic import BaseModel


class ValidationResult(BaseModel):
    matched: bool
    expected_amount: Decimal
    actual_amount: Decimal

    # Optional audit metadata for persistence only. NOT part of the /verify API
    # response (ResponseMapper emits only matched/expected/actual). Defaults keep
    # existing callers/tests unaffected.
    confident: Optional[bool] = None
    llm_used: Optional[bool] = None
    ocr_method: Optional[str] = None
    llm_confidence: Optional[float] = None
