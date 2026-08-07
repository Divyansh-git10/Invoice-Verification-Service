from decimal import Decimal

from pydantic import BaseModel


class VerificationResponse(BaseModel):
    matched: bool
    expected_amount: Decimal
    actual_amount: Decimal
