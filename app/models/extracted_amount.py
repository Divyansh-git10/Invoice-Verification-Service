from decimal import Decimal

from pydantic import BaseModel


class ExtractedAmount(BaseModel):
    amount: Decimal
