from decimal import Decimal

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict


class VerificationRequest(BaseModel):
    invoice_file: UploadFile
    expected_amount: Decimal

    model_config = ConfigDict(arbitrary_types_allowed=True)
