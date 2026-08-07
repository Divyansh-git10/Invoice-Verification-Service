from decimal import Decimal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/verify", tags=["Invoice Verification"])


@router.post("")
async def verify_invoice(
    invoice_file: UploadFile = File(...),
    expected_amount: Decimal = Form(...),
):
    raise HTTPException(
        status_code=501,
        detail="Invoice verification not implemented yet.",
    )
