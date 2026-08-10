from decimal import Decimal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.response import VerificationResponse
from app.api.response_mapper import ResponseMapper
from app.core.exceptions import (
    AmountNotFoundException,
    ExtractionException,
    FileTooLargeException,
    OcrExecutionException,
    UnsupportedFileTypeException,
)
from app.extractors.invoice_amount_extractor import build_default_extractor
from app.services.invoice_verification_service import InvoiceVerificationService
from app.validators.amount_validator import AmountValidator

router = APIRouter(prefix="/verify", tags=["Invoice Verification"])


# --- Manual wiring (composition root) --------------------------------------
# The default V1 service: local Tesseract extractor + exact-equality validator,
# wired by hand. No DI framework, factory, or registry. Tests can substitute a
# service via FastAPI's dependency_overrides on `get_verification_service`.
_default_service = InvoiceVerificationService(
    extractor=build_default_extractor(),
    validator=AmountValidator(),
)


def get_verification_service() -> InvoiceVerificationService:
    return _default_service


def _error(error_type: str, message: str) -> dict:
    return {"type": error_type, "message": message}


@router.post("", response_model=VerificationResponse)
async def verify_invoice(
    invoice_file: UploadFile = File(...),
    expected_amount: Decimal = Form(...),
    service: InvoiceVerificationService = Depends(get_verification_service),
) -> VerificationResponse:
    """Verify that the uploaded invoice's total matches the expected amount.

    The API layer only: reads the file bytes and MIME type, delegates the
    workflow to the service, and maps the domain result to the transport
    model. A business mismatch is a successful verification (HTTP 200 with
    matched=false); extraction failures map to the error taxonomy.
    """
    file_bytes = await invoice_file.read()
    mime_type = invoice_file.content_type

    try:
        result = service.verify(file_bytes, mime_type, expected_amount)
    except UnsupportedFileTypeException as exc:
        raise HTTPException(status_code=415, detail=_error("unsupported_file_type", str(exc)))
    except FileTooLargeException as exc:
        raise HTTPException(status_code=413, detail=_error("file_too_large", str(exc)))
    except AmountNotFoundException as exc:
        raise HTTPException(status_code=422, detail=_error("amount_not_found", str(exc)))
    except OcrExecutionException as exc:
        raise HTTPException(status_code=500, detail=_error("ocr_execution_failed", str(exc)))
    except ExtractionException as exc:
        # Base extraction failure (e.g. empty/unreadable document).
        raise HTTPException(status_code=422, detail=_error("extraction_failed", str(exc)))

    return ResponseMapper.to_response(result)
