from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.response import VerificationResponse
from app.api.response_mapper import ResponseMapper
from app.core.exceptions import (
    AmountNotFoundException,
    ExtractionException,
    FileTooLargeException,
    OcrExecutionException,
    UnsupportedFileTypeException,
)
from app.core.logger import get_logger
from app.db.session import get_db
from app.extractors.invoice_amount_extractor import build_default_extractor
from app.repositories.verification_repository import VerificationRepository
from app.resolvers.llm_amount_resolver import build_default_resolver
from app.services.invoice_verification_service import InvoiceVerificationService
from app.validators.amount_validator import AmountValidator

logger = get_logger(__name__)

router = APIRouter(prefix="/verify", tags=["Invoice Verification"])


# Composition root. The resolver is wired only when GROQ_API_KEY is set,
# otherwise it is None and the pipeline stays deterministic-only. Tests
# override the default service via dependency_overrides on
# get_verification_service.
_default_service = InvoiceVerificationService(
    extractor=build_default_extractor(),
    validator=AmountValidator(),
    resolver=build_default_resolver(),
)


def get_verification_service() -> InvoiceVerificationService:
    return _default_service


_repository = VerificationRepository()


def get_repository() -> VerificationRepository:
    return _repository


def _error(error_type: str, message: str) -> dict:
    return {"type": error_type, "message": message}


@router.post("", response_model=VerificationResponse)
async def verify_invoice(
    invoice_file: UploadFile = File(...),
    expected_amount: Decimal = Form(...),
    service: InvoiceVerificationService = Depends(get_verification_service),
    db: Optional[Session] = Depends(get_db),
    repository: VerificationRepository = Depends(get_repository),
) -> VerificationResponse:
    """Verify the uploaded invoice total against expected_amount.

    A business mismatch is still HTTP 200 (matched=false); extraction
    failures map to the error taxonomy below. On success the result metadata is
    persisted (when a database is configured); if persistence fails the request
    fails closed (500 persistence_failed) rather than reporting a stored result
    that was not stored.
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

    # Persist the successful result. `db` is None when DATABASE_URL is unset
    # (persistence disabled for local/tests) -> skip. Fail closed on DB errors.
    if db is not None:
        try:
            repository.save(db, result, mime_type=mime_type)
        except Exception as exc:  # noqa: BLE001 - any DB failure fails the request
            logger.exception(
                "Verification succeeded but persistence failed "
                "(matched=%s expected=%s actual=%s)",
                result.matched, result.expected_amount, result.actual_amount,
            )
            raise HTTPException(
                status_code=500,
                detail=_error(
                    "persistence_failed",
                    "Verification completed but the result could not be persisted.",
                ),
            ) from exc

    return ResponseMapper.to_response(result)
