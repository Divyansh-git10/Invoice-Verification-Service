"""Persistence for verification results. The only place that issues SQL — the
API route and domain service stay free of database logic."""
from decimal import Decimal
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Verification
from app.models.validation_result import ValidationResult


class VerificationRepository:
    """Writes verification metadata (never the uploaded file) to Postgres."""

    def save(
        self,
        db: Session,
        result: ValidationResult,
        *,
        mime_type: Optional[str] = None,
    ) -> Verification:
        """Persist one verification and return the stored row. Commits the
        transaction; on failure the caller decides the HTTP outcome
        (fail-closed). Raises on any database error."""
        row = Verification(
            expected_amount=result.expected_amount,
            actual_amount=result.actual_amount,
            matched=result.matched,
            status="matched" if result.matched else "mismatch",
            llm_used=bool(result.llm_used),
            ocr_method=result.ocr_method,
            confident=result.confident,
            llm_confidence=(
                Decimal(str(result.llm_confidence))
                if result.llm_confidence is not None
                else None
            ),
            mime_type=mime_type,
        )
        try:
            db.add(row)
            db.commit()
            db.refresh(row)
            return row
        except Exception:
            db.rollback()
            raise
