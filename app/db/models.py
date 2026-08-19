"""ORM model for persisted verification results (metadata only — never the
uploaded file/image). Portable column types (Uuid, Numeric, DateTime(tz)) so the
same model works on Postgres in production and SQLite in tests."""
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, DateTime, Numeric, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    expected_amount: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    actual_amount: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    matched: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # "matched"|"mismatch"
    llm_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ocr_method: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    confident: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    llm_confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4), nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
