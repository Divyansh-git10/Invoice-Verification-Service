"""Repository persistence tests on SQLite in-memory (no Postgres needed)."""
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401 - register model on Base.metadata
from app.db.base import Base
from app.db.models import Verification
from app.models.validation_result import ValidationResult
from app.repositories.verification_repository import VerificationRepository


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def test_save_persists_all_fields(session):
    repo = VerificationRepository()
    result = ValidationResult(
        matched=True, expected_amount=Decimal("27904.40"),
        actual_amount=Decimal("27904.40"), confident=False, llm_used=True,
        ocr_method="tesseract", llm_confidence=0.95,
    )

    row = repo.save(session, result, mime_type="image/jpeg")

    assert row.id is not None
    assert row.created_at is not None
    fetched = session.get(Verification, row.id)
    assert fetched.matched is True
    assert fetched.status == "matched"
    assert Decimal(str(fetched.expected_amount)) == Decimal("27904.40")
    assert Decimal(str(fetched.actual_amount)) == Decimal("27904.40")
    assert fetched.llm_used is True
    assert fetched.ocr_method == "tesseract"
    assert fetched.confident is False
    assert float(fetched.llm_confidence) == pytest.approx(0.95)
    assert fetched.mime_type == "image/jpeg"


def test_save_mismatch_status_and_defaults(session):
    repo = VerificationRepository()
    # No audit fields set -> llm_used defaults False, others null.
    result = ValidationResult(
        matched=False, expected_amount=Decimal("1"), actual_amount=Decimal("2")
    )

    row = repo.save(session, result)

    assert row.status == "mismatch"
    assert row.llm_used is False
    assert row.confident is None
    assert row.llm_confidence is None
    rows = session.execute(select(Verification)).scalars().all()
    assert len(rows) == 1
