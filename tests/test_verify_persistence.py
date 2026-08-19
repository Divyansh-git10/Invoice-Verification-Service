"""/verify persistence integration: row written, API contract preserved, and
fail-closed on persistence errors. Uses SQLite via get_db override; no Postgres."""
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import get_db, get_repository, get_verification_service
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import Verification
from app.main import app
from app.models.extracted_amount import ExtractedAmount
from app.repositories.verification_repository import VerificationRepository
from app.services.invoice_verification_service import InvoiceVerificationService
from app.validators.amount_validator import AmountValidator

PDF = "application/pdf"


class FakeExtractor:
    def __init__(self, amount):
        self._amount = amount

    def extract(self, file_bytes, mime_type):
        return ExtractedAmount(amount=self._amount)


def _post(client, expected, content_type=PDF):
    return client.post(
        "/verify",
        files={"invoice_file": ("invoice.pdf", b"%PDF-data", content_type)},
        data={"expected_amount": str(expected)},
    )


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def client(sqlite_engine):
    SessionLocal = sessionmaker(bind=sqlite_engine)

    def _get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_verification_service] = (
        lambda: InvoiceVerificationService(FakeExtractor(Decimal("18750.00")), AmountValidator())
    )
    app.dependency_overrides[get_db] = _get_db
    c = TestClient(app)
    yield c, sqlite_engine
    app.dependency_overrides.clear()


def test_verify_persists_row_and_preserves_contract(client):
    c, engine = client
    r = _post(c, "18750.00")

    assert r.status_code == 200
    body = r.json()
    # API contract unchanged: exactly the three fields, nothing added.
    assert set(body.keys()) == {"matched", "expected_amount", "actual_amount"}
    assert body["matched"] is True

    with sessionmaker(bind=engine)() as s:
        rows = s.execute(select(Verification)).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.matched is True
    assert row.status == "matched"
    assert row.ocr_method == "tesseract"
    assert row.llm_used is False  # deterministic-only (no resolver wired)
    assert row.mime_type == PDF


def test_persistence_failure_returns_500_fail_closed(client):
    c, _ = client

    class FailingRepo(VerificationRepository):
        def save(self, *a, **k):
            raise RuntimeError("db down")

    app.dependency_overrides[get_repository] = lambda: FailingRepo()
    r = _post(c, "18750.00")

    assert r.status_code == 500
    assert r.json()["detail"]["type"] == "persistence_failed"


def test_persistence_skipped_when_no_database():
    # get_db yields None (DATABASE_URL unset) -> no persistence, still 200.
    def _no_db():
        yield None

    app.dependency_overrides[get_verification_service] = (
        lambda: InvoiceVerificationService(FakeExtractor(Decimal("100.00")), AmountValidator())
    )
    app.dependency_overrides[get_db] = _no_db
    try:
        c = TestClient(app)
        r = _post(c, "100.00")
        assert r.status_code == 200
        assert r.json()["matched"] is True
    finally:
        app.dependency_overrides.clear()
