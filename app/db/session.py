"""Database engine + session wiring.

Persistence is OPTIONAL: when settings.DATABASE_URL is unset the engine is None
and `get_db()` yields None, so the API runs (and existing tests pass) without a
database. When set, a single process-wide engine + sessionmaker is created and
`get_db()` yields a request-scoped session.
"""
from typing import Iterator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _normalize_url(url: str) -> str:
    # Railway/Heroku sometimes provide the legacy scheme, which SQLAlchemy
    # rejects; map it to the driver-qualified form. Default driver is psycopg2.
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


def _init() -> None:
    """Create the engine + sessionmaker once, if DATABASE_URL is configured."""
    global _engine, _SessionLocal
    if _SessionLocal is not None or not settings.DATABASE_URL:
        return
    _engine = create_engine(
        _normalize_url(settings.DATABASE_URL),
        pool_pre_ping=True,   # survive Railway connection recycling
        future=True,
    )
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    logger.info("Database persistence enabled")


def get_engine() -> Optional[Engine]:
    _init()
    return _engine


def persistence_enabled() -> bool:
    return bool(settings.DATABASE_URL)


def create_all() -> None:
    """Create tables for v1 (no Alembic yet). No-op when persistence is off."""
    _init()
    if _engine is None:
        return
    # Import models so they are registered on Base.metadata before create_all.
    from app.db import models  # noqa: F401
    from app.db.base import Base

    Base.metadata.create_all(_engine)
    logger.info("Database tables ensured (create_all)")


def get_db() -> Iterator[Optional[Session]]:
    """FastAPI dependency. Yields a session when persistence is enabled, else
    None (callers skip persistence). Always closes the session."""
    _init()
    if _SessionLocal is None:
        yield None
        return
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
