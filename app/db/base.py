"""Declarative base for ORM models. Kept separate so models and the session/
engine can import it without a circular dependency."""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
