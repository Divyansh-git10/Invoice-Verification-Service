from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

app.include_router(router)


@app.on_event("startup")
def _init_database() -> None:
    # Create tables when DATABASE_URL is configured (v1: no Alembic yet).
    # No-op when persistence is disabled, so local/dev boots without Postgres.
    from app.db.session import create_all

    create_all()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
