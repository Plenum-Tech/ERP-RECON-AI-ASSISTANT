from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings

settings = get_settings()

# Async engine for Postgres (make sure settings.database_url is async URL,
# e.g. "postgresql+asyncpg://user:pass@host:port/dbname")
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

# Async session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def async_get_db() -> AsyncSession:
    """
    FastAPI dependency that yields an AsyncSession.

    Usage:
        async def endpoint(db: AsyncSession = Depends(async_get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            # optional explicit close; async context will handle it anyway
            await session.close()


# Backwards-compatible alias if some code still uses `get_db`
get_db = async_get_db
