"""SQLAlchemy async database engine and session management."""
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


_DB_PATH = Path.home() / ".env-doctor" / "dashboard.db"
_DB_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

engine = create_async_engine(
    _DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables and enable WAL mode."""
    _DB_PATH.parent.mkdir(exist_ok=True)

    from . import models  # noqa: F401 — ensure models are registered

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            __import__("sqlalchemy").text("PRAGMA journal_mode=WAL")
        )


async def get_session():
    """Yield an async session (for FastAPI Depends)."""
    async with async_session() as session:
        yield session
