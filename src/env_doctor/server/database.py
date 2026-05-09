"""SQLAlchemy async database engine and session management."""
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
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
    """Create all tables, run lightweight migrations, and enable WAL mode."""
    _DB_PATH.parent.mkdir(exist_ok=True)

    from . import models  # noqa: F401 — ensure models are registered

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await _run_lightweight_migrations(conn)


async def _run_lightweight_migrations(conn):
    """Idempotent ALTER statements for columns added after the initial schema.

    SQLite's ``CREATE TABLE`` is no-op when a table exists, so new columns on
    existing tables need explicit ``ALTER TABLE``. We run each statement and
    swallow ``OperationalError`` (raised when the column/index already exists).
    """
    statements = [
        "ALTER TABLE machines ADD COLUMN group_name VARCHAR(64)",
        "CREATE INDEX IF NOT EXISTS ix_machines_group_name ON machines(group_name)",
    ]
    for stmt in statements:
        try:
            await conn.execute(text(stmt))
        except OperationalError:
            # Column or index already exists — expected on every run after the first.
            pass


async def get_session():
    """Yield an async session (for FastAPI Depends)."""
    async with async_session() as session:
        yield session
