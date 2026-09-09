"""SQLAlchemy engine/session setup for the auth + chat domain.

The existing documents/chunks/tables/figures schema in
`src/indexing/metadata_store.py` stays on raw sqlite3 (stable, 63 tests
depend on it). This is deliberately a separate access layer for the new
relational domain (users, sessions, chats, messages) added on top — both
point at the same SQLite file, which is fine; SQLite doesn't care who opens
the connection. Real FK/cascade support here is worth the second layer for
auth correctness that hand-rolled SQL would risk getting wrong.

Engine creation is lazy (not done at module-import time) so tests can point
`settings.metadata_db_path` at a scratch file before the first real use,
without any risk of a stray import touching the real project database.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

from src.core.config import settings

Base = declarative_base()

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{settings.full_metadata_db_path}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _):
            # WAL mode lets FastAPI's concurrent requests read/write without
            # locking out MetadataStore's own separate sqlite3 connections
            # to the same file.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    """Create any tables that don't exist yet. Additive only — never touches
    the raw-sqlite3 tables managed by MetadataStore."""
    # Import models here (not at module top) so they register on Base.metadata
    # before create_all runs, without forcing import order on callers.
    from src.auth import models as _auth_models  # noqa: F401

    try:
        from src.chat import models as _chat_models  # noqa: F401
    except ImportError:
        pass  # not created until Phase 2

    try:
        from src.eval import models as _eval_models  # noqa: F401
    except ImportError:
        pass

    engine = _get_engine()
    Base.metadata.create_all(engine)

    # create_all() only adds missing TABLES, never columns to ones that
    # already exist — users predates is_admin, so it needs its own small
    # additive migration (same ad-hoc ALTER TABLE pattern already used in
    # src/indexing/metadata_store.py for its raw-sqlite3 schema).
    with engine.connect() as conn:
        columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)")]
        if "is_admin" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
            conn.commit()


def get_db():
    """FastAPI dependency: yields a session, closes it after the request."""
    _get_engine()  # ensures _SessionLocal is initialized
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
