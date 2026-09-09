"""User accounts, sessions, and stored provider API keys."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, LargeBinary, String, UniqueConstraint

from src.core.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # uuid4().hex — matches the existing doc_id string-ID convention
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)  # can view the Eval tab's cross-user aggregate


class UserSession(Base):
    """Named UserSession, not Session, to avoid clashing with SQLAlchemy's own Session."""

    __tablename__ = "sessions"

    id = Column(String, primary_key=True)  # the opaque token stored in the cookie
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow)


class UserApiKey(Base):
    """A user's own OpenAI/Anthropic API key, encrypted at rest (see src/core/crypto.py)."""

    __tablename__ = "user_api_keys"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String, nullable=False)  # 'openai' | 'anthropic'
    encrypted_key = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_user_provider"),)
