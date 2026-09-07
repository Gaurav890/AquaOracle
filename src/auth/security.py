"""Password hashing and session token helpers."""

import secrets
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from sqlalchemy.orm import Session as DbSession

from src.auth.models import User, UserSession
from src.core.config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        # Malformed hash (shouldn't happen in practice) — fail closed, not open.
        return False


def create_session(db: DbSession, user: User) -> UserSession:
    session = UserSession(
        id=secrets.token_urlsafe(32),
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=settings.session_ttl_days),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_valid_session(db: DbSession, session_id: str) -> Optional[UserSession]:
    session = db.get(UserSession, session_id)
    if session is None:
        return None
    if session.expires_at < datetime.utcnow():
        return None
    return session


def delete_session(db: DbSession, session_id: str) -> None:
    session = db.get(UserSession, session_id)
    if session is not None:
        db.delete(session)
        db.commit()
