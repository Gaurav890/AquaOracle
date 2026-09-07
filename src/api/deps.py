"""FastAPI dependencies: DB session and current-user resolution."""

from datetime import datetime

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.auth.models import User, UserSession
from src.core.config import settings
from src.core.db import get_db

__all__ = ["get_db", "get_current_user", "get_current_user_optional"]


def get_current_user_optional(request: Request, db: Session = Depends(get_db)):
    """Returns the logged-in User, or None if there isn't a valid session."""
    session_id = request.cookies.get(settings.session_cookie_name)
    if not session_id:
        return None

    session = db.get(UserSession, session_id)
    if session is None or session.expires_at < datetime.utcnow():
        return None

    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        return None

    session.last_seen_at = datetime.utcnow()
    db.commit()
    return user


def get_current_user(user=Depends(get_current_user_optional)) -> User:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user
