"""Per-user provider API key management. Keys are encrypted at rest
(src/core/crypto.py) and the raw value is never returned once stored."""

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.api.models.schemas import ApiKeyCreate, ApiKeyOut
from src.auth.models import User, UserApiKey
from src.core.crypto import decrypt_api_key, encrypt_api_key
from src.generation.provider_factory import SUPPORTED_PROVIDERS

router = APIRouter(prefix="/api/settings", tags=["settings"])

CLOUD_PROVIDERS = tuple(p for p in SUPPORTED_PROVIDERS if p != "ollama")


def _mask(raw_key: str) -> str:
    if len(raw_key) <= 8:
        return "*" * len(raw_key)
    return f"{raw_key[:4]}...{raw_key[-4:]}"


@router.get("/api-keys", response_model=List[ApiKeyOut])
def list_api_keys(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Providers this user has configured — masked suffix + created_at only,
    never the raw key. Decrypting to build the mask is a local operation
    (Fernet, no network call), not a privacy concern for a user's own key."""
    rows = db.query(UserApiKey).filter(UserApiKey.user_id == user.id).all()
    return [
        ApiKeyOut(provider=r.provider, masked_key=_mask(decrypt_api_key(r.encrypted_key)), created_at=r.created_at)
        for r in rows
    ]


@router.put("/api-keys", status_code=status.HTTP_204_NO_CONTENT)
def set_api_key(payload: ApiKeyCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if payload.provider not in CLOUD_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {payload.provider!r} (expected one of {CLOUD_PROVIDERS})",
        )
    existing = db.query(UserApiKey).filter_by(user_id=user.id, provider=payload.provider).first()
    encrypted = encrypt_api_key(payload.api_key)
    if existing:
        existing.encrypted_key = encrypted
    else:
        db.add(UserApiKey(id=uuid.uuid4().hex, user_id=user.id, provider=payload.provider, encrypted_key=encrypted))
    db.commit()


@router.delete("/api-keys/{provider}", status_code=status.HTTP_204_NO_CONTENT)
def delete_api_key(provider: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    row = db.query(UserApiKey).filter_by(user_id=user.id, provider=provider).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No key on file for this provider")
    db.delete(row)
    db.commit()
