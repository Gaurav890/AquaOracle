"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --- Auth ---------------------------------------------------------------

class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    display_name: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Chats ----------------------------------------------------------------

class ChatCreate(BaseModel):
    title: Optional[str] = None


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatProviderUpdate(BaseModel):
    provider: str
    model: Optional[str] = None


class ChatOut(BaseModel):
    id: str
    title: str
    provider: str
    model: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageOut(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SendMessageRequest(BaseModel):
    message: str = Field(min_length=1)


# --- Provider API keys ------------------------------------------------------

class ApiKeyCreate(BaseModel):
    provider: str
    api_key: str = Field(min_length=1)


class ApiKeyOut(BaseModel):
    provider: str
    masked_key: str  # e.g. "sk-...ab12" — the raw key is never returned
    created_at: datetime
