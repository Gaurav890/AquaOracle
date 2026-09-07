"""Chat conversations and their messages."""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text

from src.core.db import Base


class Chat(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, default="New chat")
    # 'provider'/'model' columns added now even though the UI only exposes
    # switching them in a later phase, to avoid a second migration.
    provider = Column(String, nullable=False, default="ollama")
    model = Column(String, nullable=True)  # None = provider's default
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)  # bumped per message; sidebar sorts by this


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    chat_id = Column(String, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # assistant rows only
    provider = Column(String, nullable=True)  # what actually generated this reply
    model = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatDocument(Base):
    """Per-chat document scoping: a document uploaded inside a chat is
    scoped to that chat via this junction table, regardless of whether it's
    also marked shared (documents.is_shared) on the raw-sqlite3 side."""

    __tablename__ = "chat_documents"

    chat_id = Column(String, ForeignKey("chats.id", ondelete="CASCADE"), primary_key=True)
    doc_id = Column(String, primary_key=True)  # soft reference — documents table lives in MetadataStore's schema
    added_at = Column(DateTime, default=datetime.utcnow)
