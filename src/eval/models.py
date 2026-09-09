"""Per-query audit log: one row per assistant message, recording scope,
authority, budget, verification, and containment — plus the raw metadata
dict verbatim, so any past answer can be fully reconstructed after the
fact even if a field isn't yet surfaced into the structured columns below.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text

from src.core.db import Base


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True)
    message_id = Column(String, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True)
    chat_id = Column(String, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    # Denormalized (not just reachable via chat_id) so per-user/admin
    # filtering and the summary aggregation don't need a join through chats.
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Each holds a JSON object; see src/eval/service.py for the exact shape
    # written into each one. NOT NULL except containment, which is only
    # populated when something actually deviated from the success path.
    scope_json = Column(Text, nullable=False)
    authority_json = Column(Text, nullable=False)
    budget_json = Column(Text, nullable=False)
    verification_json = Column(Text, nullable=False)
    containment_json = Column(Text, nullable=True)

    raw_metadata_json = Column(Text, nullable=True)

    user_feedback = Column(String, nullable=True)  # 'up' | 'down' | NULL
    user_feedback_comment = Column(Text, nullable=True)
    feedback_at = Column(DateTime, nullable=True)
