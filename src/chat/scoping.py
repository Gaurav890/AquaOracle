"""Resolves which documents a given chat is allowed to retrieve from."""

from typing import List

from sqlalchemy.orm import Session

from src.chat.models import ChatDocument
from src.indexing.metadata_store import MetadataStore


def get_allowed_doc_ids(db: Session, metadata_store: MetadataStore, chat_id: str, user_id: str) -> List[str]:
    """
    Doc IDs this chat may retrieve from: its own uploads (via the
    chat_documents junction table) plus the user's shared/legacy docs
    (MetadataStore.list_shared_doc_ids — includes CLI-ingested documents,
    which have no owner and are visible to everyone).
    """
    chat_doc_ids = {row.doc_id for row in db.query(ChatDocument.doc_id).filter(ChatDocument.chat_id == chat_id)}
    shared_doc_ids = set(metadata_store.list_shared_doc_ids(owner_user_id=user_id))
    return list(chat_doc_ids | shared_doc_ids)
