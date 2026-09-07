"""Resolves which documents a given chat is allowed to retrieve from."""

from typing import List

from sqlalchemy.orm import Session

from src.chat.models import ChatDocument
from src.indexing.metadata_store import MetadataStore


def get_allowed_doc_ids(db: Session, metadata_store: MetadataStore, chat_id: str, user_id: str) -> List[str]:
    """
    Doc IDs this chat may retrieve from.

    If the chat has its own documents (uploaded or added into it via the
    chat_documents junction table), retrieval is scoped to ONLY those — a
    user who uploads a specific doc to ask about it means that literally,
    and blending in the much larger shared/legacy corpus was drowning out
    small, specific uploads in practice (a 4-chunk upload has little chance
    against thousands of unrelated chunks for a vague query).

    Only when the chat has no documents of its own does it fall back to the
    user's shared docs plus legacy/CLI-ingested docs (MetadataStore.
    list_shared_doc_ids — the latter have no owner and are visible to
    everyone) — so a brand-new chat still has something to search rather
    than immediately hitting the "no documents" short-circuit.
    """
    chat_doc_ids = {row.doc_id for row in db.query(ChatDocument.doc_id).filter(ChatDocument.chat_id == chat_id)}
    if chat_doc_ids:
        return list(chat_doc_ids)

    shared_doc_ids = set(metadata_store.list_shared_doc_ids(owner_user_id=user_id))
    return list(shared_doc_ids)
