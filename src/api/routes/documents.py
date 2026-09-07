"""Per-chat document upload/listing, the user's global document list, and
the shared-knowledge-base toggle.
"""

import re
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.api.routes.chats import _get_owned_chat
from src.auth.models import User
from src.chat.models import ChatDocument
from src.core.config import settings
from src.document_processing.ingestion_service import ingest_pdf
from src.indexing.metadata_store import MetadataStore

router = APIRouter(prefix="/api", tags=["documents"])


class DocumentOut(BaseModel):
    doc_id: str
    title: Optional[str] = None
    file_name: Optional[str] = None
    organization: Optional[str] = None
    year: Optional[int] = None
    page_count: int = 0
    chunk_count: int = 0
    is_shared: bool = False


def _doc_to_out(doc: dict) -> DocumentOut:
    return DocumentOut(
        doc_id=doc.get("doc_id", ""),
        title=doc.get("title") or doc.get("file_name"),
        file_name=doc.get("file_name"),
        organization=doc.get("organization"),
        year=doc.get("year"),
        page_count=doc.get("page_count") or 0,
        chunk_count=doc.get("chunk_count") or 0,
        is_shared=bool(doc.get("is_shared")),
    )


_SAFE_FILENAME_RE = re.compile(r"[^\w.\- ]")


@router.post("/chats/{chat_id}/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_chat_document(
    chat_id: str,
    file: UploadFile = File(...),
    share: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    chat = _get_owned_chat(db, chat_id, user)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")

    upload_dir = settings.full_data_path / "uploads" / user.id
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _SAFE_FILENAME_RE.sub("_", file.filename)
    dest_path = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"

    content = await file.read()
    dest_path.write_bytes(content)

    result = ingest_pdf(dest_path, owner_user_id=user.id, is_shared=share)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.error)

    db.add(ChatDocument(chat_id=chat.id, doc_id=result.doc_id))
    db.commit()

    metadata_store = MetadataStore(settings.full_metadata_db_path)
    doc = metadata_store.get_document(result.doc_id)
    return _doc_to_out(doc or {"doc_id": result.doc_id, "file_name": result.file_name})


@router.get("/chats/{chat_id}/documents", response_model=List[DocumentOut])
def list_chat_documents(chat_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_owned_chat(db, chat_id, user)
    doc_ids = [row.doc_id for row in db.query(ChatDocument.doc_id).filter(ChatDocument.chat_id == chat_id)]

    metadata_store = MetadataStore(settings.full_metadata_db_path)
    docs = []
    for doc_id in doc_ids:
        doc = metadata_store.get_document(doc_id)
        if doc:
            docs.append(_doc_to_out(doc))
    return docs


@router.get("/documents", response_model=List[DocumentOut])
def list_my_documents(user: User = Depends(get_current_user)):
    metadata_store = MetadataStore(settings.full_metadata_db_path)
    return [_doc_to_out(d) for d in metadata_store.list_documents(owner_user_id=user.id)]


@router.post("/documents/{doc_id}/share", response_model=DocumentOut)
def set_document_shared(doc_id: str, share: bool, user: User = Depends(get_current_user)):
    metadata_store = MetadataStore(settings.full_metadata_db_path)
    doc = metadata_store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.get("owner_user_id") != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't own this document")

    metadata_store.set_shared(doc_id, share)
    return _doc_to_out(metadata_store.get_document(doc_id))


@router.delete("/chats/{chat_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document_from_chat(
    chat_id: str, doc_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """Un-scopes a document from this chat. Does not delete the document
    itself (it may be shared or used by other chats)."""
    _get_owned_chat(db, chat_id, user)
    db.query(ChatDocument).filter(ChatDocument.chat_id == chat_id, ChatDocument.doc_id == doc_id).delete()
    db.commit()
