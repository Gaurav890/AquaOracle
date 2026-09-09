"""Chat CRUD, message history, and streamed message sending."""

import json
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.api.models.schemas import (
    ChatCreate,
    ChatOut,
    ChatProviderUpdate,
    ChatUpdate,
    FeedbackRequest,
    MessageOut,
    SendMessageRequest,
)
from src.api.services.chat_service import (
    build_response_generator,
    get_decrypted_api_key,
    retrieval_config,
    vector_store_lock,
)
from src.api.services.streaming import sse_event, stream_chat_response
from src.auth.models import User
from src.chat.models import Chat, Message
from src.chat.scoping import get_allowed_doc_ids
from src.core.config import settings
from src.eval.models import QueryLog
from src.eval.service import save_query_log
from src.generation.provider_factory import SUPPORTED_PROVIDERS, MissingApiKeyError
from src.indexing.metadata_store import MetadataStore

router = APIRouter(prefix="/api/chats", tags=["chats"])
messages_router = APIRouter(prefix="/api/messages", tags=["messages"])


def _get_owned_chat(db: Session, chat_id: str, user: User) -> Chat:
    chat = db.get(Chat, chat_id)
    if chat is None or chat.user_id != user.id:
        # 404, not 403 — don't reveal whether a chat_id exists for another user.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


@router.get("", response_model=List[ChatOut])
def list_chats(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Chat).filter(Chat.user_id == user.id).order_by(Chat.updated_at.desc()).all()


@router.post("", response_model=ChatOut, status_code=status.HTTP_201_CREATED)
def create_chat(payload: ChatCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    chat = Chat(id=uuid.uuid4().hex, user_id=user.id, title=payload.title or "New chat")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("/{chat_id}", response_model=ChatOut)
def get_chat(chat_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return _get_owned_chat(db, chat_id, user)


@router.patch("/{chat_id}", response_model=ChatOut)
def rename_chat(
    chat_id: str, payload: ChatUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    chat = _get_owned_chat(db, chat_id, user)
    chat.title = payload.title
    db.commit()
    db.refresh(chat)
    return chat


@router.patch("/{chat_id}/provider", response_model=ChatOut)
def set_chat_provider(
    chat_id: str, payload: ChatProviderUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    if payload.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {payload.provider!r} (expected one of {SUPPORTED_PROVIDERS})",
        )
    chat = _get_owned_chat(db, chat_id, user)
    chat.provider = payload.provider
    chat.model = payload.model
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(chat_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    chat = _get_owned_chat(db, chat_id, user)
    db.delete(chat)  # Message rows cascade via ondelete="CASCADE" (foreign_keys=ON in db.py)
    db.commit()


@router.get("/{chat_id}/messages", response_model=List[MessageOut])
def list_messages(chat_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_owned_chat(db, chat_id, user)
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()
    feedback_by_message_id = {
        row.message_id: row.user_feedback
        for row in db.query(QueryLog.message_id, QueryLog.user_feedback).filter(QueryLog.chat_id == chat_id)
    }
    return [_message_to_out(m, feedback_by_message_id.get(m.id)) for m in messages]


def _message_to_out(m: Message, user_feedback: Optional[str] = None) -> MessageOut:
    return MessageOut(
        id=m.id,
        chat_id=m.chat_id,
        role=m.role,
        content=m.content,
        sources=json.loads(m.sources_json) if m.sources_json else None,
        provider=m.provider,
        model=m.model,
        created_at=m.created_at,
        user_feedback=user_feedback,
    )


@router.post("/{chat_id}/messages")
def send_message(
    chat_id: str,
    payload: SendMessageRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    chat = _get_owned_chat(db, chat_id, user)

    # Persist the user's message before generation starts, so a reload
    # always shows it even if generation fails partway through.
    user_message = Message(id=uuid.uuid4().hex, chat_id=chat.id, role="user", content=payload.message)
    db.add(user_message)
    if chat.title == "New chat":
        chat.title = payload.message[:60]
    chat.updated_at = datetime.utcnow()
    db.commit()

    def persist_assistant_message(response) -> str:
        assistant_message = Message(
            id=uuid.uuid4().hex,
            chat_id=chat.id,
            role="assistant",
            content=response.answer,
            sources_json=json.dumps(response.sources),
            provider=chat.provider,
            model=response.metadata.get("model"),
        )
        db.add(assistant_message)
        chat.updated_at = datetime.utcnow()
        save_query_log(
            db,
            message_id=assistant_message.id,
            chat_id=chat.id,
            user_id=user.id,
            question=payload.message,
            provider=chat.provider,
            model=chat.model,
            allowed_doc_ids=allowed_doc_ids,
            response=response,
        )
        db.commit()
        return assistant_message.id

    def persist_error(message: str, containment_event: str = "generation_error") -> str:
        assistant_message = Message(
            id=uuid.uuid4().hex,
            chat_id=chat.id,
            role="assistant",
            content=f"Error: {message}",
            provider=chat.provider,
        )
        db.add(assistant_message)
        chat.updated_at = datetime.utcnow()
        save_query_log(
            db,
            message_id=assistant_message.id,
            chat_id=chat.id,
            user_id=user.id,
            question=payload.message,
            provider=chat.provider,
            model=chat.model,
            allowed_doc_ids=allowed_doc_ids,
            containment_event=containment_event,
            containment_detail=message,
        )
        db.commit()
        return assistant_message.id

    def persist_error_as_assistant_note(message: str) -> str:
        assistant_message = Message(id=uuid.uuid4().hex, chat_id=chat.id, role="assistant", content=message)
        db.add(assistant_message)
        chat.updated_at = datetime.utcnow()
        save_query_log(
            db,
            message_id=assistant_message.id,
            chat_id=chat.id,
            user_id=user.id,
            question=payload.message,
            provider=chat.provider,
            model=chat.model,
            allowed_doc_ids=allowed_doc_ids,
            containment_event="no_documents",
        )
        db.commit()
        return assistant_message.id

    metadata_store = MetadataStore(settings.full_metadata_db_path)
    allowed_doc_ids = get_allowed_doc_ids(db, metadata_store, chat_id=chat.id, user_id=user.id)

    async def generate():
        if not allowed_doc_ids:
            # Short-circuit rather than querying Qdrant with an empty
            # MatchAny (undocumented edge-case behavior) or, worse, silently
            # falling back to searching every document in the index.
            message = (
                "This chat doesn't have any documents yet. Upload a PDF or add one from your "
                "shared knowledge base to start asking questions."
            )
            persist_error_as_assistant_note(message)
            yield sse_event("done", {"answer": message, "sources": []})
            return

        api_key = None
        if chat.provider != "ollama":
            api_key = get_decrypted_api_key(db, user.id, chat.provider)

        with vector_store_lock:
            try:
                response_gen, vector_store = build_response_generator(
                    provider=chat.provider, model=chat.model, api_key=api_key
                )
            except MissingApiKeyError as e:
                persist_error(str(e), containment_event="missing_api_key")
                yield sse_event("error", {"message": str(e)})
                return
            try:
                async for chunk in stream_chat_response(
                    response_gen,
                    question=payload.message,
                    top_k=retrieval_config.vector_top_k,
                    top_n=retrieval_config.rerank_top_n,
                    doc_filter={"doc_id": allowed_doc_ids},
                    on_done=persist_assistant_message,
                    on_error=persist_error,
                ):
                    yield chunk
            finally:
                vector_store.close()

    return StreamingResponse(generate(), media_type="text/event-stream")


@messages_router.post("/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
def submit_feedback(
    message_id: str, payload: FeedbackRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    message = db.get(Message, message_id)
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    # Reuses the same ownership check as everywhere else — 404, not 403, so
    # this doesn't reveal whether a message_id exists for another user.
    _get_owned_chat(db, message.chat_id, user)

    log = db.query(QueryLog).filter(QueryLog.message_id == message_id).first()
    if log is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No query log for this message")

    log.user_feedback = payload.rating
    log.user_feedback_comment = payload.comment
    log.feedback_at = datetime.utcnow()
    db.commit()
