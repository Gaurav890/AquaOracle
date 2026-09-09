"""Bridges ResponseGenerator.generate(on_token=...) — synchronous, blocking —
to an async Server-Sent Events stream for FastAPI's StreamingResponse.

Uses a background thread + queue.Queue to move tokens from the sync
generation call to the async world, with the blocking queue.get() offloaded
via loop.run_in_executor so it doesn't block the event loop.
"""

import asyncio
import json
import queue
import threading
from typing import AsyncIterator, Callable, Optional

from loguru import logger

from src.generation.response_generator import RAGResponse, ResponseGenerator

log = logger.bind(name="Streaming")


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_chat_response(
    response_gen: ResponseGenerator,
    question: str,
    top_k: int,
    top_n: int,
    doc_filter: Optional[dict] = None,
    on_done: Optional[Callable[[RAGResponse], Optional[str]]] = None,
    on_error: Optional[Callable[[str], None]] = None,
) -> AsyncIterator[str]:
    """Yields SSE-framed strings: 'token' events as text arrives, then one
    final 'done' event with the full answer + sources, or 'error'.

    `on_done`/`on_error` are called synchronously right before the matching
    SSE event is yielded — callers use these to persist the assistant's
    message (or lack thereof) without this module needing to know about
    chats/messages/the DB. `on_done`'s return value (the persisted
    message's id, or None) is included in the 'done' event so the frontend
    can attach feedback controls to the message it just streamed.
    """
    token_queue: "queue.Queue" = queue.Queue()

    def on_token(piece: str) -> None:
        token_queue.put(("token", piece))

    def run() -> None:
        try:
            response = response_gen.generate(
                question=question,
                top_k=top_k,
                top_n=top_n,
                doc_filter=doc_filter,
                on_token=on_token,
            )
            token_queue.put(("done", response))
        except Exception as e:  # surfaced to the client as an 'error' SSE event
            log.exception("Chat generation failed")
            token_queue.put(("error", str(e)))

    threading.Thread(target=run, daemon=True).start()

    loop = asyncio.get_event_loop()
    while True:
        kind, payload = await loop.run_in_executor(None, token_queue.get)
        if kind == "token":
            yield sse_event("token", {"text": payload})
        elif kind == "done":
            message_id = on_done(payload) if on_done else None
            yield sse_event("done", {"answer": payload.answer, "sources": payload.sources, "message_id": message_id})
            return
        else:
            if on_error:
                on_error(payload)
            yield sse_event("error", {"message": payload})
            return
