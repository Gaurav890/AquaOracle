"""Builds and saves the QueryLog row for one query — the scope/authority/
budget/verification/containment breakdown, derived from RAGResponse.metadata
(see src/generation/response_generator.py) plus the chat/user context that
only the caller (src/api/routes/chats.py) knows about.
"""

import dataclasses
import json
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from src.eval.models import QueryLog
from src.eval.pricing import estimate_cost_usd
from src.generation.response_generator import RAGResponse


def _metadata_to_jsonable(metadata: dict) -> dict:
    """RAGResponse.metadata carries a UsageInfo dataclass instance under
    "usage" — everything else in it is already plain dict/list/str/int."""
    out = dict(metadata)
    usage = out.get("usage")
    if usage is not None:
        out["usage"] = dataclasses.asdict(usage)
    return out


def save_query_log(
    db: Session,
    *,
    message_id: str,
    chat_id: str,
    user_id: str,
    question: str,
    provider: str,
    model: Optional[str],
    allowed_doc_ids: List[str],
    response: Optional[RAGResponse] = None,
    containment_event: Optional[str] = None,
    containment_detail: Optional[str] = None,
) -> QueryLog:
    """Always produces one row — on the success path (`response` given) and
    on every deviation path (`containment_event` given instead, or as well
    if retrieval succeeded before generation failed)."""
    metadata = response.metadata if response else {}
    usage = metadata.get("usage")
    timing = metadata.get("timing_ms", {})
    retrieval_metadata = metadata.get("retrieval_metadata", {})
    rerank_enabled = any(s.get("stage") == "reranking" for s in retrieval_metadata.get("stages", []))

    scope = {
        "doc_ids_allowed": allowed_doc_ids,
        "doc_ids_retrieved": metadata.get("retrieved_doc_ids", []),
        "chunk_ids_retrieved": metadata.get("retrieved_chunk_ids", []),
        "rerank_enabled": rerank_enabled,
    }

    authority = {
        "provider": provider,
        "external_call": provider != "ollama",
        "persisted": True,
    }

    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else None
    budget = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "vector_search_ms": timing.get("vector_search"),
        "rerank_ms": timing.get("rerank"),
        "generation_ms": timing.get("generation"),
        "total_ms": timing.get("total"),
        "estimated_cost_usd": estimate_cost_usd(provider, model, prompt_tokens, completion_tokens),
    }

    verification = metadata.get("verification") or {
        "cited_indices": [],
        "ungrounded_citation_indices": [],
        "avg_rerank_score_of_cited": None,
        "no_citations_flag": False,
    }

    containment = None
    if containment_event is not None:
        containment = {"event": containment_event, "detail": containment_detail}

    log = QueryLog(
        id=uuid.uuid4().hex,
        message_id=message_id,
        chat_id=chat_id,
        user_id=user_id,
        question=question,
        provider=provider,
        model=model,
        created_at=datetime.utcnow(),
        scope_json=json.dumps(scope),
        authority_json=json.dumps(authority),
        budget_json=json.dumps(budget),
        verification_json=json.dumps(verification),
        containment_json=json.dumps(containment) if containment else None,
        raw_metadata_json=json.dumps(_metadata_to_jsonable(metadata)) if metadata else None,
    )
    db.add(log)
    return log
