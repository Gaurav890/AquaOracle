"""Eval tab: per-user (or, for admins, cross-user) query logs and a
provider/model comparison summary."""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.auth.models import User
from src.chat.models import Chat
from src.eval.models import QueryLog

router = APIRouter(prefix="/api/eval", tags=["eval"])


class QueryLogOut(BaseModel):
    id: str
    message_id: str
    chat_id: str
    chat_title: Optional[str] = None
    user_id: str
    question: str
    provider: str
    model: Optional[str] = None
    created_at: str
    scope: Dict[str, Any]
    authority: Dict[str, Any]
    budget: Dict[str, Any]
    verification: Dict[str, Any]
    containment: Optional[Dict[str, Any]] = None
    user_feedback: Optional[str] = None
    user_feedback_comment: Optional[str] = None


class ModelSummary(BaseModel):
    provider: str
    model: Optional[str] = None
    count: int
    avg_prompt_tokens: Optional[float] = None
    avg_completion_tokens: Optional[float] = None
    avg_total_ms: Optional[float] = None
    thumbs_up: int
    thumbs_down: int
    no_feedback: int
    groundedness_pass_rate: Optional[float] = None
    containment_events: int
    total_estimated_cost_usd: Optional[float] = None


def _resolve_scope(scope: str, user: User) -> Optional[str]:
    """Returns the user_id to filter by, or None for no filter (scope=all,
    admin only). Raises 400/403 for an invalid/unauthorized scope."""
    if scope == "mine":
        return user.id
    if scope == "all":
        if not user.is_admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required for scope=all")
        return None
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="scope must be 'mine' or 'all'")


def _average(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 1) if present else None


@router.get("/logs", response_model=List[QueryLogOut])
def list_query_logs(
    scope: str = "mine",
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    user_id = _resolve_scope(scope, user)
    query = db.query(QueryLog)
    if user_id is not None:
        query = query.filter(QueryLog.user_id == user_id)
    logs = query.order_by(QueryLog.created_at.desc()).offset(offset).limit(limit).all()

    chat_ids = {log.chat_id for log in logs}
    titles = {c.id: c.title for c in db.query(Chat).filter(Chat.id.in_(chat_ids))} if chat_ids else {}

    return [
        QueryLogOut(
            id=log.id,
            message_id=log.message_id,
            chat_id=log.chat_id,
            chat_title=titles.get(log.chat_id),
            user_id=log.user_id,
            question=log.question,
            provider=log.provider,
            model=log.model,
            created_at=log.created_at.isoformat(),
            scope=json.loads(log.scope_json),
            authority=json.loads(log.authority_json),
            budget=json.loads(log.budget_json),
            verification=json.loads(log.verification_json),
            containment=json.loads(log.containment_json) if log.containment_json else None,
            user_feedback=log.user_feedback,
            user_feedback_comment=log.user_feedback_comment,
        )
        for log in logs
    ]


@router.get("/summary", response_model=List[ModelSummary])
def eval_summary(scope: str = "mine", db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    user_id = _resolve_scope(scope, user)
    query = db.query(QueryLog)
    if user_id is not None:
        query = query.filter(QueryLog.user_id == user_id)
    logs = query.all()

    groups: Dict[tuple, List[QueryLog]] = defaultdict(list)
    for log in logs:
        groups[(log.provider, log.model)].append(log)

    summaries = []
    for (provider, model), group in groups.items():
        budgets = [json.loads(log.budget_json) for log in group]
        verifications = [json.loads(log.verification_json) for log in group]
        costs = [b["estimated_cost_usd"] for b in budgets if b.get("estimated_cost_usd") is not None]

        summaries.append(ModelSummary(
            provider=provider,
            model=model,
            count=len(group),
            avg_prompt_tokens=_average([b.get("prompt_tokens") for b in budgets]),
            avg_completion_tokens=_average([b.get("completion_tokens") for b in budgets]),
            avg_total_ms=_average([b.get("total_ms") for b in budgets]),
            thumbs_up=sum(1 for log in group if log.user_feedback == "up"),
            thumbs_down=sum(1 for log in group if log.user_feedback == "down"),
            no_feedback=sum(1 for log in group if log.user_feedback is None),
            groundedness_pass_rate=_average(
                [0.0 if v.get("ungrounded_citation_indices") else 1.0 for v in verifications]
            ),
            containment_events=sum(1 for log in group if log.containment_json is not None),
            total_estimated_cost_usd=round(sum(costs), 6) if costs else None,
        ))

    summaries.sort(key=lambda s: s.count, reverse=True)
    return summaries
