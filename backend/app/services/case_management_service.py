"""
Fase D3.1 — Case Management Service.
Operational follow-up only: creating/closing a case never modifies alerts,
scoring results, or any fraud confirmation flag.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.models.models import (
    CaseManagementCase,
    CaseManagementComment,
    CaseManagementHistory,
    VALID_ORIGIN_TYPES,
    VALID_PRIORITIES,
    VALID_CASE_STATUSES,
    _FORBIDDEN_FIELDS,
)

# Fields callers are allowed to update via update_case
_UPDATABLE_FIELDS = {"status", "priority", "assigned_to", "description", "title"}


# ── internal helpers ─────────────────────────────────────────────────────────

def _case_to_dict(case: CaseManagementCase) -> Dict[str, Any]:
    """Serialize a case ORM object to a safe dict — never exposes forbidden fields."""
    return {
        "id": case.id,
        "case_code": case.case_code,
        "source_run": case.source_run,
        "origin_type": case.origin_type,
        "origin_ref_id": case.origin_ref_id,
        "summary_alert_id": case.summary_alert_id,
        "transaction_id": case.transaction_id,
        "scoring_run_id": case.scoring_run_id,
        "customer_hash": case.customer_hash,
        "title": case.title,
        "description": case.description,
        "priority": case.priority,
        "status": case.status,
        "assigned_to": case.assigned_to,
        "created_by": case.created_by,
        "closed_by": case.closed_by,
        "conclusion": case.conclusion,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
        "closed_at": case.closed_at.isoformat() if case.closed_at else None,
    }


def _comment_to_dict(c: CaseManagementComment) -> Dict[str, Any]:
    return {
        "id": c.id,
        "case_id": c.case_id,
        "user_id": c.user_id,
        "comment_text": c.comment_text,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


def _history_to_dict(h: CaseManagementHistory) -> Dict[str, Any]:
    return {
        "id": h.id,
        "case_id": h.case_id,
        "action": h.action,
        "old_value": h.old_value,
        "new_value": h.new_value,
        "changed_by": h.changed_by,
        "changed_at": h.changed_at.isoformat() if h.changed_at else None,
    }


def _add_history(
    db: Session,
    case_id: int,
    action: str,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    changed_by: Optional[str] = None,
) -> None:
    entry = CaseManagementHistory(
        case_id=case_id,
        action=action,
        old_value=old_value,
        new_value=new_value,
        changed_by=changed_by,
    )
    db.add(entry)


def _generate_case_code(case_id: int) -> str:
    now = datetime.now(timezone.utc)
    return f"CASE-{now.strftime('%Y%m')}-{case_id:05d}"


# ── public API ───────────────────────────────────────────────────────────────

def create_case(
    db: Session,
    title: str,
    origin_type: str,
    priority: str = "MEDIUM",
    description: Optional[str] = None,
    source_run: Optional[str] = None,
    origin_ref_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
    transaction_id: Optional[str] = None,
    scoring_run_id: Optional[str] = None,
    customer_hash: Optional[str] = None,
    assigned_to: Optional[str] = None,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    if not title or not title.strip():
        raise ValueError("title is required")
    if origin_type not in VALID_ORIGIN_TYPES:
        raise ValueError(f"Invalid origin_type '{origin_type}'. Must be one of {sorted(VALID_ORIGIN_TYPES)}")
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'. Must be one of {sorted(VALID_PRIORITIES)}")
    has_ref = any([
        description and description.strip(),
        summary_alert_id,
        transaction_id,
        scoring_run_id,
        origin_ref_id,
    ])
    if not has_ref:
        raise ValueError(
            "A case must have at least one reference: description, summary_alert_id, "
            "transaction_id, scoring_run_id, or origin_ref_id"
        )

    case = CaseManagementCase(
        case_code="PENDING",
        source_run=source_run,
        origin_type=origin_type,
        origin_ref_id=origin_ref_id,
        summary_alert_id=summary_alert_id,
        transaction_id=transaction_id,
        scoring_run_id=scoring_run_id,
        customer_hash=customer_hash,
        title=title.strip(),
        description=description,
        priority=priority,
        status="OPEN",
        assigned_to=assigned_to,
        created_by=created_by,
    )
    db.add(case)
    db.flush()  # get the auto-generated id

    case.case_code = _generate_case_code(case.id)
    db.flush()

    _add_history(db, case.id, "CASE_CREATED", new_value=case.case_code, changed_by=created_by)
    db.commit()
    db.refresh(case)
    return _case_to_dict(case)


def list_cases(
    db: Session,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    origin_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> Dict[str, Any]:
    q = db.query(CaseManagementCase)
    if status:
        q = q.filter(CaseManagementCase.status == status)
    if priority:
        q = q.filter(CaseManagementCase.priority == priority)
    if origin_type:
        q = q.filter(CaseManagementCase.origin_type == origin_type)
    total = q.count()
    offset = (page - 1) * page_size
    cases = q.order_by(CaseManagementCase.created_at.desc()).offset(offset).limit(page_size).all()
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [_case_to_dict(c) for c in cases],
    }


def get_case(db: Session, case_id: int) -> Optional[Dict[str, Any]]:
    case = db.query(CaseManagementCase).filter(CaseManagementCase.id == case_id).first()
    if case is None:
        return None
    return _case_to_dict(case)


def update_case(
    db: Session,
    case_id: int,
    changed_by: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    case = db.query(CaseManagementCase).filter(CaseManagementCase.id == case_id).first()
    if case is None:
        raise ValueError(f"Case {case_id} not found")

    for field in kwargs:
        if field in _FORBIDDEN_FIELDS:
            raise ValueError(f"Field '{field}' cannot be set on a case")

    for field, value in kwargs.items():
        if field not in _UPDATABLE_FIELDS:
            continue
        old_val = str(getattr(case, field)) if getattr(case, field) is not None else None
        new_val = str(value) if value is not None else None

        if field == "status":
            if value not in VALID_CASE_STATUSES:
                raise ValueError(f"Invalid status '{value}'. Must be one of {sorted(VALID_CASE_STATUSES)}")
            _add_history(db, case_id, "STATUS_CHANGED", old_value=old_val, new_value=new_val, changed_by=changed_by)
        elif field == "priority":
            if value not in VALID_PRIORITIES:
                raise ValueError(f"Invalid priority '{value}'. Must be one of {sorted(VALID_PRIORITIES)}")
            _add_history(db, case_id, "PRIORITY_CHANGED", old_value=old_val, new_value=new_val, changed_by=changed_by)
        elif field == "assigned_to":
            _add_history(db, case_id, "ASSIGNED", old_value=old_val, new_value=new_val, changed_by=changed_by)

        setattr(case, field, value)

    db.commit()
    db.refresh(case)
    return _case_to_dict(case)


def add_comment(
    db: Session,
    case_id: int,
    comment_text: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not comment_text or not comment_text.strip():
        raise ValueError("comment_text is required")
    case = db.query(CaseManagementCase).filter(CaseManagementCase.id == case_id).first()
    if case is None:
        raise ValueError(f"Case {case_id} not found")

    comment = CaseManagementComment(
        case_id=case_id,
        user_id=user_id,
        comment_text=comment_text.strip(),
    )
    db.add(comment)
    db.flush()

    _add_history(db, case_id, "COMMENT_ADDED", new_value=str(comment.id), changed_by=user_id)
    db.commit()
    db.refresh(comment)
    return _comment_to_dict(comment)


def list_comments(db: Session, case_id: int) -> List[Dict[str, Any]]:
    comments = (
        db.query(CaseManagementComment)
        .filter(CaseManagementComment.case_id == case_id)
        .order_by(CaseManagementComment.created_at.asc())
        .all()
    )
    return [_comment_to_dict(c) for c in comments]


def list_history(db: Session, case_id: int) -> List[Dict[str, Any]]:
    entries = (
        db.query(CaseManagementHistory)
        .filter(CaseManagementHistory.case_id == case_id)
        .order_by(CaseManagementHistory.changed_at.asc())
        .all()
    )
    return [_history_to_dict(h) for h in entries]


def close_case(
    db: Session,
    case_id: int,
    conclusion: str,
    closed_by: Optional[str] = None,
) -> Dict[str, Any]:
    if not conclusion or not conclusion.strip():
        raise ValueError("conclusion is required to close a case")
    case = db.query(CaseManagementCase).filter(CaseManagementCase.id == case_id).first()
    if case is None:
        raise ValueError(f"Case {case_id} not found")
    if case.status == "CLOSED":
        raise ValueError("Case is already closed")

    old_status = case.status
    case.status = "CLOSED"
    case.conclusion = conclusion.strip()
    case.closed_by = closed_by
    case.closed_at = datetime.now(timezone.utc)

    _add_history(db, case_id, "CASE_CLOSED", old_value=old_status, new_value="CLOSED", changed_by=closed_by)
    db.commit()
    db.refresh(case)
    return _case_to_dict(case)


def reopen_case(
    db: Session,
    case_id: int,
    user: Optional[str] = None,
) -> Dict[str, Any]:
    case = db.query(CaseManagementCase).filter(CaseManagementCase.id == case_id).first()
    if case is None:
        raise ValueError(f"Case {case_id} not found")
    if case.status != "CLOSED":
        raise ValueError("Only CLOSED cases can be reopened")

    case.status = "OPEN"
    case.closed_at = None
    case.closed_by = None

    _add_history(db, case_id, "CASE_REOPENED", old_value="CLOSED", new_value="OPEN", changed_by=user)
    db.commit()
    db.refresh(case)
    return _case_to_dict(case)


def get_cases_summary(db: Session) -> Dict[str, Any]:
    from sqlalchemy import func as sqlfunc

    status_counts: Dict[str, int] = {s: 0 for s in VALID_CASE_STATUSES}
    priority_counts: Dict[str, int] = {p: 0 for p in VALID_PRIORITIES}

    for status, count in db.query(CaseManagementCase.status, sqlfunc.count(CaseManagementCase.id)).group_by(CaseManagementCase.status).all():
        status_counts[status] = count

    for priority, count in db.query(CaseManagementCase.priority, sqlfunc.count(CaseManagementCase.id)).group_by(CaseManagementCase.priority).all():
        priority_counts[priority] = count

    total = db.query(CaseManagementCase).count()
    return {
        "total": total,
        "by_status": status_counts,
        "by_priority": priority_counts,
    }
