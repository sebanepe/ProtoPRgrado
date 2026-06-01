"""
Fase D3.1 — Case Management Routes.
Prefix: /api/cases

Creating or closing a case never modifies alerts, scoring, or any fraud flag.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.services.permission_service import require_permission
from backend.app.services import case_management_service as svc

router = APIRouter(prefix="/api/cases", tags=["cases", "case_management", "d3"])


def _422(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


def _404(case_id: int) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Case {case_id} not found")


# ── literal routes first (must precede /{case_id}) ──────────────────────────

@router.get("/summary", response_model=None)
def cases_summary(
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    """Return counts of cases grouped by status and priority."""
    return svc.get_cases_summary(db)


@router.post("/from-scoring-result", response_model=None)
def create_from_scoring_result(
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    """
    Create a case from a scoring result without modifying the original scoring run.
    The scoring result is referenced only as metadata (scoring_run_id, transaction_id).
    """
    try:
        return svc.create_case(
            db,
            title=body.get("title", ""),
            origin_type="SCORING_RESULT",
            priority=body.get("priority", "MEDIUM"),
            description=body.get("description"),
            source_run=body.get("source_run"),
            origin_ref_id=body.get("origin_ref_id"),
            summary_alert_id=body.get("summary_alert_id"),
            transaction_id=body.get("transaction_id"),
            scoring_run_id=body.get("scoring_run_id"),
            customer_hash=body.get("customer_hash"),
            assigned_to=body.get("assigned_to"),
            created_by=body.get("created_by"),
        )
    except ValueError as e:
        raise _422(e)


# ── collection endpoints ─────────────────────────────────────────────────────

@router.post("", response_model=None)
def create_case(
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    """Create a new investigation case."""
    try:
        return svc.create_case(
            db,
            title=body.get("title", ""),
            origin_type=body.get("origin_type", ""),
            priority=body.get("priority", "MEDIUM"),
            description=body.get("description"),
            source_run=body.get("source_run"),
            origin_ref_id=body.get("origin_ref_id"),
            summary_alert_id=body.get("summary_alert_id"),
            transaction_id=body.get("transaction_id"),
            scoring_run_id=body.get("scoring_run_id"),
            customer_hash=body.get("customer_hash"),
            assigned_to=body.get("assigned_to"),
            created_by=body.get("created_by"),
        )
    except ValueError as e:
        raise _422(e)


@router.get("", response_model=None)
def list_cases(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    origin_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    """List cases with optional filters and pagination."""
    return svc.list_cases(db, status=status, priority=priority, origin_type=origin_type, page=page, page_size=page_size)


# ── per-case endpoints ───────────────────────────────────────────────────────

@router.get("/{case_id}", response_model=None)
def get_case(
    case_id: int,
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    result = svc.get_case(db, case_id)
    if result is None:
        raise _404(case_id)
    return result


@router.patch("/{case_id}", response_model=None)
def update_case(
    case_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    """Update status, priority, assigned_to, title, or description of a case."""
    changed_by = body.pop("changed_by", None)
    try:
        return svc.update_case(db, case_id, changed_by=changed_by, **body)
    except ValueError as e:
        raise _422(e)


@router.post("/{case_id}/comments", response_model=None)
def add_comment(
    case_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    try:
        return svc.add_comment(
            db,
            case_id=case_id,
            comment_text=body.get("comment_text", ""),
            user_id=body.get("user_id"),
        )
    except ValueError as e:
        raise _422(e)


@router.get("/{case_id}/comments", response_model=None)
def list_comments(
    case_id: int,
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> list:
    return svc.list_comments(db, case_id)


@router.get("/{case_id}/history", response_model=None)
def list_history(
    case_id: int,
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> list:
    return svc.list_history(db, case_id)


@router.post("/{case_id}/close", response_model=None)
def close_case(
    case_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    try:
        return svc.close_case(
            db,
            case_id=case_id,
            conclusion=body.get("conclusion", ""),
            closed_by=body.get("closed_by"),
        )
    except ValueError as e:
        raise _422(e)


@router.post("/{case_id}/reopen", response_model=None)
def reopen_case(
    case_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("cases")),
) -> Dict[str, Any]:
    try:
        return svc.reopen_case(db, case_id=case_id, user=body.get("user"))
    except ValueError as e:
        raise _422(e)
