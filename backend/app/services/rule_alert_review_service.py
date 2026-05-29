"""
Service for managing rule-based alert reviews.
Handles business logic for storing and retrieving human review decisions.
Merges review status with CSV data without modifying original files.
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from backend.app.repositories import alert_review_repository


def create_or_update_alert_review(
    db: Session,
    source_run: str,
    rule_code: str,
    new_status: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
    transaction_id: Optional[str] = None,
    customer_hash: Optional[str] = None,
    analyst_notes: Optional[str] = None,
    reviewed_by_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create or update a review for an alert.
    
    Args:
        source_run: Run identifier (e.g., "preprocessed_run_26")
        rule_code: Rule code that generated the alert
        new_status: New status (NEW, IN_REVIEW, DISMISSED, FALSE_POSITIVE, CONFIRMED_FRAUD)
        alert_id: ID of detailed alert (optional)
        summary_alert_id: ID of summary alert (optional)
        transaction_id: Transaction ID (optional)
        customer_hash: Customer hash (optional)
        analyst_notes: Notes from analyst (optional)
        reviewed_by_id: ID of user making review (optional)
    
    Returns:
        Dict with review data
    """
    # Map status to analyst_label
    analyst_label_map = {
        "CONFIRMED_FRAUD": "fraud",
        "FALSE_POSITIVE": "false_positive",
        "DISMISSED": "dismissed",
        "IN_REVIEW": None,
        "NEW": None,
    }
    
    analyst_label = analyst_label_map.get(new_status)
    
    review = alert_review_repository.create_review(
        db,
        source_run=source_run,
        rule_code=rule_code,
        new_status=new_status,
        alert_id=alert_id,
        summary_alert_id=summary_alert_id,
        transaction_id=transaction_id,
        customer_hash=customer_hash,
        analyst_label=analyst_label,
        analyst_notes=analyst_notes,
        reviewed_by_id=reviewed_by_id,
    )
    
    return {
        "id": review.id,
        "source_run": review.source_run,
        "alert_id": review.alert_id,
        "summary_alert_id": review.summary_alert_id,
        "rule_code": review.rule_code,
        "new_status": review.new_status,
        "analyst_notes": review.analyst_notes,
        "reviewed_by_id": review.reviewed_by_id,
        "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
    }


def get_alert_review(
    db: Session,
    source_run: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get the latest review for an alert."""
    review = alert_review_repository.get_review(db, source_run, alert_id, summary_alert_id)
    if not review:
        return None
    
    return {
        "id": review.id,
        "source_run": review.source_run,
        "alert_id": review.alert_id,
        "summary_alert_id": review.summary_alert_id,
        "rule_code": review.rule_code,
        "new_status": review.new_status,
        "analyst_notes": review.analyst_notes,
        "reviewed_by_id": review.reviewed_by_id,
        "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
    }


def get_alert_review_history(
    db: Session,
    source_run: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get all reviews for an alert."""
    reviews = alert_review_repository.get_review_history(db, source_run, alert_id, summary_alert_id)
    return [
        {
            "id": review.id,
            "source_run": review.source_run,
            "alert_id": review.alert_id,
            "summary_alert_id": review.summary_alert_id,
            "rule_code": review.rule_code,
            "previous_status": review.previous_status,
            "new_status": review.new_status,
            "analyst_notes": review.analyst_notes,
            "reviewed_by_id": review.reviewed_by_id,
            "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
        }
        for review in reviews
    ]


def get_current_status(
    db: Session,
    source_run: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
) -> Optional[str]:
    """
    Get current status for an alert.
    Returns DB status if review exists, None otherwise (meaning original CSV status should be used).
    """
    return alert_review_repository.get_current_status(db, source_run, alert_id, summary_alert_id)


def merge_status_with_item(
    db: Session,
    source_run: str,
    item: Dict[str, Any],
    is_summary: bool = False,
) -> Dict[str, Any]:
    """
    Merge review status from DB with CSV item data.
    If a review exists, override the status field.
    Otherwise, keep original CSV status or default to "NEW".
    
    Args:
        source_run: Run identifier
        item: Item dict from CSV (may have 'alert_id' or 'summary_alert_id')
        is_summary: True if this is a summary item, False for detailed alert
    
    Returns:
        Item dict with potentially updated 'status' field
    """
    item_copy = dict(item)
    
    alert_id = item_copy.get("alert_id")
    summary_alert_id = item_copy.get("summary_alert_id")
    
    # Get current status from DB
    db_status = get_current_status(db, source_run, alert_id, summary_alert_id)
    
    if db_status:
        item_copy["status"] = db_status
    elif "status" not in item_copy:
        item_copy["status"] = "NEW"
    
    return item_copy


def merge_items_with_reviews(
    db: Session,
    source_run: str,
    items: List[Dict[str, Any]],
    is_summary: bool = False,
) -> List[Dict[str, Any]]:
    """
    Merge review statuses with multiple items from CSV.
    Preserves all original fields while updating status from DB.
    """
    return [merge_status_with_item(db, source_run, item, is_summary) for item in items]


def list_all_reviews(
    db: Session,
    source_run: str,
    status: Optional[str] = None,
    rule_code: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> Dict[str, Any]:
    """List all reviews for a run with pagination."""
    offset = (page - 1) * page_size
    reviews, total = alert_review_repository.list_reviews(
        db,
        source_run,
        status=status,
        rule_code=rule_code,
        limit=page_size,
        offset=offset,
    )
    
    total_pages = (total + page_size - 1) // page_size if total else 0
    
    return {
        "page": page,
        "page_size": page_size,
        "total_items": total,
        "total_pages": total_pages,
        "items": [
            {
                "id": review.id,
                "source_run": review.source_run,
                "alert_id": review.alert_id,
                "summary_alert_id": review.summary_alert_id,
                "rule_code": review.rule_code,
                "new_status": review.new_status,
                "analyst_notes": review.analyst_notes,
                "reviewed_by_id": review.reviewed_by_id,
                "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
            }
            for review in reviews
        ],
    }
