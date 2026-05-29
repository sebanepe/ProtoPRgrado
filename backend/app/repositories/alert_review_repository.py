"""
Repository for rule-based alert reviews.
Non-destructive persistence for human review decisions on rule-generated alerts.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.models.models import RuleAlertReview


def create_review(
    db: Session,
    source_run: str,
    rule_code: str,
    new_status: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
    transaction_id: Optional[str] = None,
    customer_hash: Optional[str] = None,
    previous_status: Optional[str] = None,
    analyst_label: Optional[str] = None,
    analyst_notes: Optional[str] = None,
    reviewed_by_id: Optional[int] = None,
) -> RuleAlertReview:
    """Create or update a review record for an alert."""
    # Validate that at least one alert identifier exists
    if not alert_id and not summary_alert_id:
        raise ValueError("At least one of alert_id or summary_alert_id must be provided")
    
    # Validate status
    valid_statuses = {"NEW", "IN_REVIEW", "DISMISSED", "FALSE_POSITIVE", "CONFIRMED_FRAUD"}
    if new_status not in valid_statuses:
        raise ValueError(f"Invalid status: {new_status}. Valid statuses: {valid_statuses}")
    
    # Check if review already exists
    existing = db.query(RuleAlertReview).filter(
        RuleAlertReview.source_run == source_run,
        RuleAlertReview.rule_code == rule_code,
    )
    
    if alert_id:
        existing = existing.filter(RuleAlertReview.alert_id == alert_id)
    if summary_alert_id:
        existing = existing.filter(RuleAlertReview.summary_alert_id == summary_alert_id)
    
    existing = existing.first()
    
    if existing:
        # Update existing review
        existing.new_status = new_status
        existing.analyst_label = analyst_label
        existing.analyst_notes = analyst_notes
        existing.reviewed_by_id = reviewed_by_id
        db.commit()
        db.refresh(existing)
        return existing
    
    # Create new review
    review = RuleAlertReview(
        source_run=source_run,
        alert_id=alert_id,
        summary_alert_id=summary_alert_id,
        rule_code=rule_code,
        transaction_id=transaction_id,
        customer_hash=customer_hash,
        previous_status=previous_status,
        new_status=new_status,
        analyst_label=analyst_label,
        analyst_notes=analyst_notes,
        reviewed_by_id=reviewed_by_id,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def get_review(
    db: Session,
    source_run: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
) -> Optional[RuleAlertReview]:
    """Get the latest review for an alert."""
    query = db.query(RuleAlertReview).filter(RuleAlertReview.source_run == source_run)
    
    if alert_id:
        query = query.filter(RuleAlertReview.alert_id == alert_id)
    elif summary_alert_id:
        query = query.filter(RuleAlertReview.summary_alert_id == summary_alert_id)
    else:
        return None
    
    # Return most recent review
    return query.order_by(RuleAlertReview.reviewed_at.desc()).first()


def get_review_history(
    db: Session,
    source_run: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
) -> List[RuleAlertReview]:
    """Get all review history for an alert."""
    query = db.query(RuleAlertReview).filter(RuleAlertReview.source_run == source_run)
    
    if alert_id:
        query = query.filter(RuleAlertReview.alert_id == alert_id)
    elif summary_alert_id:
        query = query.filter(RuleAlertReview.summary_alert_id == summary_alert_id)
    else:
        return []
    
    return query.order_by(RuleAlertReview.reviewed_at.desc()).all()


def list_reviews(
    db: Session,
    source_run: str,
    status: Optional[str] = None,
    rule_code: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
) -> tuple[List[RuleAlertReview], int]:
    """List reviews for a run, optionally filtered."""
    query = db.query(RuleAlertReview).filter(RuleAlertReview.source_run == source_run)
    
    if status:
        query = query.filter(RuleAlertReview.new_status == status)
    if rule_code:
        query = query.filter(RuleAlertReview.rule_code == rule_code)
    
    total = query.count()
    results = query.order_by(RuleAlertReview.reviewed_at.desc()).offset(offset).limit(limit).all()
    return results, total


def get_current_status(
    db: Session,
    source_run: str,
    alert_id: Optional[str] = None,
    summary_alert_id: Optional[str] = None,
) -> Optional[str]:
    """Get current status for an alert from reviews, or None if no review exists."""
    review = get_review(db, source_run, alert_id, summary_alert_id)
    return review.new_status if review else None


def delete_review(db: Session, review_id: int) -> bool:
    """Delete a review record."""
    review = db.query(RuleAlertReview).filter(RuleAlertReview.id == review_id).first()
    if not review:
        return False
    db.delete(review)
    db.commit()
    return True
