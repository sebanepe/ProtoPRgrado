"""
Integration tests for PHASE B.3: Rule-based Alert Review Endpoints
Tests human review workflows, status updates, and history tracking.
"""

import pytest
from sqlalchemy.orm import Session

from backend.app.models.models import RuleAlertReview
from backend.app.repositories import alert_review_repository
from backend.app.services import rule_alert_review_service


@pytest.fixture()
def db(db_session):
    """Provide a clean alert-review test session using the conftest test DB."""
    db_session.query(RuleAlertReview).delete()
    db_session.commit()

    try:
        yield db_session
    finally:
        db_session.query(RuleAlertReview).delete()
        db_session.commit()


class TestAlertReviewRepository:
    """Test alert review repository functions."""

    def test_create_review(self, db: Session):
        """Test creating a new alert review."""
        review = alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="IN_REVIEW",
            alert_id="alert_001",
            analyst_notes="Needs investigation",
        )
        assert review is not None
        assert review.source_run == "preprocessed_run_1"
        assert review.new_status == "IN_REVIEW"
        assert review.alert_id == "alert_001"

    def test_create_review_invalid_status(self, db: Session):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            alert_review_repository.create_review(
                db,
                source_run="preprocessed_run_1",
                rule_code="RULE_001",
                new_status="INVALID_STATUS",
                alert_id="alert_001",
            )

    def test_create_review_no_identifiers(self, db: Session):
        """Test that review without alert_id or summary_alert_id raises ValueError."""
        with pytest.raises(ValueError, match="At least one of"):
            alert_review_repository.create_review(
                db,
                source_run="preprocessed_run_1",
                rule_code="RULE_001",
                new_status="IN_REVIEW",
            )

    def test_update_existing_review(self, db: Session):
        """Test updating an existing review."""
        # Create first review
        review1 = alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="IN_REVIEW",
            alert_id="alert_001",
        )
        review1_id = review1.id

        # Update with same identifiers
        review2 = alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="FALSE_POSITIVE",
            alert_id="alert_001",
            analyst_notes="Actually false positive",
        )

        # Should update the existing review, not create a new one
        assert review2.id == review1_id
        assert review2.new_status == "FALSE_POSITIVE"
        assert review2.analyst_notes == "Actually false positive"

    def test_get_review(self, db: Session):
        """Test retrieving a review."""
        alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="IN_REVIEW",
            alert_id="alert_001",
        )

        review = alert_review_repository.get_review(
            db, source_run="preprocessed_run_1", alert_id="alert_001"
        )
        assert review is not None
        assert review.new_status == "IN_REVIEW"

    def test_get_review_not_found(self, db: Session):
        """Test retrieving non-existent review returns None."""
        review = alert_review_repository.get_review(
            db, source_run="preprocessed_run_999", alert_id="alert_999"
        )
        assert review is None

    def test_get_review_history(self, db: Session):
        """Test retrieving review history."""
        # Create multiple reviews (updates) for same alert
        alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="IN_REVIEW",
            alert_id="alert_001",
        )
        alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="FALSE_POSITIVE",
            alert_id="alert_001",
            analyst_notes="False positive found",
        )

        history = alert_review_repository.get_review_history(
            db, source_run="preprocessed_run_1", alert_id="alert_001"
        )
        assert len(history) >= 1
        # Latest should be FALSE_POSITIVE
        assert history[0].new_status == "FALSE_POSITIVE"

    def test_get_current_status(self, db: Session):
        """Test getting current status for alert."""
        alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="IN_REVIEW",
            alert_id="alert_001",
        )

        status = alert_review_repository.get_current_status(
            db, source_run="preprocessed_run_1", alert_id="alert_001"
        )
        assert status == "IN_REVIEW"

    def test_list_reviews_with_filters(self, db: Session):
        """Test listing reviews with optional filters."""
        alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="IN_REVIEW",
            alert_id="alert_001",
        )
        alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_002",
            new_status="DISMISSED",
            alert_id="alert_002",
        )

        # List all
        reviews, total = alert_review_repository.list_reviews(
            db, source_run="preprocessed_run_1"
        )
        assert total >= 2

        # Filter by status
        reviews, total = alert_review_repository.list_reviews(
            db, source_run="preprocessed_run_1", status="IN_REVIEW"
        )
        assert len(reviews) >= 1
        assert reviews[0].new_status == "IN_REVIEW"

        # Filter by rule
        reviews, total = alert_review_repository.list_reviews(
            db, source_run="preprocessed_run_1", rule_code="RULE_001"
        )
        assert len(reviews) >= 1
        assert reviews[0].rule_code == "RULE_001"


class TestAlertReviewService:
    """Test alert review service functions."""

    def test_create_or_update_alert_review(self, db: Session):
        """Test service function for creating/updating review."""
        result = rule_alert_review_service.create_or_update_alert_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="CONFIRMED_FRAUD",
            alert_id="alert_001",
            analyst_notes="Confirmed as fraud",
        )
        assert result is not None
        assert result["new_status"] == "CONFIRMED_FRAUD"
        assert result["analyst_notes"] == "Confirmed as fraud"

    def test_merge_status_with_item(self, db: Session):
        """Test merging review status with CSV item."""
        # Create a review
        alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="FALSE_POSITIVE",
            alert_id="alert_001",
        )

        # Merge with CSV item
        item = {
            "alert_id": "alert_001",
            "status": "NEW",
            "customer_hash": "cust_001",
            "rule_code": "RULE_001",
        }
        merged = rule_alert_review_service.merge_status_with_item(
            db, "preprocessed_run_1", item, is_summary=False
        )
        # Status should be overridden from DB
        assert merged["status"] == "FALSE_POSITIVE"

    def test_merge_status_no_review(self, db: Session):
        """Test merging when no review exists (use CSV status)."""
        item = {"alert_id": "alert_999", "status": "NEW", "rule_code": "RULE_001"}
        merged = rule_alert_review_service.merge_status_with_item(
            db, "preprocessed_run_999", item, is_summary=False
        )
        # Should keep original status
        assert merged["status"] == "NEW"

    def test_confirmed_fraud_only_manual(self, db: Session):
        """Test that CONFIRMED_FRAUD must be set manually."""
        # System should allow manual CONFIRMED_FRAUD
        result = rule_alert_review_service.create_or_update_alert_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="CONFIRMED_FRAUD",
            alert_id="alert_001",
            analyst_notes="Manual fraud confirmation",
        )
        assert result["new_status"] == "CONFIRMED_FRAUD"


class TestAlertReviewValidations:
    """Test validations for alert reviews."""

    def test_all_valid_statuses_accepted(self, db: Session):
        """Test that all valid statuses are accepted."""
        valid_statuses = ["NEW", "IN_REVIEW", "DISMISSED", "FALSE_POSITIVE", "CONFIRMED_FRAUD"]
        for status in valid_statuses:
            review = alert_review_repository.create_review(
                db,
                source_run=f"preprocessed_run_test_{status}",
                rule_code="RULE_001",
                new_status=status,
                alert_id=f"alert_{status}",
            )
            assert review.new_status == status

    def test_analyst_label_mapping(self, db: Session):
        """Test analyst_label mapping based on status."""
        # CONFIRMED_FRAUD should map to "fraud"
        review = alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="CONFIRMED_FRAUD",
            alert_id="alert_fraud",
            analyst_label="fraud",
        )
        assert review.analyst_label == "fraud"

        # FALSE_POSITIVE should map to "false_positive"
        review = alert_review_repository.create_review(
            db,
            source_run="preprocessed_run_1",
            rule_code="RULE_001",
            new_status="FALSE_POSITIVE",
            alert_id="alert_fp",
            analyst_label="false_positive",
        )
        assert review.analyst_label == "false_positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
