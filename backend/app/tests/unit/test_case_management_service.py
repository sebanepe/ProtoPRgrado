"""
Fase D3.1 — Unit tests for case_management_service.
Uses the root conftest db_session (SQLite in-memory).
"""
import pytest
from sqlalchemy.orm import Session

from backend.app.models.models import CaseManagementCase, CaseManagementComment, CaseManagementHistory
from backend.app.services import case_management_service as svc


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_case(db: Session, **overrides) -> dict:
    defaults = dict(
        title="Test Case",
        origin_type="MANUAL",
        priority="MEDIUM",
        description="Test description",
    )
    defaults.update(overrides)
    return svc.create_case(db, **defaults)


# ── create_case ───────────────────────────────────────────────────────────────

class TestCreateCase:
    def test_create_case_valid(self, db_session: Session):
        result = _make_case(db_session)
        assert result["case_code"].startswith("CASE-")
        assert result["status"] == "OPEN"
        assert result["origin_type"] == "MANUAL"
        assert result["title"] == "Test Case"

    def test_create_case_generates_unique_codes(self, db_session: Session):
        c1 = _make_case(db_session, title="A", description="desc A")
        c2 = _make_case(db_session, title="B", description="desc B")
        assert c1["case_code"] != c2["case_code"]

    def test_create_case_invalid_origin_type(self, db_session: Session):
        with pytest.raises(ValueError, match="Invalid origin_type"):
            _make_case(db_session, origin_type="FAKE_TYPE")

    def test_create_case_invalid_priority(self, db_session: Session):
        with pytest.raises(ValueError, match="Invalid priority"):
            _make_case(db_session, priority="SUPER_HIGH")

    def test_create_case_missing_title(self, db_session: Session):
        with pytest.raises(ValueError, match="title is required"):
            _make_case(db_session, title="")

    def test_create_case_empty_no_refs(self, db_session: Session):
        with pytest.raises(ValueError, match="at least one reference"):
            svc.create_case(db_session, title="No refs", origin_type="MANUAL")

    def test_create_case_with_scoring_ref(self, db_session: Session):
        result = svc.create_case(
            db_session,
            title="From scoring",
            origin_type="SCORING_RESULT",
            scoring_run_id="batch_run_99",
        )
        assert result["scoring_run_id"] == "batch_run_99"
        assert result["status"] == "OPEN"

    def test_no_is_fraud_in_response(self, db_session: Session):
        result = _make_case(db_session)
        assert "is_fraud" not in result
        assert "confirmed_fraud" not in result
        assert "PAN_TARJETA" not in result
        assert "pan_card" not in result


# ── list_cases / get_case ─────────────────────────────────────────────────────

class TestListGetCase:
    def test_list_cases_returns_all(self, db_session: Session):
        _make_case(db_session, title="C1", description="d1")
        _make_case(db_session, title="C2", description="d2")
        result = svc.list_cases(db_session)
        assert result["total"] >= 2

    def test_list_cases_filter_by_status(self, db_session: Session):
        _make_case(db_session, title="Open", description="open desc")
        result = svc.list_cases(db_session, status="OPEN")
        assert all(c["status"] == "OPEN" for c in result["items"])

    def test_get_case_returns_dict(self, db_session: Session):
        created = _make_case(db_session)
        fetched = svc.get_case(db_session, created["id"])
        assert fetched["id"] == created["id"]
        assert fetched["case_code"] == created["case_code"]

    def test_get_case_not_found_returns_none(self, db_session: Session):
        assert svc.get_case(db_session, 999999) is None


# ── update_case ───────────────────────────────────────────────────────────────

class TestUpdateCase:
    def test_update_status(self, db_session: Session):
        case = _make_case(db_session)
        updated = svc.update_case(db_session, case["id"], status="IN_ANALYSIS")
        assert updated["status"] == "IN_ANALYSIS"

    def test_update_priority(self, db_session: Session):
        case = _make_case(db_session)
        updated = svc.update_case(db_session, case["id"], priority="HIGH")
        assert updated["priority"] == "HIGH"

    def test_update_invalid_status(self, db_session: Session):
        case = _make_case(db_session)
        with pytest.raises(ValueError, match="Invalid status"):
            svc.update_case(db_session, case["id"], status="NONEXISTENT")

    def test_update_invalid_priority(self, db_session: Session):
        case = _make_case(db_session)
        with pytest.raises(ValueError, match="Invalid priority"):
            svc.update_case(db_session, case["id"], priority="MEGA")

    def test_update_rejects_forbidden_field(self, db_session: Session):
        case = _make_case(db_session)
        with pytest.raises(ValueError, match="cannot be set"):
            svc.update_case(db_session, case["id"], is_fraud=True)

    def test_update_status_history_recorded(self, db_session: Session):
        case = _make_case(db_session)
        svc.update_case(db_session, case["id"], status="ESCALATED", changed_by="analyst1")
        history = svc.list_history(db_session, case["id"])
        actions = [h["action"] for h in history]
        assert "STATUS_CHANGED" in actions

    def test_update_priority_history_recorded(self, db_session: Session):
        case = _make_case(db_session)
        svc.update_case(db_session, case["id"], priority="CRITICAL", changed_by="analyst1")
        history = svc.list_history(db_session, case["id"])
        actions = [h["action"] for h in history]
        assert "PRIORITY_CHANGED" in actions


# ── comments ─────────────────────────────────────────────────────────────────

class TestComments:
    def test_add_comment(self, db_session: Session):
        case = _make_case(db_session)
        comment = svc.add_comment(db_session, case["id"], comment_text="Looks suspicious", user_id="user1")
        assert comment["comment_text"] == "Looks suspicious"
        assert comment["case_id"] == case["id"]

    def test_add_empty_comment_raises(self, db_session: Session):
        case = _make_case(db_session)
        with pytest.raises(ValueError, match="comment_text is required"):
            svc.add_comment(db_session, case["id"], comment_text="")

    def test_list_comments(self, db_session: Session):
        case = _make_case(db_session)
        svc.add_comment(db_session, case["id"], comment_text="First")
        svc.add_comment(db_session, case["id"], comment_text="Second")
        comments = svc.list_comments(db_session, case["id"])
        assert len(comments) == 2

    def test_comment_adds_history_entry(self, db_session: Session):
        case = _make_case(db_session)
        svc.add_comment(db_session, case["id"], comment_text="Check this")
        history = svc.list_history(db_session, case["id"])
        assert any(h["action"] == "COMMENT_ADDED" for h in history)


# ── history ───────────────────────────────────────────────────────────────────

class TestHistory:
    def test_case_created_in_history(self, db_session: Session):
        case = _make_case(db_session)
        history = svc.list_history(db_session, case["id"])
        assert any(h["action"] == "CASE_CREATED" for h in history)


# ── close / reopen ────────────────────────────────────────────────────────────

class TestCloseReopen:
    def test_close_case_without_conclusion_raises(self, db_session: Session):
        case = _make_case(db_session)
        with pytest.raises(ValueError, match="conclusion is required"):
            svc.close_case(db_session, case["id"], conclusion="")

    def test_close_case_with_conclusion(self, db_session: Session):
        case = _make_case(db_session)
        closed = svc.close_case(db_session, case["id"], conclusion="Resolved after review", closed_by="analyst")
        assert closed["status"] == "CLOSED"
        assert closed["conclusion"] == "Resolved after review"
        assert closed["closed_by"] == "analyst"
        assert closed["closed_at"] is not None

    def test_close_case_records_history(self, db_session: Session):
        case = _make_case(db_session)
        svc.close_case(db_session, case["id"], conclusion="Done")
        history = svc.list_history(db_session, case["id"])
        assert any(h["action"] == "CASE_CLOSED" for h in history)

    def test_close_already_closed_raises(self, db_session: Session):
        case = _make_case(db_session)
        svc.close_case(db_session, case["id"], conclusion="First close")
        with pytest.raises(ValueError, match="already closed"):
            svc.close_case(db_session, case["id"], conclusion="Second close")

    def test_reopen_case(self, db_session: Session):
        case = _make_case(db_session)
        svc.close_case(db_session, case["id"], conclusion="Closing to test reopen")
        reopened = svc.reopen_case(db_session, case["id"], user="supervisor")
        assert reopened["status"] == "OPEN"
        assert reopened["closed_at"] is None

    def test_reopen_non_closed_raises(self, db_session: Session):
        case = _make_case(db_session)
        with pytest.raises(ValueError, match="Only CLOSED"):
            svc.reopen_case(db_session, case["id"])

    def test_reopen_records_history(self, db_session: Session):
        case = _make_case(db_session)
        svc.close_case(db_session, case["id"], conclusion="Closing")
        svc.reopen_case(db_session, case["id"], user="supervisor")
        history = svc.list_history(db_session, case["id"])
        assert any(h["action"] == "CASE_REOPENED" for h in history)


# ── summary ───────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_structure(self, db_session: Session):
        summary = svc.get_cases_summary(db_session)
        assert "total" in summary
        assert "by_status" in summary
        assert "by_priority" in summary

    def test_summary_counts(self, db_session: Session):
        _make_case(db_session, title="S1", description="d1", priority="HIGH")
        _make_case(db_session, title="S2", description="d2", priority="LOW")
        summary = svc.get_cases_summary(db_session)
        assert summary["total"] >= 2
        assert summary["by_status"]["OPEN"] >= 2
