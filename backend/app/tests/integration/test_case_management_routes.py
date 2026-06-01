"""
Fase D3.1 — Integration tests for case management HTTP endpoints.
Uses the integration conftest: SQLite in-memory + TestClient with admin auth header.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.models.models import CaseManagementCase, CaseManagementComment, CaseManagementHistory


@pytest.fixture(autouse=True)
def clean_cases(db_session: Session):
    """Ensure a clean state for each test."""
    db_session.query(CaseManagementHistory).delete()
    db_session.query(CaseManagementComment).delete()
    db_session.query(CaseManagementCase).delete()
    db_session.commit()
    yield
    db_session.query(CaseManagementHistory).delete()
    db_session.query(CaseManagementComment).delete()
    db_session.query(CaseManagementCase).delete()
    db_session.commit()


# ── helpers ───────────────────────────────────────────────────────────────────

def _create(client: TestClient, **overrides) -> dict:
    payload = {
        "title": "Integration Test Case",
        "origin_type": "MANUAL",
        "priority": "MEDIUM",
        "description": "Created during integration test",
    }
    payload.update(overrides)
    resp = client.post("/api/cases", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── POST /api/cases ───────────────────────────────────────────────────────────

class TestCreateCase:
    def test_post_case_valid(self, test_client: TestClient):
        resp = test_client.post("/api/cases", json={
            "title": "Valid Case",
            "origin_type": "RULE_ALERT",
            "priority": "HIGH",
            "summary_alert_id": "alert_001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["case_code"].startswith("CASE-")
        assert data["status"] == "OPEN"
        assert "is_fraud" not in data
        assert "confirmed_fraud" not in data

    def test_post_case_invalid_origin(self, test_client: TestClient):
        resp = test_client.post("/api/cases", json={
            "title": "Bad case",
            "origin_type": "INVALID_TYPE",
            "description": "some desc",
        })
        assert resp.status_code == 422

    def test_post_case_missing_title(self, test_client: TestClient):
        resp = test_client.post("/api/cases", json={
            "title": "",
            "origin_type": "MANUAL",
            "description": "desc",
        })
        assert resp.status_code == 422

    def test_post_case_no_refs(self, test_client: TestClient):
        resp = test_client.post("/api/cases", json={
            "title": "No refs",
            "origin_type": "MANUAL",
        })
        assert resp.status_code == 422


# ── GET /api/cases ────────────────────────────────────────────────────────────

class TestListCases:
    def test_get_cases_list(self, test_client: TestClient):
        _create(test_client)
        _create(test_client, title="Second case", description="second")
        resp = test_client.get("/api/cases")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] >= 2

    def test_get_cases_filter_status(self, test_client: TestClient):
        _create(test_client)
        resp = test_client.get("/api/cases?status=OPEN")
        assert resp.status_code == 200
        data = resp.json()
        assert all(c["status"] == "OPEN" for c in data["items"])

    def test_get_cases_pagination(self, test_client: TestClient):
        resp = test_client.get("/api/cases?page=1&page_size=10")
        assert resp.status_code == 200


# ── GET /api/cases/summary ────────────────────────────────────────────────────

class TestSummary:
    def test_summary(self, test_client: TestClient):
        _create(test_client)
        resp = test_client.get("/api/cases/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "by_status" in data
        assert "by_priority" in data
        assert data["total"] >= 1

    def test_summary_not_confused_with_case_id(self, test_client: TestClient):
        # Route /api/cases/summary must not be interpreted as /api/cases/{case_id}
        resp = test_client.get("/api/cases/summary")
        assert resp.status_code == 200
        # Must return a dict with 'by_status', not a 404 or case detail
        assert "by_status" in resp.json()


# ── GET /api/cases/{case_id} ──────────────────────────────────────────────────

class TestGetCase:
    def test_get_case_detail(self, test_client: TestClient):
        created = _create(test_client)
        resp = test_client.get(f"/api/cases/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == created["id"]

    def test_get_case_not_found(self, test_client: TestClient):
        resp = test_client.get("/api/cases/99999")
        assert resp.status_code == 404


# ── PATCH /api/cases/{case_id} ────────────────────────────────────────────────

class TestUpdateCase:
    def test_patch_case_status(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.patch(f"/api/cases/{case['id']}", json={"status": "IN_ANALYSIS"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "IN_ANALYSIS"

    def test_patch_case_priority(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.patch(f"/api/cases/{case['id']}", json={"priority": "CRITICAL"})
        assert resp.status_code == 200
        assert resp.json()["priority"] == "CRITICAL"

    def test_patch_case_invalid_status(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.patch(f"/api/cases/{case['id']}", json={"status": "BOGUS"})
        assert resp.status_code == 422


# ── POST /api/cases/{case_id}/comments ───────────────────────────────────────

class TestComments:
    def test_post_comment(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.post(f"/api/cases/{case['id']}/comments", json={"comment_text": "Investigating now"})
        assert resp.status_code == 200
        assert resp.json()["comment_text"] == "Investigating now"

    def test_get_comments(self, test_client: TestClient):
        case = _create(test_client)
        test_client.post(f"/api/cases/{case['id']}/comments", json={"comment_text": "Note 1"})
        test_client.post(f"/api/cases/{case['id']}/comments", json={"comment_text": "Note 2"})
        resp = test_client.get(f"/api/cases/{case['id']}/comments")
        assert resp.status_code == 200
        comments = resp.json()
        assert len(comments) == 2

    def test_post_empty_comment_rejected(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.post(f"/api/cases/{case['id']}/comments", json={"comment_text": ""})
        assert resp.status_code == 422


# ── GET /api/cases/{case_id}/history ─────────────────────────────────────────

class TestHistory:
    def test_get_history(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.get(f"/api/cases/{case['id']}/history")
        assert resp.status_code == 200
        history = resp.json()
        assert isinstance(history, list)
        assert any(h["action"] == "CASE_CREATED" for h in history)


# ── POST /api/cases/{case_id}/close ──────────────────────────────────────────

class TestCloseCase:
    def test_close_case(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.post(f"/api/cases/{case['id']}/close", json={"conclusion": "Issue resolved after review"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "CLOSED"
        assert data["conclusion"] == "Issue resolved after review"
        assert data["closed_at"] is not None

    def test_close_no_conclusion(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.post(f"/api/cases/{case['id']}/close", json={"conclusion": ""})
        assert resp.status_code == 422

    def test_close_does_not_expose_is_fraud(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.post(f"/api/cases/{case['id']}/close", json={"conclusion": "Done"})
        assert resp.status_code == 200
        data = resp.json()
        assert "is_fraud" not in data
        assert "confirmed_fraud" not in data


# ── POST /api/cases/{case_id}/reopen ─────────────────────────────────────────

class TestReopenCase:
    def test_reopen_case(self, test_client: TestClient):
        case = _create(test_client)
        test_client.post(f"/api/cases/{case['id']}/close", json={"conclusion": "Closing to reopen"})
        resp = test_client.post(f"/api/cases/{case['id']}/reopen", json={"user": "supervisor"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "OPEN"
        assert data["closed_at"] is None

    def test_reopen_non_closed_returns_422(self, test_client: TestClient):
        case = _create(test_client)
        resp = test_client.post(f"/api/cases/{case['id']}/reopen", json={})
        assert resp.status_code == 422


# ── POST /api/cases/from-scoring-result ──────────────────────────────────────

class TestFromScoringResult:
    def test_from_scoring_result(self, test_client: TestClient):
        resp = test_client.post("/api/cases/from-scoring-result", json={
            "title": "High-risk scoring result",
            "scoring_run_id": "batch_run_42",
            "transaction_id": "txn_0099",
            "priority": "HIGH",
            "description": "Score exceeded threshold",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["origin_type"] == "SCORING_RESULT"
        assert data["scoring_run_id"] == "batch_run_42"
        assert "is_fraud" not in data
        assert "confirmed_fraud" not in data
