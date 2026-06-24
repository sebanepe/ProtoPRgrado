from __future__ import annotations

from backend.app.models.models import (
    ArtifactRegistry,
    Dataset,
    PreprocessingRun,
    RuleAlertReview,
    RuleRun,
    User,
)
from backend.app.services.traceability_service import get_import_alert_summary

_SENSITIVE_KEYS = {"is_fraud", "confirmed_fraud", "PAN_TARJETA", "TARJETA",
                   "password", "password_hash", "pan_card", "raw_card"}


def _make_user(db, full_name="Analista Test", email="analista@test.com"):
    u = User(full_name=full_name, email=email, password_hash="x", role="user")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_dataset(db, name="DS1", status="IMPORTED", total_records=100, uploaded_by_id=None):
    d = Dataset(name=name, original_filename=f"{name}.csv",
                status=status, total_records=total_records,
                uploaded_by_id=uploaded_by_id)
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


def _make_prep_run(db, dataset_id, status="SUCCESS", processed_records=90, executed_by_id=None):
    pr = PreprocessingRun(
        input_dataset_id=dataset_id,
        status=status,
        total_records=100,
        processed_records=processed_records,
        removed_records=10,
        executed_by_id=executed_by_id,
    )
    db.add(pr)
    db.commit()
    db.refresh(pr)
    return pr


def _make_rule_run(db, source_run, detailed=5, grouped=2, status="AVAILABLE"):
    rr = RuleRun(
        source_run=source_run,
        detailed_alert_count=detailed,
        grouped_alert_count=grouped,
        status=status,
    )
    db.add(rr)
    db.commit()
    db.refresh(rr)
    return rr


def _make_review(db, source_run, alert_id=None, summary_alert_id=None,
                 new_status="NEW", rule_code="R001"):
    r = RuleAlertReview(
        source_run=source_run,
        alert_id=alert_id,
        summary_alert_id=summary_alert_id,
        rule_code=rule_code,
        new_status=new_status,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def _make_artifact(db, source_run, artifact_type="PREPROCESSED_CSV",
                   status="AVAILABLE", row_count=100):
    ar = ArtifactRegistry(
        source_run=source_run,
        artifact_type=artifact_type,
        phase="PHASE_A",
        file_path=f"/tmp/{source_run}_{artifact_type}.csv",
        file_name=f"{artifact_type}.csv",
        status=status,
        row_count=row_count,
    )
    db.add(ar)
    db.commit()
    db.refresh(ar)
    return ar


# ─── Test 1 ────────────────────────────────────────────────────────────────
def test_dataset_sin_preprocesamiento_aparece_en_resumen(db_session):
    """Dataset without preprocessing run must appear with null downstream fields and 0 counts."""
    _make_dataset(db_session, name="DS_sin_prep", status="IMPORTED")

    rows = get_import_alert_summary(db_session)

    assert len(rows) == 1
    row = rows[0]
    assert row["preprocessing_run_id"] is None
    assert row["rule_run_id"] is None
    assert row["detailed_alert_count"] == 0
    assert row["grouped_alert_count"] == 0
    assert row["detailed_confirmed_by_review_count"] == 0
    assert row["grouped_confirmed_by_review_count"] == 0


# ─── Test 2 ────────────────────────────────────────────────────────────────
def test_dataset_con_preprocesamiento_sin_reglas(db_session):
    """Dataset with preprocessing but no rule run and no rule artifacts → rule fields are 0/None."""
    ds = _make_dataset(db_session, name="DS_prep_no_rules")
    _make_prep_run(db_session, ds.id)

    rows = get_import_alert_summary(db_session)

    assert len(rows) == 1
    row = rows[0]
    assert row["preprocessing_run_id"] is not None
    assert row["rule_run_id"] is None
    assert row["rule_run_status"] is None
    assert row["detailed_alert_count"] == 0
    assert row["grouped_alert_count"] == 0


# ─── Test 3 ────────────────────────────────────────────────────────────────
def test_dataset_con_pipeline_completo(db_session):
    """Full pipeline: counts from RuleRun columns, NOT from rule_alert_reviews."""
    ds = _make_dataset(db_session)
    pr = _make_prep_run(db_session, ds.id)
    source_run = f"preprocessed_run_{pr.id}"
    _make_rule_run(db_session, source_run, detailed=12, grouped=4)

    rows = get_import_alert_summary(db_session)

    assert len(rows) == 1
    row = rows[0]
    assert row["rule_run_id"] is not None
    assert row["detailed_alert_count"] == 12
    assert row["grouped_alert_count"] == 4
    assert row["rule_run_status"] == "AVAILABLE"


# ─── Test 4 ────────────────────────────────────────────────────────────────
def test_conteo_revisiones_usa_status_mas_reciente(db_session):
    """Latest review per alert_id wins. Two reviews same alert: last = CONFIRMED_FRAUD."""
    ds = _make_dataset(db_session)
    pr = _make_prep_run(db_session, ds.id)
    source_run = f"preprocessed_run_{pr.id}"
    _make_rule_run(db_session, source_run)

    # First review: NEW
    _make_review(db_session, source_run, alert_id="A001", new_status="NEW")
    # Second review (higher id): CONFIRMED_FRAUD — this should win
    _make_review(db_session, source_run, alert_id="A001", new_status="CONFIRMED_FRAUD")

    rows = get_import_alert_summary(db_session)
    row = rows[0]

    assert row["detailed_confirmed_by_review_count"] == 1
    assert row["detailed_new_count"] == 0


# ─── Test 5 ────────────────────────────────────────────────────────────────
def test_multiples_preprocessing_runs_por_dataset(db_session):
    """One dataset with 2 preprocessing runs → 2 rows, same dataset_id."""
    ds = _make_dataset(db_session)
    _make_prep_run(db_session, ds.id)
    _make_prep_run(db_session, ds.id, status="FAILED", processed_records=0)

    rows = get_import_alert_summary(db_session)

    assert len(rows) == 2
    assert all(r["dataset_id"] == ds.id for r in rows)
    statuses = {r["preprocessing_run_status"] for r in rows}
    assert statuses == {"SUCCESS", "FAILED"}


# ─── Test 6 ────────────────────────────────────────────────────────────────
def test_estado_artefacto_reflejado(db_session):
    """AVAILABLE artifact is correctly reflected in artifact_preprocessed_csv field."""
    ds = _make_dataset(db_session)
    pr = _make_prep_run(db_session, ds.id)
    source_run = f"preprocessed_run_{pr.id}"
    _make_artifact(db_session, source_run, artifact_type="PREPROCESSED_CSV", status="AVAILABLE")

    rows = get_import_alert_summary(db_session)
    row = rows[0]

    assert row["artifact_preprocessed_csv"] == "AVAILABLE"
    assert row["artifact_rule_alerts_csv"] == "MISSING"
    assert row["artifact_rule_summary_csv"] == "MISSING"


# ─── Test 7 ────────────────────────────────────────────────────────────────
def test_campos_sensibles_no_expuestos(db_session):
    """No row may contain keys that expose sensitive fields."""
    ds = _make_dataset(db_session)
    pr = _make_prep_run(db_session, ds.id)
    source_run = f"preprocessed_run_{pr.id}"
    _make_rule_run(db_session, source_run)
    _make_review(db_session, source_run, alert_id="A1", new_status="CONFIRMED_FRAUD")

    rows = get_import_alert_summary(db_session)

    for row in rows:
        exposed = _SENSITIVE_KEYS & set(row.keys())
        assert not exposed, f"Sensitive keys found in response: {exposed}"


# ─── Test 8 ────────────────────────────────────────────────────────────────
def test_rule_run_derivado_desde_artefactos(db_session):
    """
    If no RuleRun record exists but RULE_ALERTS_CSV and RULE_SUMMARY_CSV are
    AVAILABLE in artifact_registry, counts are derived from row_count and
    rule_run_status is 'DERIVADO'.
    """
    ds = _make_dataset(db_session)
    pr = _make_prep_run(db_session, ds.id)
    source_run = f"preprocessed_run_{pr.id}"

    # No RuleRun record — simulates the inconsistency seen with MuestraMayoDefensa
    _make_artifact(db_session, source_run, artifact_type="RULE_ALERTS_CSV",
                   status="AVAILABLE", row_count=250)
    _make_artifact(db_session, source_run, artifact_type="RULE_SUMMARY_CSV",
                   status="AVAILABLE", row_count=80)

    rows = get_import_alert_summary(db_session)

    assert len(rows) == 1
    row = rows[0]
    assert row["rule_run_id"] is None
    assert row["rule_run_status"] == "DERIVADO"
    assert row["detailed_alert_count"] == 250
    assert row["grouped_alert_count"] == 80


# ─── Test 9 ────────────────────────────────────────────────────────────────
def test_usuario_importador_y_preprocesador_en_respuesta(db_session):
    """User who imported dataset and user who ran preprocessing appear in response."""
    user1 = _make_user(db_session, full_name="María López", email="maria@test.com")
    user2 = _make_user(db_session, full_name="Juan Pérez", email="juan@test.com")

    ds = _make_dataset(db_session, uploaded_by_id=user1.id)
    _make_prep_run(db_session, ds.id, executed_by_id=user2.id)

    rows = get_import_alert_summary(db_session)

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset_uploaded_by"] == "María López"
    assert row["preprocessing_executed_by"] == "Juan Pérez"
