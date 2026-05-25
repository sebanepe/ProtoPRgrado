from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    Text,
    ForeignKey,
    Numeric,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="user")
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    failed_login_attempts = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    logs = relationship("SystemLog", back_populates="user", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    amount = Column(Numeric(14, 2), nullable=False)
    transaction_type = Column(String(50), nullable=True)
    channel = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    device_id = Column(String(255), nullable=True)
    customer_hash = Column(String(255), nullable=True, index=True)
    transaction_datetime = Column(DateTime(timezone=True), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    is_fraud = Column(Boolean, nullable=False, default=False)
    imported_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    alerts = relationship("FraudAlert", back_populates="transaction", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_name = Column(String(255), nullable=True)
    file_path = Column(String(1024), nullable=True)
    file_hash = Column(String(255), nullable=True)
    source_type = Column(String(50), nullable=False, default="CSV")
    total_records = Column(Integer, nullable=False, default=0)
    valid_records = Column(Integer, nullable=False, default=0)
    invalid_records = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="PENDING")
    uploaded_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)


# New models: roles, permissions, role_permissions


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_system = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Permission(Base):
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(200), unique=True, nullable=False)
    module = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RolePermission(Base):
    __tablename__ = "role_permissions"

    id = Column(Integer, primary_key=True, index=True)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    permission_id = Column(Integer, ForeignKey("permissions.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("uq_role_permission", "role_id", "permission_id", unique=True),
    )


# Preprocessing runs


class PreprocessingRun(Base):
    __tablename__ = "preprocessing_runs"

    id = Column(Integer, primary_key=True, index=True)
    input_dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    executed_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    output_file_path = Column(String(1024), nullable=True)
    status = Column(String(50), nullable=False, default="PENDING")
    total_records = Column(Integer, nullable=False, default=0)
    processed_records = Column(Integer, nullable=False, default=0)
    removed_records = Column(Integer, nullable=False, default=0)
    params_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)


class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    preprocessing_run_id = Column(Integer, ForeignKey("preprocessing_runs.id", ondelete="SET NULL"), nullable=True)
    name = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    row_count = Column(Integer, nullable=False, default=0)
    feature_columns_json = Column(Text, nullable=True)
    excluded_columns_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    feature_set_id = Column(Integer, ForeignKey("feature_sets.id", ondelete="SET NULL"), nullable=True)
    trained_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name = Column(String(255), nullable=False)
    algorithm = Column(String(255), nullable=False)
    version = Column(String(100), nullable=False)
    artifact_path = Column(String(1024), nullable=False)
    target_column = Column(String(255), nullable=False, default="is_fraud")
    feature_columns_json = Column(Text, nullable=True)
    excluded_columns_json = Column(Text, nullable=True)
    hyperparameters_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    confusion_matrix_json = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="TRAINED")
    is_candidate = Column(Boolean, nullable=False, default=True)
    trained_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("uq_model_name_version", "name", "version", unique=True),
    )


class ModelConfig(Base):
    __tablename__ = "model_config"

    id = Column(Integer, primary_key=True, index=True)
    active_model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="SET NULL"), nullable=True)
    created_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    alert_threshold = Column(Float, nullable=False, default=0.8)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    deactivated_at = Column(DateTime(timezone=True), nullable=True)

    # Business rules enforced elsewhere: only one active config at a time

    # Backwards-compatible fields expected by existing code/tests
    updated_by = Column(String(255), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ScoringRun(Base):
    __tablename__ = "scoring_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="SET NULL"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    feature_set_id = Column(Integer, ForeignKey("feature_sets.id", ondelete="SET NULL"), nullable=True)
    model_config_id = Column(Integer, ForeignKey("model_config.id", ondelete="SET NULL"), nullable=True)
    executed_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    threshold_used = Column(Float, nullable=True)
    total_scored = Column(Integer, nullable=False, default=0)
    alerts_generated = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="PENDING")
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)


class ScoredTransaction(Base):
    __tablename__ = "scored_transactions"

    id = Column(Integer, primary_key=True, index=True)
    scoring_run_id = Column(Integer, ForeignKey("scoring_runs.id", ondelete="CASCADE"), nullable=False)
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="SET NULL"), nullable=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    prediction_label = Column(Integer, nullable=True)
    explanation_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, index=True)
    scored_transaction_id = Column(Integer, ForeignKey("scored_transactions.id", ondelete="SET NULL"), nullable=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="SET NULL"), nullable=True)
    scoring_run_id = Column(Integer, ForeignKey("scoring_runs.id", ondelete="SET NULL"), nullable=True)
    assigned_to_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    model_name = Column(String(255), nullable=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default="NEW")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, onupdate=func.now())
    transaction = relationship("Transaction", back_populates="alerts")


class AlertStatusHistory(Base):
    __tablename__ = "alert_status_history"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("fraud_alerts.id", ondelete="CASCADE"), nullable=False)
    changed_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    old_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=False)
    comment = Column(Text, nullable=True)
    changed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FraudCase(Base):
    __tablename__ = "fraud_cases"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("fraud_alerts.id", ondelete="SET NULL"), nullable=True)
    assigned_to_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    opened_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    case_number = Column(String(100), unique=True, nullable=False)
    status = Column(String(50), nullable=False, default="OPEN")
    priority = Column(String(50), nullable=False, default="MEDIUM")
    summary = Column(Text, nullable=True)
    conclusion = Column(Text, nullable=True)
    opened_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)


class CaseComment(Base):
    __tablename__ = "case_comments"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("fraud_cases.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=False)
    comment = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ReportExport(Base):
    __tablename__ = "report_exports"

    id = Column(Integer, primary_key=True, index=True)
    requested_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    report_type = Column(String(100), nullable=False)
    file_path = Column(String(1024), nullable=True)
    status = Column(String(50), nullable=False, default="PENDING")
    filters_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="logs")


# Indexes for performance
# Indexes defined via column `index=True` flags above; avoid duplicate Index definitions


# Note: ModelConfig moved earlier to reference `ml_models` and support advanced configs
