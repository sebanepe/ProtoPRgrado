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
    is_active = Column(Boolean, nullable=False, default=True)
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
    is_fraud = Column(Boolean, nullable=False, default=False)
    imported_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    alerts = relationship("FraudAlert", back_populates="transaction", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    total_records = Column(Integer, nullable=False, default=0)
    valid_records = Column(Integer, nullable=False, default=0)
    invalid_records = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


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


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False)
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String(50), nullable=True)
    model_name = Column(String(255), nullable=True)
    status = Column(String(50), nullable=False, default="open")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    transaction = relationship("Transaction", back_populates="alerts")


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
