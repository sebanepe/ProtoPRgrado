"""
Initialize database: create tables, seed roles, permissions, assign role permissions, create initial admin.
Run: python -m backend.app.init_db
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import os

from backend.app.database import Base, engine, SessionLocal
from backend.app.models.models import (
    ArtifactRegistry,
    BatchScoringRun,
    CaseManagementCase,
    CaseManagementComment,
    CaseManagementHistory,
    ModelRegistry,
    Role,
    Permission,
    RolePermission,
    RuleRun,
    SupervisedDatasetRun,
    User,
)
from backend.app.database import engine
from backend.app.services.auth_service import hash_password
from backend.app.repositories.user_repository import get_user_by_email, create_user
from sqlalchemy.exc import OperationalError
from sqlalchemy import text, inspect

PERMISSIONS = [
    ("dashboard.view", "dashboard", "view"),
    ("dataset.view", "dataset", "view"),
    ("dataset.validate", "dataset", "validate"),
    ("dataset.import", "dataset", "import"),
    ("preprocessing.run", "preprocessing", "run"),
    ("preprocessing.view", "preprocessing", "view"),
    ("model.view", "model", "view"),
    ("model.train", "model", "train"),
    ("model.evaluate", "model", "evaluate"),
    ("model.activate", "model", "activate"),
    ("model.configure_threshold", "model", "configure_threshold"),
    ("scoring.run", "scoring", "run"),
    ("scoring.view", "scoring", "view"),
    ("alerts.view", "alerts", "view"),
    ("alerts.detail", "alerts", "detail"),
    ("alerts.update_status", "alerts", "update_status"),
    ("cases.view", "cases", "view"),
    ("cases.create", "cases", "create"),
    ("cases.update", "cases", "update"),
    ("reports.view", "reports", "view"),
    ("reports.export", "reports", "export"),
    ("users.view", "users", "view"),
    ("users.create", "users", "create"),
    ("users.update", "users", "update"),
    ("users.activate", "users", "activate"),
    ("users.deactivate", "users", "deactivate"),
    ("users.assign_role", "users", "assign_role"),
    ("settings.view", "settings", "view"),
    ("settings.update", "settings", "update"),
    ("logs.view", "logs", "view"),
]

ROLE_PERMISSIONS = {
    "ADMIN": [p[0] for p in PERMISSIONS],
    "DATA_SCIENTIST": [
        "dashboard.view",
        "dataset.view",
        "dataset.validate",
        "dataset.import",
        "preprocessing.run",
        "preprocessing.view",
        "model.view",
        "model.train",
        "model.evaluate",
        "model.activate",
        "model.configure_threshold",
        "scoring.run",
        "scoring.view",
        "alerts.view",
        "alerts.detail",
        "reports.view",
        "reports.export",
        "settings.view",
        "settings.update",
    ],
    "FRAUD_ANALYST": [
        "dashboard.view",
        "scoring.run",
        "scoring.view",
        "alerts.view",
        "alerts.detail",
        "alerts.update_status",
        "cases.view",
        "cases.create",
        "cases.update",
        "reports.view",
    ],
}

DEFAULT_ADMIN_EMAIL = os.getenv("DEFAULT_ADMIN_EMAIL", "sebanpb@gmail.com")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "mariokart8$")


def ensure_tables():
    Base.metadata.create_all(bind=engine)


def ensure_traceability_tables():
    """Create C4.2.5 traceability tables without altering existing data."""
    for model in (
        ArtifactRegistry,
        RuleRun,
        ModelRegistry,
        SupervisedDatasetRun,
        BatchScoringRun,
        CaseManagementCase,
        CaseManagementComment,
        CaseManagementHistory,
    ):
        model.__table__.create(bind=engine, checkfirst=True)


def ensure_transactions_merchant_rubro_column():
    """Add merchant_rubro_proxy to transactions if the existing table lacks it."""
    try:
        inspector = inspect(engine)
        columns = {column["name"] for column in inspector.get_columns("transactions")}
    except Exception:
        return

    if "merchant_rubro_proxy" in columns:
        return

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("ALTER TABLE transactions ADD COLUMN merchant_rubro_proxy VARCHAR(20) NULL"))
        else:
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS merchant_rubro_proxy VARCHAR(20) NULL"))


def ensure_user_role_column():
    """If the existing users table doesn't have role_id, add it (SQLite/dev friendly)."""
    conn = engine.connect()
    try:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            res = conn.execute("PRAGMA table_info(users);")
            cols = [r[1] for r in res.fetchall()]
            if "role_id" not in cols:
                # add nullable role_id column
                conn.execute("ALTER TABLE users ADD COLUMN role_id INTEGER;")
                print("Added role_id column to users table")
    finally:
        conn.close()


def get_or_create_role(session: Session, code: str, name: str, description: str | None = None):
    r = session.query(Role).filter(Role.code == code).first()
    if r:
        return r
    r = Role(code=code, name=name, description=description)
    session.add(r)
    session.commit()
    session.refresh(r)
    return r


def get_or_create_permission(session: Session, code: str, module: str, action: str, description: str | None = None):
    p = session.query(Permission).filter(Permission.code == code).first()
    if p:
        return p
    p = Permission(code=code, module=module, action=action, description=description)
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


def add_role_permission(session: Session, role: Role, perm: Permission):
    existing = session.query(RolePermission).filter(RolePermission.role_id == role.id, RolePermission.permission_id == perm.id).first()
    if existing:
        return existing
    rp = RolePermission(role_id=role.id, permission_id=perm.id)
    session.add(rp)
    session.commit()
    session.refresh(rp)
    return rp


def ensure_roles_and_permissions(session: Session):
    roles = {}
    for rc in ROLE_PERMISSIONS.keys():
        roles[rc] = get_or_create_role(session, code=rc, name=rc.title())

    perms = {}
    for code, module, action in PERMISSIONS:
        perms[code] = get_or_create_permission(session, code=code, module=module, action=action)

    for role_code, perm_codes in ROLE_PERMISSIONS.items():
        role = roles[role_code]
        for pc in perm_codes:
            perm = perms.get(pc)
            if perm:
                add_role_permission(session, role, perm)

    return roles, perms


def ensure_admin_user(session: Session, roles: dict):
    # Use raw SQL checks to avoid ORM trying to access columns that may not exist yet
    conn = engine.connect()
    try:
        row = conn.execute(text("SELECT id FROM users WHERE email = :email"), {"email": DEFAULT_ADMIN_EMAIL}).fetchone()
        if row:
            # Ensure existing admin has a username set
            try:
                with engine.begin() as trans:
                    trans.execute(
                        text("UPDATE users SET username = 'admin' WHERE email = :email AND (username IS NULL OR username = '')"),
                        {"email": DEFAULT_ADMIN_EMAIL},
                    )
            except Exception:
                pass
            print("Admin user already exists:", DEFAULT_ADMIN_EMAIL)
            return None
        # create user via raw INSERT to avoid ORM column mismatch
        hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
        # Try INSERT with username column; fall back if column does not exist yet
        try:
            with engine.begin() as trans:
                trans.execute(
                    text(
                        "INSERT INTO users (full_name, email, password_hash, role, username, is_active, failed_login_attempts, created_at)"
                        " VALUES (:full_name, :email, :password_hash, :role, :username, :is_active, :failed_login_attempts, CURRENT_TIMESTAMP)"
                    ),
                    {
                        "full_name": "Initial Admin",
                        "email": DEFAULT_ADMIN_EMAIL,
                        "password_hash": hashed,
                        "role": "ADMIN",
                        "username": "admin",
                        "is_active": True,
                        "failed_login_attempts": 0,
                    },
                )
        except Exception:
            with engine.begin() as trans:
                trans.execute(
                    text(
                        "INSERT INTO users (full_name, email, password_hash, role, is_active, failed_login_attempts, created_at)"
                        " VALUES (:full_name, :email, :password_hash, :role, :is_active, :failed_login_attempts, CURRENT_TIMESTAMP)"
                    ),
                    {
                        "full_name": "Initial Admin",
                        "email": DEFAULT_ADMIN_EMAIL,
                        "password_hash": hashed,
                        "role": "ADMIN",
                        "is_active": True,
                        "failed_login_attempts": 0,
                    },
                )
        # optionally set role_id if column exists and role provided
        admin_role = roles.get("ADMIN")
        if admin_role:
            try:
                with engine.begin() as trans:
                    trans.execute(text("UPDATE users SET role_id = :role_id WHERE email = :email"), {"role_id": admin_role.id, "email": DEFAULT_ADMIN_EMAIL})
            except Exception:
                pass
        print("Created default admin:", DEFAULT_ADMIN_EMAIL)
        return None
    finally:
        conn.close()


def main():
    try:
        ensure_tables()
        ensure_traceability_tables()
        ensure_transactions_merchant_rubro_column()
    except OperationalError as oe:
        msg = f"Could not connect to the database: {oe}\nEnsure your DATABASE_URL is correct and the DB is reachable."
        print(msg)
        raise
    # compatibility: add role_id to users table if missing (dev sqlite)
    try:
        ensure_user_role_column()
    except Exception:
        pass
    session = SessionLocal()
    try:
        roles, perms = ensure_roles_and_permissions(session)
        admin = ensure_admin_user(session, roles)
        print("Roles:", [r.code for r in session.query(Role).all()])
        print("Permissions:", [p.code for p in session.query(Permission).all()][:10], "... total", session.query(Permission).count())
    finally:
        session.close()


if __name__ == "__main__":
    main()
