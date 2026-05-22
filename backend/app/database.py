from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings


# Create engine with sensible defaults: enable pool_pre_ping to avoid stale
# connections and use a short connect timeout for networked DBs. For sqlite
# use the expected sqlite-specific `check_same_thread` connect arg.
connect_args = {}
engine_kwargs = {"pool_pre_ping": True, "pool_timeout": 5}

if settings.database_url.startswith("sqlite"):
    # SQLite in-process database needs this flag when used with threads
    connect_args = {"check_same_thread": False}
else:
    # For psycopg2, set a short connect timeout so health checks fail fast
    connect_args = {"connect_timeout": 5}

engine = create_engine(settings.database_url, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
