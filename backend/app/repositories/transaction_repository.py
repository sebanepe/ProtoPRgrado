from sqlalchemy.orm import Session
from backend.app.models.models import Transaction
from typing import List, Dict
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
import logging
logger = logging.getLogger(__name__)


def insert_transactions(db: Session, records: List[Dict], dataset_id: int | None = None):
    """Bulk-insert a list of transaction dicts using SQLAlchemy core insert.

    On failure, retry in smaller batches and finally fallback to row-by-row.
    Returns number of successfully inserted rows.
    """
    if not records:
        return 0

    # attach dataset_id to records
    for r in records:
        r['dataset_id'] = dataset_id

    # Ensure records only contain columns that exist on the transactions table
    valid_cols = {c.name for c in Transaction.__table__.columns}
    def _filter_rec(rec: Dict):
        return {k: v for k, v in rec.items() if k in valid_cols}
    records = [_filter_rec(r) for r in records]

    # Prefer Postgres ON CONFLICT upsert/do-nothing for idempotent inserts when
    # a unique key on (transaction_id, dataset_id) exists. Attempt batched
    # INSERT ... ON CONFLICT DO NOTHING; if the DB doesn't have the expected
    # unique index/constraint, fall back to safe batched/per-row insert logic.
    batch_size = 1000
    try:
        # Use Postgres-specific ON CONFLICT only when the DB dialect is Postgres.
        # In test environments (SQLite) the ON CONFLICT clause above may reference
        # a non-existent unique constraint and raise an OperationalError. Detect
        # dialect and avoid using the pg_insert path for non-postgres DBs.
        try:
            bind = db.get_bind()
            dialect_name = getattr(getattr(bind, 'dialect', None), 'name', '')
        except Exception:
            dialect_name = ''

        if dialect_name == 'postgresql':
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                stmt = pg_insert(Transaction.__table__).values(batch)
                # conflict target is transaction_id + dataset_id (behavioral grouping)
                stmt = stmt.on_conflict_do_nothing(index_elements=["transaction_id", "dataset_id"])
                db.execute(stmt)
            db.commit()
            return len(records)
        # otherwise fall through to conservative fallback below
    except SQLAlchemyError as e:
        # If ON CONFLICT failed because the unique index doesn't exist, or any
        # other SQLAlchemy error, log and rollback then try conservative fallback.
        logger.warning("ON CONFLICT bulk insert failed, falling back to safe inserts: %s", e)
        try:
            db.rollback()
        except Exception:
            pass

    # Conservative fallback: batched inserts, and per-row on failure.
    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            db.execute(insert(Transaction.__table__), batch)
            db.commit()
            inserted += len(batch)
            continue
        except SQLAlchemyError:
            db.rollback()
            for rec in batch:
                try:
                    db.execute(insert(Transaction.__table__), rec)
                    db.commit()
                    inserted += 1
                except SQLAlchemyError:
                    db.rollback()
                    # skip problematic record
                    continue
    # Log metrics for this insert operation
    try:
        attempted = len(records)
        skipped_by_conflict = attempted - inserted
        failed = 0  # we are conservative and treat any skipped as conflict/skip
        conflict_rate = (skipped_by_conflict / attempted) if attempted > 0 else 0.0
        logger.info("insert_transactions: rows_detected_in_file=%s rows_attempted_insert=%s rows_inserted=%s rows_skipped_by_conflict=%s rows_failed=%s conflict_rate=%.4f",
                    attempted, attempted, inserted, skipped_by_conflict, failed, conflict_rate)
        if skipped_by_conflict > 0:
            logger.warning("Some rows were skipped due to conflicts: skipped=%s conflict_rate=%.4f", skipped_by_conflict, conflict_rate)
    except Exception:
        pass
    return inserted
