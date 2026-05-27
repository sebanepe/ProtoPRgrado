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

    # Prefer Postgres ON CONFLICT upsert/do-nothing for idempotent inserts when
    # a unique key on (transaction_id, dataset_id) exists. Attempt batched
    # INSERT ... ON CONFLICT DO NOTHING; if the DB doesn't have the expected
    # unique index/constraint, fall back to safe batched/per-row insert logic.
    batch_size = 1000
    try:
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            stmt = pg_insert(Transaction.__table__).values(batch)
            # conflict target is transaction_id + dataset_id (behavioral grouping)
            stmt = stmt.on_conflict_do_nothing(index_elements=["transaction_id", "dataset_id"])
            db.execute(stmt)
        db.commit()
        return len(records)
    except SQLAlchemyError as e:
        # If ON CONFLICT failed because the unique index doesn't exist, or any
        # other SQLAlchemy error, log and rollback then try conservative fallback.
        logger.warning("ON CONFLICT bulk insert failed, falling back to safe inserts: %s", e)
        db.rollback()

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
    return inserted
