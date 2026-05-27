import argparse
import json
from backend.app.database import SessionLocal
from backend.app.models.models import Transaction
from sqlalchemy.sql import func


def diagnose(dataset_id: int):
    session = SessionLocal()
    try:
        q = session.query(Transaction).filter(Transaction.dataset_id == dataset_id)
        total = q.count()
        country_notnull = q.filter(Transaction.country_code != None).count()
        pos_notnull = q.filter(Transaction.pos_entry_mode != None).count()
        haspin_dist = {0: 0, 1: 0}
        for val, cnt in session.query(Transaction.has_pinblock, func.count(Transaction.id)).filter(Transaction.dataset_id == dataset_id).group_by(Transaction.has_pinblock).all():
            haspin_dist[int(val) if val is not None else -1] = cnt
        merchant_unknown = q.filter((Transaction.merchant_hash == None) | (Transaction.merchant_hash == 'UNKNOWN_MERCHANT')).count()
        # customer_hash that look like raw PAN (digits >=12)
        raw_cust = 0
        for ch, cnt in session.query(Transaction.customer_hash, func.count(Transaction.id)).filter(Transaction.dataset_id == dataset_id).group_by(Transaction.customer_hash).all():
            try:
                if ch and ch.isdigit() and len(ch) >= 12:
                    raw_cust += cnt
            except Exception:
                pass
        out = {
            'dataset_id': dataset_id,
            'total_transactions': total,
            'country_code_not_null': country_notnull,
            'pos_entry_mode_not_null': pos_notnull,
            'has_pinblock_distribution': haspin_dist,
            'merchant_hash_unknown_count': merchant_unknown,
            'customer_hash_raw_pan_count': raw_cust,
        }
        print(json.dumps(out, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', type=int, required=True)
    args = parser.parse_args()
    from sqlalchemy.sql import func
    diagnose(args.dataset_id)
