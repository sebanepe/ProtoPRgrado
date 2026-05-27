import io
import csv


def _make_csv_bytes(rows, cols):
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return b.getvalue().encode("utf-8")


def test_import_valid_csv_integration(test_client, sample_transactions):
    cols = [
        "transaction_id",
        "amount",
        "transaction_type",
        "channel",
        "location",
        "device_id",
        "customer_hash",
        "transaction_datetime",
        "is_fraud",
    ]
    b = _make_csv_bytes(sample_transactions, cols)
    files = {"file": ("tx.csv", b, "text/csv")}
    r = test_client.post("/datasets/import", files=files)
    # endpoint may run synchronously (200/201) or enqueue background import (202)
    assert r.status_code in (200, 201, 202)
    details = r.json().get("details")
    assert details and details.get("total") == len(sample_transactions)


def test_import_invalid_csv_missing_columns(test_client):
    # CSV missing required columns
    rows = [{"foo": "1", "bar": "2"}]
    b = _make_csv_bytes(rows, ["foo", "bar"])
    files = {"file": ("bad.csv", b, "text/csv")}
    r = test_client.post("/datasets/import", files=files)
    assert r.status_code == 400


def test_import_empty_csv_fails(test_client):
    b = b""
    files = {"file": ("empty.csv", b, "text/csv")}
    r = test_client.post("/datasets/import", files=files)
    assert r.status_code == 400
