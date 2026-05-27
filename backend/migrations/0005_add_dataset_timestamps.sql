-- 0005_add_dataset_timestamps.sql
-- Add started_at, finished_at and error_message to datasets table
-- Compatible with PostgreSQL
BEGIN;

ALTER TABLE IF EXISTS datasets
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE IF EXISTS datasets
    ADD COLUMN IF NOT EXISTS finished_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE IF EXISTS datasets
    ADD COLUMN IF NOT EXISTS error_message TEXT;

COMMIT;
