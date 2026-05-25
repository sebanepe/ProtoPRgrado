-- Migration: add dataset_id FK to transactions
-- Adds nullable integer column dataset_id referencing datasets(id)

BEGIN;

-- Add the column if it does not exist
ALTER TABLE IF EXISTS public.transactions
ADD COLUMN IF NOT EXISTS dataset_id INTEGER;

-- Add foreign key constraint (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_transactions_dataset_id'
    ) THEN
        ALTER TABLE public.transactions
        ADD CONSTRAINT fk_transactions_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON DELETE SET NULL;
    END IF;
END$$;

-- Create index to speed lookups
CREATE INDEX IF NOT EXISTS ix_transactions_dataset_id ON public.transactions(dataset_id);

COMMIT;

-- Notes:
-- Apply with psql: docker compose exec -T db psql -U protouser -d protodb < backend/migrations/0002_add_dataset_id_to_transactions.sql
