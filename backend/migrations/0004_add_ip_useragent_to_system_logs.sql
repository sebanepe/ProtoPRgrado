-- Migration: add ip and user_agent columns to system_logs
-- Adds auditing fields to capture requester IP and User-Agent

ALTER TABLE system_logs
  ADD COLUMN IF NOT EXISTS ip VARCHAR(64);

ALTER TABLE system_logs
  ADD COLUMN IF NOT EXISTS user_agent VARCHAR(1024);

-- Optional: create index on user_id to speed up queries filtering by user
-- CREATE INDEX IF NOT EXISTS idx_system_logs_user_id ON system_logs (user_id);

-- Optional: create index on action for quick lookups
-- CREATE INDEX IF NOT EXISTS idx_system_logs_action ON system_logs (action);
