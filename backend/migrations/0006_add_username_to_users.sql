-- Migration 0006: add username and updated_at to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(100) UNIQUE;
CREATE INDEX IF NOT EXISTS ix_users_username ON users (username);
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE;
