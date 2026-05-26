-- Migration: add smote_report_json and pipeline_path to feature_sets
-- Run this against your Postgres DB to add the new columns without recreating the table.

ALTER TABLE feature_sets
  ADD COLUMN IF NOT EXISTS smote_report_json TEXT;

ALTER TABLE feature_sets
  ADD COLUMN IF NOT EXISTS pipeline_path VARCHAR(1024);

-- You may want to create an index on pipeline_path if you query it frequently:
-- CREATE INDEX IF NOT EXISTS idx_feature_sets_pipeline_path ON feature_sets (pipeline_path);
