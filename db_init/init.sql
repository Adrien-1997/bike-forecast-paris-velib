PRAGMA threads=4;
CREATE SCHEMA IF NOT EXISTS dim;
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
-- Référentiels
CREATE TABLE IF NOT EXISTS dim.station (
  station_id INTEGER,
  name VARCHAR,
  lat DOUBLE,
  lon DOUBLE,
  capacity INTEGER,
  status_init VARCHAR,
  valid_from TIMESTAMP,
  valid_to   TIMESTAMP,
  is_current BOOLEAN
);
-- BRONZE (brut typé/dédupliqué)
CREATE TABLE IF NOT EXISTS bronze.raw_snapshots_5min (
  ts_utc TIMESTAMP,
  ts_paris TIMESTAMP,
  tbin_utc TIMESTAMP,
  station_id INTEGER,
  bikes INTEGER,
  capacity INTEGER,
  mechanical INTEGER,
  ebike INTEGER,
  status VARCHAR,
  lat DOUBLE,
  lon DOUBLE,
  ingested_at TIMESTAMP,
  ingest_latency_s DOUBLE,
  source_etag VARCHAR
);
CREATE TABLE IF NOT EXISTS bronze.weather_5min (
  tbin_utc TIMESTAMP,
  temp_C DOUBLE,
  wind_mps DOUBLE,
  precip_mm DOUBLE,
  weather_src VARCHAR
);
-- SILVER (prêt monitoring par jour)
CREATE TABLE IF NOT EXISTS silver.daily_compact (
  date DATE,
  station_id INTEGER,
  bins INTEGER,
  bins_present INTEGER,
  completeness_pct DOUBLE,
  bikes_median DOUBLE,
  bikes_p90 DOUBLE,
  bikes_min INTEGER,
  bikes_max INTEGER,
  ingest_latency_p95_s DOUBLE,
  status_mode VARCHAR
);
CREATE TABLE IF NOT EXISTS silver.station_health_daily (
  date DATE,
  station_id INTEGER,
  gaps_count INTEGER,
  max_gap_bins INTEGER,
  outlier_bins INTEGER,
  alerts_json VARCHAR
);
-- GOLD (indicateurs globaux par jour)
CREATE TABLE IF NOT EXISTS gold.data_health_daily (
  date DATE,
  ts_max TIMESTAMP,
  freshness_min DOUBLE,
  completeness_pct_24h DOUBLE,
  missing_bins_24h INTEGER,
  ingest_latency_p95_s DOUBLE,
  schema_ok BOOLEAN
);
-- VUES de confort (fenêtres récentes)
CREATE VIEW IF NOT EXISTS gold.health_7d AS
  SELECT * FROM gold.data_health_daily
  WHERE date >= current_date() - INTERVAL 7 DAY;
CREATE VIEW IF NOT EXISTS silver.station_health_7d AS
  SELECT * FROM silver.station_health_daily
  WHERE date >= current_date() - INTERVAL 7 DAY;
