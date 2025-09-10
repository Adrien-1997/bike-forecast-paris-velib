# tools/make_report.py
from pathlib import Path
import io
import numpy as np
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

DB_PATH = ROOT / "warehouse.duckdb"
HOURLY_EXPORT = ROOT / "exports" / "velib_hourly.parquet"
HISTORY_MD = DOCS / "history.md"
KPI_PARTIAL = DOCS / "partials" / "kpi_results.md"

TZ = "Europe/Paris"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _paris(dt):
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    return dt.tz_convert(TZ)

def _station_names(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = """
    WITH ranked AS (
      SELECT
        stationcode, name, ts_utc,
        ROW_NUMBER() OVER (PARTITION BY stationcode ORDER BY ts_utc DESC) rn
      FROM velib_snapshots
      WHERE name IS NOT NULL
    )
    SELECT stationcode, ANY_VALUE(name) AS name
    FROM ranked
    WHERE rn <= 3
    GROUP BY 1;
    """
    df = con.execute(q).fetchdf()
    df["stationcode"] = df["stationcode"].astype(str)
    df["name"] = df["name"].astype(str)
    return df.drop_duplicates(subset=["stationcode"])

# -----------------------------------------------------------------------------
# Latest snapshot for KPI
# -----------------------------------------------------------------------------
def _read_latest_snapshot(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = """
    WITH ranked AS (
      SELECT
        stationcode, name, lat, lon, capacity,
        ts_utc, numbikesavailable AS nb_velos, numdocksavailable AS nb_bornes,
        ROW_NUMBER() OVER (PARTITION BY stationcode ORDER BY ts_utc DESC) AS rn
      FROM velib_snapshots
    )
    SELECT stationcode, name, lat, lon, capacity, ts_utc, nb_velos, nb_bornes
    FROM ranked
    WHERE rn = 1 AND lat IS NOT NULL AND lon IS NOT NULL;
    """
    df = con.execute(q).fetchdf()
    if df.empty:
        return df
    # normalize
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["nb_velos"] = pd.to_numeric(df["nb_velos"], errors="coerce").fillna(0)
    df["nb_bornes"] = pd.to_numeric(df["nb_bornes"], errors="coerce").fillna(0)
    return df

def _kpis_from_latest(df_latest: pd.DataFrame) -> dict:
    if df_latest.empty:
        return {
            "stations": 0,
            "bikes_total": 0,
            "docks_total": 0,
            "occ_mean_pct": 0.0,
            "ts_latest": None,
        }
    df = df_latest.copy()
    df["occ_ratio"] = (df["nb_velos"] / df["capacity"]).where(df["capacity"] > 0, np.nan)
    stations = df["stationcode"].nunique()
    bikes_total = int(df["nb_velos"].sum())
    docks_total = int(df["nb_bornes"].sum())
    occ_mean = float(pd.to_numeric(df["occ_ratio"], errors="coerce").dropna().mean() or 0.0)
    ts_latest = pd.to_datetime(df["ts_utc"], errors="coerce").max()
    return {
        "stations": stations,
        "bikes_total": bikes_total,
        "docks_total": docks_total,
        "occ_mean_pct": round(100 * occ_mean, 1),
        "ts_latest": ts_latest,
    }

def _write_kpi_partial(k: dict):
    KPI_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    ts = "-" if k["ts_latest"] is None else _paris(pd.Series([k["ts_latest"]]).iloc[0]).strftime("%d/%m %H:%M")
    buf = io.StringIO()
    buf.write(f"**Dernier snapshot** : `{ts}` (Europe/Paris)\n\n")
    buf.write("**KPI (instantané)**\n\n")
    buf.write(f"- Stations couvertes : **{k['stations']}**\n")
    buf.write(f"- Vélos disponibles (total) : **{k['bikes_total']}**\n")
    buf.write(f"- Bornes libres (total) : **{k['docks_total']}**\n")
    buf.write(f"- Taux moyen d’occupation : **{k['occ_mean_pct']} %**\n")
    KPI_PARTIAL.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[kpi] OK -> {KPI_PARTIAL}")

# -----------------------------------------------------------------------------
# Hourly time series (72h)
# -----------------------------------------------------------------------------
def _read_hourly_timeseries(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    if HOURLY_EXPORT.exists():
        df = pd.read_parquet(HOURLY_EXPORT)
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=72)
        df = df[pd.to_datetime(df["hour_utc"]) >= cutoff].copy()
        return df

    q = """
    WITH base AS (
      SELECT
        ts_utc::TIMESTAMP AS ts_utc, stationcode,
        COALESCE(numbikesavailable,0) AS bikes,
        COALESCE(numdocksavailable,0) AS docks,
        NULLIF(capacity,0) AS capacity
      FROM velib_snapshots
      WHERE ts_utc >= now() - INTERVAL 72 HOUR
    ),
    enriched AS (
      SELECT * ,
        CASE WHEN capacity IS NOT NULL THEN bikes / capacity
             WHEN (bikes + docks) > 0 THEN bikes::DOUBLE / (bikes + docks)
             ELSE NULL END AS occ_ratio
      FROM base
    )
    SELECT
      date_trunc('hour', ts_utc) AS hour_utc,
      stationcode,
      CAST(avg(bikes) AS INTEGER)       AS nb_velos_hour,
      CAST(avg(docks) AS INTEGER)       AS nb_bornes_hour,
      max(capacity)                     AS capacity_hour,
      CASE WHEN max(capacity) > 0 THEN avg(bikes)::DOUBLE / max(capacity)
           WHEN (avg(bikes)+avg(docks))>0 THEN avg(bikes)::DOUBLE/(avg(bikes)+avg(docks))
           ELSE NULL END                AS occ_ratio_hour
    FROM enriched
    GROUP BY 1,2
    ORDER BY 1,2;
    """
    return con.execute(q).fetchdf()

def _plot_history(df_hourly: pd.DataFrame, out_path: Path):
    if df_hourly.empty:
        return False
    df = df_hourly.copy()
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce")
    s = df.groupby("hour_utc")["occ_ratio_hour"].mean().sort_index()
    s_roll = s.rolling(3, min_periods=1).mean()

    plt.figure(figsize=(10, 4.5))
    plt.plot(s.index.tz_convert(TZ), s.values, label="Moyenne horaire (occ)")
    plt.plot(s_roll.index.tz_convert(TZ), s_roll.values, linestyle="--", label="Lissé (3h)")
    plt.axhline(0.2, linestyle=":", label="Seuil 20%")
    plt.axhline(0.8, linestyle=":", label="Seuil 80%")
    plt.title("Occupation moyenne du réseau — ~72h")
    plt.xlabel("Heure (Europe/Paris)")
    plt.ylabel("Taux d’occupation (0–1)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True

def _plot_bikes_total(df_hourly: pd.DataFrame, out_path: Path):
    if df_hourly.empty:
        return False
    df = df_hourly.copy()
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce")
    s = df.groupby("hour_utc")["nb_velos_hour"].sum().sort_index()
    plt.figure(figsize=(10, 4.5))
    plt.plot(s.index.tz_convert(TZ), s.values)
    plt.title("Vélos disponibles — total réseau (horaire)")
    plt.xlabel("Heure (Europe/Paris)")
    plt.ylabel("Vélos")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True

# -----------------------------------------------------------------------------
# (Optionnel) Charger des prédictions si elles existent, mais ne pas planter si absentes
# -----------------------------------------------------------------------------
def _load_predictions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame | None:
    preds_parq = ROOT / "exports" / "velib_predictions.parquet"
    if not preds_parq.exists():
        return None

    df = pd.read_parquet(preds_parq)
    if df is None or len(df) == 0:
        return None

    # Harmonisation
    ren = {"pred_nb_velos": "y_nb_pred", "nb_pred": "y_nb_pred", "nb_velos_hour": "y_nb_obs"}
    for a, b in ren.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]

    # Types & base
    if "stationcode" in df.columns:
        df["stationcode"] = df["stationcode"].astype(str)
    if "hour_utc" not in df.columns:
        # impossible de tracer / agréger proprement sans timestamp
        return None

    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["hour_utc"]).reset_index(drop=True)

    # Calcul occ_ratio_pred (safe)
    pred_series = df["y_nb_pred"] if "y_nb_pred" in df.columns else None
    cap_col = "capacity_hour" if "capacity_hour" in df.columns else ("capacity" if "capacity" in df.columns else None)

    if pred_series is not None and cap_col is not None:
        pred = pd.to_numeric(pred_series, errors="coerce").to_numpy()
        cap = pd.to_numeric(df[cap_col], errors="coerce").to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            occ = np.where((cap > 0) & np.isfinite(pred), pred / cap, np.nan)
        df["occ_ratio_pred"] = pd.Series(occ)
    else:
        # colonne absente -> ne rien casser
        if "occ_ratio_pred" not in df.columns:
            df["occ_ratio_pred"] = np.nan

    # Clean obs/pred en numérique si présents
    for c in ["y_nb_obs", "y_nb_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# -----------------------------------------------------------------------------
# Rédaction du rapport "History"
# -----------------------------------------------------------------------------
def _write_history_md(kpis: dict, fig_occ: str, fig_bikes: str):
    CONTENT = f"""# Historique & KPI

**Dernier snapshot** : `{
    "-" if kpis['ts_latest'] is None else _paris(pd.Series([kpis['ts_latest']]).iloc[0]).strftime("%d/%m %H:%M")
}` (Europe/Paris)

**KPI (instantané)**

- Stations couvertes : **{kpis['stations']}**
- Vélos disponibles (total) : **{kpis['bikes_total']}**
- Bornes libres (total) : **{kpis['docks_total']}**
- Taux moyen d’occupation : **{kpis['occ_mean_pct']} %**

## Tendance d’occupation

![Mean occupancy](assets/figs/{fig_occ})

## Vélos disponibles — total réseau

![Bikes total](assets/figs/{fig_bikes})
"""
    HISTORY_MD.write_text(CONTENT, encoding="utf-8")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if not DB_PATH.exists():
        raise SystemExit("warehouse.duckdb introuvable. Lance d'abord l'ingestion.")

    con = duckdb.connect(str(DB_PATH))

    # 1) KPI snapshot
    latest = _read_latest_snapshot(con)
    kpis = _kpis_from_latest(latest)
    _write_kpi_partial(kpis)

    # 2) Time series (hourly)
    hourly = _read_hourly_timeseries(con)
    if hourly is None:
        hourly = pd.DataFrame()

    # 3) Plots
    fig_occ = "occupancy_last72h.png"
    fig_bikes = "bikes_total_last72h.png"
    _plot_history(hourly, FIGS / fig_occ)
    _plot_bikes_total(hourly, FIGS / fig_bikes)

    # 4) (Optionnel) Charger les prédictions si dispo (mais ne rien imposer)
    try:
        _ = _load_predictions(con)  # si besoin, on pourrait écrire des visuels ici
    except Exception as e:
        print(f"[report] skip predictions: {e}")

    # 5) Write markdown (history.md)
    _write_history_md(kpis, fig_occ, fig_bikes)

    print("[report] OK → docs/history.md + docs/assets/figs/")

if __name__ == "__main__":
    main()
