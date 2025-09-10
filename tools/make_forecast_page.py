# tools/make_forecast_page.py
from pathlib import Path
import io
import numpy as np
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DB   = ROOT / "warehouse.duckdb"
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT_MD = DOCS / "forecast.md"

TZ = "Europe/Paris"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _paris(dt):
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    return dt.dt.tz_convert(TZ).dt.strftime("%d/%m %Hh")

def _station_names(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """DataFrame stationcode -> name (dernier nom connu)."""
    q = """
    WITH ranked AS (
      SELECT
        stationcode,
        name,
        ts_utc,
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

def _nice_title(row):
    name = row.get("name")
    sc   = row.get("stationcode")
    if pd.isna(name) or not str(name).strip():
        return f"Station {sc}"
    return f"{name} ({sc})"

def _plot_obs_pred(df, stationcode, title, out_png):
    d = df[df["stationcode"] == str(stationcode)].copy()
    if d.empty:
        return False
    d["hour_utc"] = pd.to_datetime(d["hour_utc"], utc=True, errors="coerce")
    d = d.sort_values("hour_utc")
    x = d["hour_utc"].dt.tz_convert(TZ)

    plt.figure(figsize=(8.6, 3.6))
    if "y_nb_obs" in d and d["y_nb_obs"].notna().any():
        plt.plot(x, d["y_nb_obs"], label="observé (nb vélos)")
    elif "nb_velos_hour" in d and d["nb_velos_hour"].notna().any():
        plt.plot(x, d["nb_velos_hour"], label="observé (nb vélos)")

    if "y_nb_pred" in d and d["y_nb_pred"].notna().any():
        plt.plot(x, d["y_nb_pred"], linestyle="--", label="prédit T+1h")

    plt.title(title)
    plt.xlabel("Heure locale (Europe/Paris)")
    plt.ylabel("Vélos disponibles")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    return True

# ----------------------------------------------------------------------
# Entrée : prédictions (ou baseline si pas de fichier de prédictions)
# ----------------------------------------------------------------------
def _load_predictions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Sortie attendue :
      stationcode (str), hour_utc (UTC), y_nb_pred, y_nb_obs (optionnel), occ_ratio_pred (calculé ici)
    """
    preds_parq = ROOT / "exports" / "velib_predictions.parquet"

    def _dedup_and_coalesce_pred(df: pd.DataFrame) -> pd.DataFrame:
        # 1) Supprimer colonnes strictement dupliquées (même nom au sens pandas)
        if df.columns.duplicated().any():
            # Essaie de coalescer d'abord les doublons de y_nb_pred
            ycols = [c for c in df.columns if c == "y_nb_pred" or c.startswith("y_nb_pred.")]
            if len(ycols) > 1:
                df["y_nb_pred"] = pd.to_numeric(df[ycols].bfill(axis=1).iloc[:, 0], errors="coerce")
                # drop toutes les autres colonnes y_nb_pred.* sauf la principale
                keep = [c for c in df.columns if c not in ycols or c == "y_nb_pred"]
                df = df[keep]
            # Ensuite, drop tout doublon de nom restant
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # Si y_nb_pred n'existe toujours pas, crée-la
        if "y_nb_pred" not in df.columns:
            df["y_nb_pred"] = np.nan

        # Forcer en numérique (1D)
        df["y_nb_pred"] = pd.to_numeric(df["y_nb_pred"], errors="coerce")
        return df

    if preds_parq.exists():
        df = pd.read_parquet(preds_parq)

        # Harmonisation colonnes éventuelles
        ren = {"pred_nb_velos": "y_nb_pred", "nb_pred": "y_nb_pred", "nb_velos_hour": "y_nb_obs"}
        for a, b in ren.items():
            if a in df.columns and b not in df.columns:
                df[b] = df[a]

        # Types & nettoyage de base
        if "stationcode" in df.columns:
            df["stationcode"] = df["stationcode"].astype(str)
        df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["stationcode", "hour_utc"]).reset_index(drop=True)

        # Dédup & coalesce
        df = _dedup_and_coalesce_pred(df)

        # Capacité (si dispo) → calcul positionnel
        cap_col = "capacity_hour" if "capacity_hour" in df.columns else ("capacity" if "capacity" in df.columns else None)
        if cap_col is not None:
            cap = pd.to_numeric(df[cap_col], errors="coerce").to_numpy()
            pred = df["y_nb_pred"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                occ = np.where((cap > 0) & np.isfinite(pred), pred / cap, np.nan)
            df["occ_ratio_pred"] = pd.Series(occ)
        else:
            df["occ_ratio_pred"] = np.nan

        # Nettoyage obs/pred
        for c in ["y_nb_obs", "y_nb_pred"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # ---------- Baseline si pas de fichier de prédictions ----------
    hourly = ROOT / "exports" / "velib_hourly.parquet"
    if hourly.exists():
        df = pd.read_parquet(hourly)
    else:
        q = """
        WITH base AS (
          SELECT
            date_trunc('hour', ts_utc) AS hour_utc,
            stationcode,
            avg(numbikesavailable) AS nb_velos_hour
          FROM velib_snapshots
          WHERE ts_utc >= now() - INTERVAL 72 HOUR
          GROUP BY 1,2
        )
        SELECT * FROM base;
        """
        df = con.execute(q).fetchdf()

    df = df.rename(columns={"bikes_avg": "nb_velos_hour"}).copy()
    df["stationcode"] = df["stationcode"].astype(str)
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["stationcode", "hour_utc"]).reset_index(drop=True)

    # prédiction naïve T+1h = valeur observée actuelle
    df_pred = df[["stationcode", "hour_utc", "nb_velos_hour"]].copy()
    df_pred["hour_utc"] = df_pred["hour_utc"] + pd.Timedelta(hours=1)
    df_pred = df_pred.rename(columns={"nb_velos_hour": "y_nb_pred"})

    out = df.merge(df_pred, on=["stationcode", "hour_utc"], how="left").reset_index(drop=True)
    out = out.rename(columns={"nb_velos_hour": "y_nb_obs"})

    # Dédup & coalesce (au cas où)
    out = _dedup_and_coalesce_pred(out)

    # occ_ratio_pred positionnel
    cap_col = "capacity_hour" if "capacity_hour" in out.columns else ("capacity" if "capacity" in out.columns else None)
    if cap_col is not None:
        cap = pd.to_numeric(out[cap_col], errors="coerce").to_numpy()
        pred = out["y_nb_pred"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            occ = np.where((cap > 0) & np.isfinite(pred), pred / cap, np.nan)
        out["occ_ratio_pred"] = pd.Series(occ)
    else:
        out["occ_ratio_pred"] = np.nan

    out["y_nb_obs"]  = pd.to_numeric(out.get("y_nb_obs"),  errors="coerce")
    out["y_nb_pred"] = pd.to_numeric(out.get("y_nb_pred"), errors="coerce")

    return out

# ----------------------------------------------------------------------
# Génération de la page
# ----------------------------------------------------------------------
def main():
    if not DB.exists():
        raise SystemExit("warehouse.duckdb introuvable. Lance d’abord l’ingestion.")

    con = duckdb.connect(str(DB))

    # 1) données
    df = _load_predictions(con)
    names_df = _station_names(con)  # stationcode -> name

    # fusion noms
    df["stationcode"] = df["stationcode"].astype(str)
    df = df.merge(names_df, on="stationcode", how="left", suffixes=("", "_nm"))

    # coalesce vers une seule colonne d'affichage
    name_cols = [c for c in df.columns if c.lower().startswith("name")]
    if name_cols:
        df["name_display"] = df[name_cols].bfill(axis=1).iloc[:, 0]
    else:
        df["name_display"] = np.nan

    # 2) fenêtre 24h pour lisibilité
    # APRÈS (robuste)
    utc_now = pd.Timestamp.now(tz="UTC")
    cutoff = utc_now - pd.Timedelta(hours=24)
    df = df[df["hour_utc"] >= cutoff].copy()

    if df.empty:
        OUT_MD.write_text("# Prévisions\n\n_Aucune donnée disponible._\n", encoding="utf-8")
        print(f"[forecast page] OK → {OUT_MD} (vide)")
        return

    # 3) snapshot dernière heure pour tops
    last_hour = df["hour_utc"].max()
    snap = df[df["hour_utc"] == last_hour].copy()

    # clean types
    for c in ["y_nb_pred", "y_nb_obs", "occ_ratio_pred"]:
        if c in snap.columns:
            snap[c] = pd.to_numeric(snap[c], errors="coerce")
    snap["when_local"] = _paris(snap["hour_utc"])

    # Top-10 faible dispo & saturation
    top_low = snap.sort_values("y_nb_pred", ascending=True).head(10).copy()
    top_sat = snap.sort_values("occ_ratio_pred", ascending=False).head(10).copy()

    def table_md(d):
        d2 = d.copy()
        d2["Station"] = d2.apply(
            lambda r: f"{(r['name'] if pd.notna(r['name']) and str(r['name']).strip() else '—')} (`{r['stationcode']}`)",
            axis=1
        )
        d2["Prédit T+1h (vélos)"] = d2["y_nb_pred"].round(0).astype("Int64") if "y_nb_pred" in d2 else pd.NA
        d2["Taux prévu"] = d2["occ_ratio_pred"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—") if "occ_ratio_pred" in d2 else "—"
        d2["Dernière obs."] = d2["when_local"]
        return d2[["Station","Prédit T+1h (vélos)","Taux prévu","Dernière obs."]].to_markdown(index=False)

    # 4) Rédaction Markdown
    buf = io.StringIO()
    buf.write("# Prévisions\n\n")
    buf.write(f"*Dernière heure considérée : **{_paris(pd.Series([last_hour]))[0]}** (Europe/Paris)*\n\n")

    buf.write("## Top-10 stations à risque (faible nb vélos prévu T+1h)\n\n")
    buf.write(table_md(top_low) + "\n\n")

    buf.write("## Top-10 risque de saturation (taux prévu élevé)\n\n")
    buf.write(table_md(top_sat) + "\n\n")

    # 5) Graphes repliables (sous-ensemble raisonnable)
    show_codes = pd.unique(pd.concat([top_low["stationcode"], top_sat["stationcode"]])).tolist()[:10]
    buf.write("## Détails par station (graphiques)\n\n")
    for sc in show_codes:
        sub = df[df["stationcode"] == str(sc)]
        if sub.empty:
            continue
        title = _nice_title(sub.iloc[-1])  # utilise name + code
        png = FIGS / f"obs_pred_{sc}_T+1h_compact.png"
        ok = _plot_obs_pred(df, str(sc), f"{title} — Observé vs Prédit", png)
        if not ok:
            continue
        buf.write(f"<details>\n<summary><strong>{title}</strong></summary>\n\n")
        buf.write(f"![{title}](assets/figs/{png.name})\n\n")
        buf.write("</details>\n\n")

    OUT_MD.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[forecast page] OK → {OUT_MD}")

if __name__ == "__main__":
    main()
