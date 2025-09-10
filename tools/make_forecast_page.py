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
def _paris_fmt(dt) -> pd.Series:
    """Retourne une Series de str 'dd/mm HHh' en Europe/Paris depuis timestamps UTC."""
    s = pd.to_datetime(dt, utc=True, errors="coerce")
    return s.dt.tz_convert(TZ).dt.strftime("%d/%m %Hh")

def _station_names(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """DataFrame stationcode -> name (dernier nom connu, robuste)."""
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
    if df.empty:
        return pd.DataFrame({"stationcode": [], "name": []})
    df["stationcode"] = df["stationcode"].astype(str)
    df["name"] = df["name"].astype("string[python]").str.strip()
    df = df.drop_duplicates(subset=["stationcode"])
    return df

def _nice_title(row):
    name = row.get("name_display") or row.get("name")
    sc   = row.get("stationcode")
    if name is None or not str(name).strip():
        return f"Station {sc}"
    return f"{name} ({sc})"

def _plot_obs_pred(df, stationcode, title, out_png):
    d = df[df["stationcode"] == str(stationcode)].copy()
    if d.empty:
        return False
    d["hour_utc"] = pd.to_datetime(d["hour_utc"], utc=True, errors="coerce")
    d = d.dropna(subset=["hour_utc"]).sort_values("hour_utc")
    x = d["hour_utc"].dt.tz_convert(TZ)

    plt.figure(figsize=(8.6, 3.6))
    # Observé
    if "y_nb_obs" in d and d["y_nb_obs"].notna().any():
        plt.plot(x, d["y_nb_obs"], label="observé (nb vélos)")
    elif "nb_velos_hour" in d and d["nb_velos_hour"].notna().any():
        plt.plot(x, d["nb_velos_hour"], label="observé (nb vélos)")
    # Prédit
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
# Prédictions (ou baseline si pas de fichier de prédictions)
# ----------------------------------------------------------------------
def _load_predictions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Sortie attendue :
      stationcode (str), hour_utc (UTC), y_nb_pred, y_nb_obs (optionnel), occ_ratio_pred (calculé ici)
    """
    preds_parq = ROOT / "exports" / "velib_predictions.parquet"

    def _dedup_and_coalesce_pred(df: pd.DataFrame) -> pd.DataFrame:
        # Si colonnes dupliquées (par nom), commence par coalescer y_nb_pred.* si nécessaire
        ycols = [c for c in df.columns if c == "y_nb_pred" or str(c).startswith("y_nb_pred.")]
        if len(ycols) > 1:
            block = df[ycols].apply(pd.to_numeric, errors="coerce")
            df["y_nb_pred"] = block.bfill(axis=1).iloc[:, 0]
            keep = [c for c in df.columns if (c not in ycols) or (c == "y_nb_pred")]
            df = df[keep]
        # Drop tout doublon résiduel
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
        # Garantir existence + numeric 1D
        if "y_nb_pred" not in df.columns:
            df["y_nb_pred"] = np.nan
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
    if df is None or df.empty:
        OUT_MD.write_text("# Prévisions\n\n_Aucune donnée disponible._\n", encoding="utf-8")
        print(f"[forecast page] OK → {OUT_MD} (vide)")
        return

    names_df = _station_names(con)  # stationcode -> name

    # fusion noms
    df["stationcode"] = df["stationcode"].astype(str)
    df = df.merge(names_df, on="stationcode", how="left", suffixes=("", "_nm"))

    # coalesce vers une seule colonne d'affichage, sans FutureWarning
    name_cols = [c for c in df.columns if str(c).lower().startswith("name")]
    if name_cols:
        name_block = df[name_cols].astype("string[python]").bfill(axis=1)
        df["name_display"] = name_block.iloc[:, 0].astype(str).str.strip()
    else:
        df["name_display"] = "—"

    # 2) fenêtre 24h pour lisibilité
    utc_now = pd.Timestamp.now(tz="UTC")
    cutoff = utc_now - pd.Timedelta(hours=24)
    df = df[df["hour_utc"] >= cutoff].copy()

    if df.empty:
        OUT_MD.write_text("# Prévisions\n\n_Aucune donnée disponible sur 24h._\n", encoding="utf-8")
        print(f"[forecast page] OK → {OUT_MD} (vide)")
        return

    # 3) snapshot dernière heure pour tops
    last_hour = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce").max()
    snap = df[df["hour_utc"] == last_hour].copy()

    # clean types
    for c in ["y_nb_pred", "y_nb_obs", "occ_ratio_pred"]:
        if c in snap.columns:
            snap[c] = pd.to_numeric(snap[c], errors="coerce")
    snap["when_local"] = _paris_fmt(snap["hour_utc"])

    # Top-10 faible dispo & saturation
    top_low = snap.sort_values("y_nb_pred", ascending=True, na_position="last").head(10).copy()
    top_sat = snap.sort_values("occ_ratio_pred", ascending=False, na_position="last").head(10).copy()

    # -------- tables Markdown (safe pour tabulate) ----------
    def table_md(d):
        d2 = d.copy()
        # garantir name_display même si d est un sous-ensemble
        if "name_display" not in d2.columns:
            name_cols = [c for c in d2.columns if str(c).lower().startswith("name")]
            if name_cols:
                name_block = d2[name_cols].astype("string[python]").bfill(axis=1)
                d2["name_display"] = name_block.iloc[:, 0].astype(str).str.strip()
            else:
                d2["name_display"] = "—"

        # libellé Station
        def _label(r):
            nm = r.get("name_display")
            txt = str(nm).strip() if (nm is not None and pd.notna(nm)) else "—"
            return f"{txt} (`{r['stationcode']}`)"

        d2["Station"] = d2.apply(_label, axis=1)

        # colonnes affichées
        if "y_nb_pred" in d2.columns:
            d2["Prédit T+1h (vélos)"] = pd.to_numeric(d2["y_nb_pred"], errors="coerce").round(0).astype("Int64")
        else:
            d2["Prédit T+1h (vélos)"] = pd.Series([pd.NA] * len(d2), dtype="Int64")

        if "occ_ratio_pred" in d2.columns:
            d2["Taux prévu"] = d2["occ_ratio_pred"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        else:
            d2["Taux prévu"] = "—"

        d2["Dernière obs."] = d2["when_local"] if "when_local" in d2.columns else "—"

        # Eviter pd.NA dans to_markdown/tabulate -> tout convertir en string propre
        view = d2[["Station", "Prédit T+1h (vélos)", "Taux prévu", "Dernière obs."]].copy()
        for c in view.columns:
            view[c] = view[c].astype("string[python]").fillna("—")

        return view.to_markdown(index=False)

    # 4) Rédaction Markdown
    buf = io.StringIO()
    buf.write("# Prévisions\n\n")
    buf.write(f"*Dernière heure considérée : **{_paris_fmt(pd.Series([last_hour]))[0]}** (Europe/Paris)*\n\n")

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
        # Prend la dernière ligne connue pour le titre (avec name_display)
        last_row = sub.iloc[-1]
        title = _nice_title(last_row)
        png = FIGS / f"obs_pred_{sc}_T+1h_compact.png"
        ok = _plot_obs_pred(df, str(sc), f"{title} — Observé vs Prédit", png)
        if not ok:
            continue
        buf.write(f'???+ info "{title}"\n\n')
        buf.write(f"    ![{title}](assets/figs/{png.name})\n\n")

    OUT_MD.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[forecast page] OK → {OUT_MD}")

if __name__ == "__main__":
    main()
