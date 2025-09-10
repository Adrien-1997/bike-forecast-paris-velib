# tools/make_forecast_page.py
from pathlib import Path
import io
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

# ---- Helpers ---------------------------------------------------------------
def _paris(dt):
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    return dt.dt.tz_convert(TZ).dt.strftime("%d/%m %Hh")

def _station_lookup(con):
    # Dernier nom connu par station
    q = """
    WITH r AS (
      SELECT stationcode, name, ts_utc,
             ROW_NUMBER() OVER (PARTITION BY stationcode ORDER BY ts_utc DESC) rn
      FROM velib_snapshots
      WHERE name IS NOT NULL
    )
    SELECT stationcode, any_value(name) AS name
    FROM r WHERE rn<=3
    GROUP BY 1;
    """
    return con.execute(q).fetchdf().set_index("stationcode")["name"].to_dict()

def _nice_title(stationcode, names):
    name = names.get(str(stationcode)) or names.get(int(stationcode)) if isinstance(stationcode, str) and stationcode.isdigit() else names.get(stationcode)
    if not name:
        return f"Station {stationcode}"
    return f"{name} ({stationcode})"

def _plot_obs_pred(df, stationcode, title, out_png):
    d = df[df["stationcode"]==stationcode].copy()
    if d.empty:
        return False
    d["hour_utc"] = pd.to_datetime(d["hour_utc"], utc=True, errors="coerce")
    d.sort_values("hour_utc", inplace=True)
    # x en heure locale
    x = d["hour_utc"].dt.tz_convert(TZ)
    plt.figure(figsize=(8, 3.4))
    if "y_nb_obs" in d:
        plt.plot(x, d["y_nb_obs"], label="observé (nb_velos_hour)")
    elif "nb_velos_hour" in d:
        plt.plot(x, d["nb_velos_hour"], label="observé (nb_velos_hour)")
    if "y_nb_pred" in d:
        plt.plot(x, d["y_nb_pred"], linestyle="--", label="prédit T+1h")
    plt.title(title)
    plt.xlabel("Heure locale (Europe/Paris)")
    plt.ylabel("nb vélos")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    return True

# ---- Entrée : predictions ---------------------------------------------------
def _load_predictions(con):
    """
    On cherche d'abord une table/CSV de prédictions, sinon on fabrique un
    miniframe basique à partir de l'agrégat horaire (baseline naïve).
    Le but est d'avoir des colonnes standardisées :
      stationcode, hour_utc, y_nb_pred, y_nb_obs, occ_ratio_pred
    """
    # 1) parquet de prédictions (si tu en écris un)
    preds_parq = ROOT / "exports" / "velib_predictions.parquet"
    if preds_parq.exists():
        df = pd.read_parquet(preds_parq)
        # harmonise les noms si besoin
        colmap = {
            "pred_nb_velos": "y_nb_pred",
            "nb_pred": "y_nb_pred",
            "nb_velos_hour": "y_nb_obs",
        }
        for a,b in colmap.items():
            if a in df.columns and b not in df.columns:
                df[b] = df[a]
        return df

    # 2) sinon, base sur exports/velib_hourly.parquet (observé) et un
    #    simple modèle "persistence" pour la démo (pred = obs dernier)
    hourly = ROOT / "exports" / "velib_hourly.parquet"
    if hourly.exists():
        df = pd.read_parquet(hourly)
    else:
        # fallback rapide depuis la DB (72h)
        df = con.execute("""
            WITH base AS (
              SELECT date_trunc('hour', ts_utc) AS hour_utc,
                     stationcode,
                     avg(numbikesavailable) AS nb_velos_hour
              FROM velib_snapshots
              WHERE ts_utc >= now() - INTERVAL 72 HOUR
              GROUP BY 1,2
            )
            SELECT * FROM base
        """).fetchdf()

    df = df.rename(columns={"bikes_avg":"nb_velos_hour"})
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["stationcode","hour_utc"])

    # "persistance" : prédire la prochaine heure = valeur actuelle
    df_pred = df.copy()
    df_pred["hour_utc"] = df_pred["hour_utc"] + pd.Timedelta(hours=1)
    df_pred = df_pred.rename(columns={"nb_velos_hour":"y_nb_pred"})

    # merge pour avoir obs vs pred aligné sur la même hour_utc
    out = df.merge(df_pred[["stationcode","hour_utc","y_nb_pred"]],
                   on=["stationcode","hour_utc"], how="left")
    out = out.rename(columns={"nb_velos_hour":"y_nb_obs"})

    # occ prédite approx si capacité connue
    if "capacity_hour" in out:
        out["occ_ratio_pred"] = out["y_nb_pred"] / out["capacity_hour"]
    elif "capacity" in out:
        out["occ_ratio_pred"] = out["y_nb_pred"] / out["capacity"]
    else:
        out["occ_ratio_pred"] = pd.NA

    return out

# ---- Génération -------------------------------------------------------------
def main():
    if not DB.exists():
        raise SystemExit("warehouse.duckdb introuvable. Lance d'abord l’ingestion.")

    con = duckdb.connect(str(DB))
    names = _station_lookup(con)
    df = _load_predictions(con)

    # agrégations pour top listes
    # on garde les dernières heures (24h) pour éviter la sur-longueur
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=24)
    df = df[df["hour_utc"] >= cutoff].copy()

    # dernière heure dispo (pour tri)
    last_hour = df["hour_utc"].max()
    snap = df[df["hour_utc"] == last_hour].copy()
    # valeurs nettoyées
    snap["y_nb_pred"]    = pd.to_numeric(snap["y_nb_pred"], errors="coerce").fillna(0)
    snap["occ_ratio_pred"] = pd.to_numeric(snap["occ_ratio_pred"], errors="coerce")
    if "y_nb_obs" in snap:
        snap["y_nb_obs"] = pd.to_numeric(snap["y_nb_obs"], errors="coerce")

    # enrichit noms + timestamp lisible
    snap["station_name"] = snap["stationcode"].map(lambda s: names.get(str(s)) or names.get(int(s), None) if isinstance(s,str) and s.isdigit() else names.get(s))
    snap["when_local"]   = _paris(snap["hour_utc"])

    # Top-10 faible dispo (tri croissant y_nb_pred)
    top_low = snap.sort_values("y_nb_pred", ascending=True).head(10).copy()
    # Top-10 risque saturation (occ_ratio_pred décroissant)
    top_sat = snap.sort_values("occ_ratio_pred", ascending=False).head(10).copy()

    # Tableaux jolis (colonnes claires)
    def table_md(d):
        d2 = d.copy()
        d2["Station"]      = d2.apply(lambda r: f"{r['station_name'] or '—'} `{r['stationcode']}`", axis=1)
        d2["Prédit T+1h (vélos)"] = d2["y_nb_pred"].round(0).astype(int)
        d2["Taux prévu"]   = (d2["occ_ratio_pred"]*100).round(1).astype("float").map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        d2["Dernière obs."] = d2["when_local"]
        cols = ["Station","Prédit T+1h (vélos)","Taux prévu","Dernière obs."]
        return d2[cols].to_markdown(index=False)

    # Rédaction markdown
    buf = io.StringIO()
    buf.write("# Prévisions\n\n")
    buf.write(f"*Dernière heure considérée : **{_paris(pd.Series([last_hour]))[0]}** (Europe/Paris)*\n\n")

    buf.write("## Top-10 stations à risque (faible nb vélos prévu T+1h)\n\n")
    buf.write(table_md(top_low) + "\n\n")

    buf.write("## Top-10 risque de saturation (taux prévu élevé)\n\n")
    buf.write(table_md(top_sat) + "\n\n")

    # Graphes repliables pour un sous-ensemble court (par défaut 8)
    show_codes = pd.unique(pd.concat([top_low["stationcode"], top_sat["stationcode"]])).tolist()[:8]
    buf.write("## Détails par station (graphiques)\n\n")
    for sc in show_codes:
        title = _nice_title(sc, names)
        png = FIGS / f"obs_pred_{sc}_T+1h_compact.png"
        ok = _plot_obs_pred(df, sc, f"{title} — Observé vs Prédit", png)
        if not ok:
            continue
        # bloc repliable
        buf.write(f"<details>\n<summary><strong>{title}</strong></summary>\n\n")
        buf.write(f"![{title}](assets/figs/{png.name})\n\n")
        buf.write("</details>\n\n")

    OUT_MD.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[forecast] wrote {OUT_MD}")

if __name__ == "__main__":
    main()
