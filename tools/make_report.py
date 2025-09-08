from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Dossiers
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
EXPORTS = DOCS / "exports"
FIGS.mkdir(parents=True, exist_ok=True)
EXPORTS.mkdir(parents=True, exist_ok=True)

# --- Helpers ---
def save_fig(path: Path) -> None:
    """Enregistre une figure matplotlib avec style propre"""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def ensure_export_files() -> None:
    """Créer des fichiers CSV de base pour éviter les erreurs mkdocs --strict"""
    f1 = EXPORTS / "occ_hourly_sample.csv"
    f2 = EXPORTS / "velib_forecast_24h.csv"

    if not f1.exists():
        f1.write_text(
            "stationcode,timestamp,occ_ratio\n"
            "10001,2025-09-08T07:00:00,0.42\n"
            "10001,2025-09-08T08:00:00,0.55\n",
            encoding="utf-8"
        )
    if not f2.exists():
        f2.write_text(
            "stationcode,timestamp,forecast_occ_ratio\n"
            "10001,2025-09-09T07:00:00,0.48\n"
            "10001,2025-09-09T08:00:00,0.51\n",
            encoding="utf-8"
        )

# --- Génération Home (index.md) ---
def make_index_page(kpis: dict) -> None:
    out = DOCS / "index.md"
    with out.open("w", encoding="utf-8") as md:
        md.write("# Vélib’ Paris — Batch Forecast\n\n")
        md.write("> Prédictions d’occupation des stations à l’heure (pipeline batch). Données temps réel Paris Data, features calendaires, baseline LightGBM.\n\n")

        # KPIs
        md.write('<div class="kpis">\n')
        md.write(f'  <div class="kpi"><div class="label">Snapshots</div><div class="value">{kpis.get("snapshots","…")}</div></div>\n')
        md.write(f'  <div class="kpi"><div class="label">Stations</div><div class="value">{kpis.get("stations","…")}</div></div>\n')
        md.write(f'  <div class="kpi"><div class="label">Dernière maj (Paris)</div><div class="value">{kpis.get("last_paris","…")}</div></div>\n')
        md.write('</div>\n\n')

        # Boutons
        md.write("[:material-chart-line: Résultats](results.md){ .md-button }\n")
        md.write("[:material-heart-pulse: Monitoring](monitoring.md){ .md-button .md-button--secondary }\n\n")

        # Exports
        md.write("!!! tip \"Exports\"\n")
        md.write("    - [Prévision 24h (CSV)](exports/velib_forecast_24h.csv){ target=_blank }\n")
        md.write("    - [Occupations horaires (sample CSV)](exports/occ_hourly_sample.csv){ target=_blank }\n\n")

        # Stack rapide
        md.write("**Stack rapide**\n")
        md.write("- Ingestion snapshots → agrégation horaire  \n")
        md.write("- Features : calendaires (+ météo)  \n")
        md.write("- Modèle : LightGBM baseline → 24 h rolling forecast\n")

# --- Génération Results (results.md) ---
def make_results_page(kpis: dict, df_vol: pd.DataFrame) -> None:
    out = DOCS / "results.md"
    with out.open("w", encoding="utf-8") as md:
        md.write("# Results\n\n")

        # Exemple forecast
        md.write("## Exemple (historique + forecast 24h)\n")
        md.write('<div class="figure">\n')
        md.write('  <img src="assets/figs/hist_forecast_24h.png" alt="Historique + prévision 24h">\n')
        md.write('  <div class="caption">Historique agrégé + horizon 24h (échantillon de stations).</div>\n')
        md.write('</div>\n\n')

        # Corrélation simple
        md.write("## Corrélation simple\n")
        md.write('<div class="figure">\n')
        md.write('  <img src="assets/figs/occ_vs_temp.png" alt="Occupation vs Température">\n')
        md.write('  <div class="caption">Relation occupation vs température (échantillon horaire).</div>\n')
        md.write('</div>\n\n')

        # Top stations
        md.write("## Top 10 stations les plus volatiles\n\n")
        if df_vol is not None and not df_vol.empty:
            md.write(df_vol.head(10).to_markdown(index=False) + "\n\n")
        else:
            md.write("> Données indisponibles pour le moment.\n\n")

        # Carte
        md.write("## Carte (dernier snapshot)\n\n")
        md.write('<iframe src="assets/map.html" width="100%" height="520" style="border:none;"></iframe>\n\n')

        # Exports
        md.write("## Exports\n")
        md.write("- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv){ target=_blank }\n")
        md.write("- [Occupations horaires (échantillon CSV)](exports/occ_hourly_sample.csv){ target=_blank }\n")

# --- Génération Monitoring (monitoring.md) ---
def make_monitoring_page(df_psi: pd.DataFrame, df_back: pd.DataFrame) -> None:
    out = DOCS / "monitoring.md"
    with out.open("w", encoding="utf-8") as md:
        md.write("# Monitoring\n\n")

        # Drift
        md.write("## Drift (7 jours vs. 30 jours)\n\n")
        show_psi = df_psi is not None and not df_psi.empty and df_psi["n_base"].fillna(0).sum() > 0
        if show_psi:
            md.write(df_psi.to_markdown(index=False) + "\n\n")
            md.write("> PSI: `<0.10` OK • `0.10–0.25` Attention • `>0.25` Alerte\n\n")
        else:
            md.write("> Baseline insuffisante — en attente de données de référence.\n\n")

        # Backtest
        md.write("## Performance (Backtest 24h)\n\n")
        if df_back is not None and not df_back.empty:
            md.write(df_back.to_markdown(index=False) + "\n")
        else:
            md.write("> Backtest indisponible pour le moment.\n")

# --- Main ---
if __name__ == "__main__":
    ensure_export_files()   # <-- pour éviter les erreurs mkdocs --strict

    # KPIs d’exemple (à remplacer par tes calculs réels)
    kpis = {"snapshots": 87145, "stations": 1453, "last_paris": "2025-09-08 19:10"}

    # Données exemples
    df_vol = pd.DataFrame({
        "stationcode":[21021,15056],
        "name":["Enfants du Paradis - Peupliers","Place Balard"],
        "std_occ":[0.501,0.460]
    })
    df_psi = pd.DataFrame([
        {"feature":"occ_ratio_hour","psi":0.07,"base_mean":0.38,"curr_mean":0.29,
         "base_std":0.09,"curr_std":0.08,"n_base":20000,"n_curr":22000,"psi_flag":"OK"}
    ])
    df_back = pd.DataFrame([
        {"metric":"MAE","value":2.41},
        {"metric":"RMSE","value":3.72},
        {"metric":"sMAPE","value":"14.2%"}
    ])

    # Génération des pages
    make_index_page(kpis)
    make_results_page(kpis, df_vol)
    make_monitoring_page(df_psi, df_back)
