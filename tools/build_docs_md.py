# tools/build_docs_md.py
from __future__ import annotations
import argparse, re
from pathlib import Path
from datetime import datetime
import os

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"

SECTIONS = {
    "network": {
        "pages": [
            ("index.md", "# Réseau — index\n\nCette section couvre :\n- **Aperçu du réseau** (KPIs du jour vs historique)\n- **Stations & profils** (table et clustering)\n- **Dynamiques spatio-temporelles** (heatmaps, saisonnalités)\n\n{LIST:figs/network}\n"),
            
(
  "overview.md",
  """# Aperçu du réseau

Cette page donne un **coup d’œil instantané** à la santé du réseau (snapshot le plus récent) et situe la **journée en cours** par rapport aux semaines précédentes.

!!! tip "Ce que vous verrez"
    - **Carte instantanée** des stations avec indicateurs *pénurie* (0 vélo) / *saturation* (0 place).  
    - **Courbe "Aujourd’hui vs médiane (mêmes jours)"** : part de stations offrant ≥1 vélo, minute par minute.  
    - **KPIs** de disponibilité, couverture et volatilité, avec **tables** d’appui (CSV).

---

## Carte instantanée (pénurie / saturation)
<div style="margin: 0.5rem 0;">
  <iframe src="../assets/maps/network_overview.html"
          style="width:100%;height:520px;border:0" loading="lazy"
          title="Carte instantanée du réseau"></iframe>
</div>

- **Couleurs** : rouge = *pénurie* (0 vélo), noir = *saturation* (0 place), bleu = OK/autre.  
- **Astuce** : zoomez et survolez pour voir le nom, l’ID et les valeurs instantanées.

---

## Courbe « Aujourd’hui vs médiane (mêmes jours) »
{FIGS:network/overview}

> La courbe trace la **part de stations** avec ≥1 vélo à chaque **hh:mm locale** pour **aujourd’hui**, et la compare à la **médiane** des mêmes jours des semaines récentes (fenêtre de référence).  
> Méthode : agrégation par hh:mm, puis médiane inter-jours sur la fenêtre *N* semaines.  
> (Voir la section *Méthodologie* plus bas.)

---

## KPIs & distributions (CSV)
{LIST:tables/network/overview}

Fichiers typiques :
- **`kpis_today.csv` / `.json`** :  
  stations actives/offline, disponibilité vélo/place, taux de **pénurie/saturation**, **couverture** récente et **volatilité** intra-journée.
- **`snapshot_distribution.csv`** :  
  part (%) instantanée : `bike_avail` (≥1 vélo), `dock_avail` (≥1 place), `penury` (=0 vélo), `saturation` (=0 place).
- **`stations_tension_last_days.csv`** *(si `--last-days > 0`)* :  
  *taux* de pénurie/saturation sur la fenêtre récente, par station, pour repérer les **zones sous tension**.

---

## Définitions rapides (indicateurs)
- **Disponibilité vélo** = `1[bikes > 0]` (part de stations offrant au moins 1 vélo).  
- **Disponibilité place** = `1[docks_avail > 0]` (si la colonne existe dans la source).  
- **Taux de pénurie** = part de stations avec `bikes == 0`.  
- **Taux de saturation** = part de stations avec `docks_avail == 0` (si dispo).  
- **Couverture (fenêtre récente)** = part d’horodatages valides (présents) sur la période `--last-days`.  
- **Volatilité intra-journée** = écart-type des vélos dispos **dans la journée en cours**, agrégé multi-stations.

---

## Méthodologie (résumé)
- **Source** : `docs/exports/events.parquet` (timestamps arrondis **15 min**).  
- **Lecture des colonnes** : détection souple (`bikes` / `nb_velos_bin` / `velos_disponibles`, etc.) et normalisation `station_id`, `ts`.  
- **Snapshot map** : dernier `ts` ; marqueurs colorés selon pénurie/saturation ; centre carte = médiane lat/lon du snapshot.  
- **Courbe “aujourd’hui vs médiane”** :  
  1) on calcule pour chaque `ts` la **part de stations** avec `bikes>0`,  
  2) on regroupe par **hh:mm locale** (Europe/Paris) et on moyenne,  
  3) on trace la série du jour, et la **médiane** des jours de même **weekday** dans la fenêtre récente.  
- **Fenêtre récente (tables tension)** : `--last-days` jours (ex : 7) ; calcul des taux par station et tri décroissant.  

---

## Lecture & limites
!!! note "À garder en tête"
    - Les indicateurs de disponibilité sont **structurels** (présence/absence) et **indépendants** d’une capacité exacte.  
    - Une station peut être très utilisée **sans** tomber souvent à 0 vélo **ou** 0 place → elle apparaît “saine”.  
    - Coupures réseau/maintenance peuvent fausser la vision sur quelques heures (couverture plus basse).  
    - `docks_avail` n’est pas toujours publié par la source → les métriques “place/saturation” peuvent être `NaN`.

---

## Paramètres de build (rappel)
- CLI :  
  ```bash
  python tools/build_network_overview.py \\
    --events docs/exports/events.parquet \\
    --last-days 7 \\
    --tz Europe/Paris
ations_tension_last_days.csv` : stations avec taux de pénurie/saturation élevés sur la fenêtre récente.
"""
),

            ("stations.md", "# Stations & profils\n\nTable filtrable, fiches station, **clustering** (k-means/HDBSCAN).\n\n## Figures\n{FIGS:network/stations}\n{LIST:tables/network/stations}\n"),
            ("dynamics.md", "# Dynamiques spatio-temporelles\n\nHeatmaps h×j, saisons, vagues de tension.\n\n## Figures\n{FIGS:network/dynamics}\n{LIST:tables/network/dynamics}\n"),
        ],
    },


"model": {
    "pages": [
        (
            "index.md",
            "# Modèle — index\n\n"
            "- **Performance & baseline**\n"
            "- **Pipeline d’entraînement & features**\n"
            "- **Explicabilité & calibration**\n\n"
            "{LIST:figs/model}\n"
        ),
        (
            "performance.md",
            "# Performance & baseline\n\n"
            "MAE/RMSE/biais, lift vs persistance, découpages.\n\n"
            "## Figures\n"
            "{FIGS:model/performance}\n"
            "{LIST:tables/model/performance}\n"
        ),
        (
            "pipeline.md",
            "# Pipeline d’entraînement & features\n\n"
            "Données d’entrée, lags/rollings, validation time-aware.\n\n"
            "## Figures/Tables\n"
            "{FIGS:model/pipeline}\n"
            "{LIST:tables/model/pipeline}\n"
        ),
        (
            "explainability.md",
            '''# Explicabilité & calibration

Objectif : rendre les prévisions **intelligibles** (quelles variables comptent, quand et où ?) et **fiables** (calibration, biais, incertitudes).

!!! tip "Ce que vous verrez"
    - **Résidus & diagnostics** : histogramme, QQ-plot, ACF, hétéroscédasticité.  
    - **Importance** globale (permutation) et **ablation** par familles de features.  
    - **Profils moyens conditionnels** (PDP/ICE) sur variables clés.  
    - **Calibration** globale et **par segments** (pente/intercept).  
    - **Incertitude** (si activée) : intervalles, coverage empirique.

---

## 1) Résidus & diagnostic

- **Résidus** `y_true − y_pred` : distribution, symétrie, extrêmes.  
- **QQ-plot** : écart aux quantiles théoriques.  
- **ACF des résidus** : dépendance temporelle résiduelle.  
- **Hétéroscédasticité** : variance des résidus vs niveau d’occupation.  
- **Erreurs relatives** par niveau d’occupation.

### Figures
![](../assets/figs/model/explainability/residual_hist.png)
![](../assets/figs/model/explainability/residual_qqplot.png)
![](../assets/figs/model/explainability/residual_acf.png)
![](../assets/figs/model/explainability/heteroscedasticity.png)
![](../assets/figs/model/explainability/heteroscedasticity_mae_by_true_quantile.png)
![](../assets/figs/model/explainability/relative_error_by_level.png)

**Tables/CSV**  
- `../assets/tables/model/explainability/acf_values.csv`  
- `../assets/tables/model/explainability/heteroscedasticity_by_true_quantile.csv`  
- `../assets/tables/model/explainability/error_episodes_by_station.csv`

---

## 2) Importance & explications
- **Permutation importance (globale)** sur un échantillon **time-aware**.  
- **Ablation** par familles de features (lags, rollings, saisonnalité, météo).  
- **PDP/ICE** : effet marginal de 2–3 variables clés.  
- **Segments** : importance/erreurs par cluster de stations.

> L’importance **décrit** une contribution prédictive moyenne (association), pas une causalité.

*(Figures à venir si générées : `permutation_importance.png`, `ablation_by_family.png`, `pdp_<feature>.png`.)*

---

## 3) Calibration & biais
- **Régression d’étalonnage** `y_true = α + β · y_pred` (β≈1, α≈0 souhaités).  
- Pentes/intercepts **par segments** (heure, cluster, capacité, zone).  
- **Courbes de calibration** (binning quantiles).

### Figures
![](../assets/figs/model/explainability/calibration_curve.png)
![](../assets/figs/model/explainability/calibration_scatter.png)
![](../assets/figs/model/explainability/calibration_beta_by_hour.png)

**Règles de surveillance**  
- |pente−1| > 0,1 ou |intercept| > 0,5 → **alerte calibration**.

---

## 4) Incertitude (si activée)
- **Intervalles** (ex. P10–P90) ou **jackknife/bootstrap**.  
- **Coverage** nominal vs empirique.  
- **Stations à forte incertitude** : priorisation du monitoring.

---

## Méthodologie (résumé)
- **Échantillonnage** : validation/test **dans l’ordre temporel**.  
- **Permutation par blocs temporels** (évite fuite d’infos).  
- **PDP/ICE** : sous-échantillon récent ; variables normalisées si besoin.  
- **Calibration** : régression robuste (Huber) & par segments.  
- **Exports** : figures → `assets/figs/model/explainability/`, tables → `assets/tables/model/explainability/`.

## Paramètres de build (rappel)
```bash
python tools/build_model_explainability.py \
  --perf docs/exports/perf.parquet \
  --tz Europe/Paris

'''
),
],






    "monitoring": {
        "pages": [
            ("index.md", "# Monitoring — index\n\n- **Santé des données**\n- **Drift des données**\n- **Santé du modèle**\n\n{LIST:figs/monitoring}\n"),
            ("data-health.md", "# Santé des données\n\nFraîcheur, complétude, schéma, anomalies.\n\n## Figures\n{FIGS:monitoring/data_health}\n{LIST:tables/monitoring/data_health}\n"),
            ("drift.md", "# Drift des données\n\nPSI/K–S, dérive par segments, séries.\n\n## Figures\n{FIGS:monitoring/drift}\n{LIST:tables/monitoring/drift}\n"),
            ("model-health.md", "# Santé du modèle\n\nMAE/RMSE/lift, calibration, couverture.\n\n## Figures\n{FIGS:monitoring/model_health}\n{LIST:tables/monitoring/model_health}\n"),
        ],
    },
    "data": {
        "pages": [
            ("index.md", "# Données — index\n\n- **Exports** (`events.parquet`, `perf.parquet`)\n- **Dictionnaire & schéma**\n- **Méthodologie & licences**\n\n{LIST:figs/exports}\n{LIST:tables/exports}\n"),
            ("exports.md", "# Exports\n\nInventaire, schéma minimal et garanties.\n\n## Figures\n{FIGS:exports}\n\n## Tables\n{LIST:tables/exports}\n"),
            ("dictionary.md", "# Dictionnaire & schéma\n\nContrat formel (types, bornes, clés) et validation.\n\n## Tables\n{LIST:tables/data/dictionary}\n"),
            ("methodology.md", "# Méthodologie & licences\n\nNormalisation (UTC 15min), cible/baseline, injection modèle, versionnage, licences.\n\n## Schéma de flux\n{IMG:figs/data/methodology/dataflow.png}\n\n## Tables\n{LIST:tables/data/methodology}\n"),
        ],
    },
},
}

INDEX_MD = """# Vélib’ Paris — Forecast (Batch)

Ce site documente l’**usage du réseau**, le **modèle de prévision**, le **monitoring** et les **exports**.
Les figures et tableaux sont régénérés régulièrement.

"""

def newest_mtime(path: Path) -> float:
    latest = 0.0
    if path.exists():
        for p in path.rglob("*"):
            try:
                latest = max(latest, p.stat().st_mtime)
            except Exception:
                pass
    return latest

def human_dt(ts: float) -> str:
    if ts <= 0:
        return "n/a"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def rel_from(md_path: Path, target_path: Path) -> str:
    """Chemin relatif depuis le dossier contenant md_path vers target_path (POSIX)."""
    try:
        rel = os.path.relpath(str(target_path), start=str(md_path.parent))
    except Exception:
        # (au cas où, chemins sur volumes différents)
        rel = str(target_path)
    return rel.replace("\\", "/")


def render_images(md_path: Path, dir_rel_under_assets_figs: str, limit: int = 12) -> str:
    figs_dir = ASSETS / "figs" / dir_rel_under_assets_figs
    if not figs_dir.exists():
        return "_Aucune figure disponible pour le moment._\n"
    imgs = sorted(figs_dir.glob("*.png"))[:limit]
    if not imgs:
        return "_Aucune figure disponible pour le moment._\n"
    return "\n\n".join(f"![{img.stem}]({rel_from(md_path, img)})" for img in imgs) + "\n"

def render_list(md_path: Path, dir_rel_under_assets: str, max_items: int = 20) -> str:
    base_dir = ASSETS / dir_rel_under_assets
    if not base_dir.exists():
        return f"_Aucun asset dans `assets/{dir_rel_under_assets}` pour le moment._\n"
    files = sorted(p for p in base_dir.rglob("*") if p.is_file())
    if not files:
        return f"_Aucun asset dans `assets/{dir_rel_under_assets}` pour le moment._\n"
    out = []
    for p in files[:max_items]:
        out.append(f"- `{rel_from(md_path, p)}`")
    if len(files) > max_items:
        out.append(f"- … (+{len(files)-max_items} autres)")
    return "\n".join(out) + "\n"

def render_tokens(md_path: Path, content: str) -> str:
    out = content

    # {FIGS:dir}
    for m in re.finditer(r"\{FIGS:([^\}]+)\}", out):
        out = out.replace(m.group(0), render_images(md_path, m.group(1).strip()))

    # {IMG:path}
    for m in re.finditer(r"\{IMG:([^\}]+)\}", out):
        img = ASSETS / m.group(1).strip()
        out = out.replace(m.group(0),
                          f"![{img.stem}]({rel_from(md_path, img)})\n" if img.exists()
                          else "_Figure non disponible._\n")

    # {LIST:dir}
    for m in re.finditer(r"\{LIST:([^\}]+)\}", out):
        out = out.replace(m.group(0), render_list(md_path, m.group(1).strip()))

    # {IFRAME:path_under_assets[|height]}
    for m in re.finditer(r"\{IFRAME:([^\}|]+)(?:\|(\d+))?\}", out):
        path = m.group(1).strip()
        height = int(m.group(2)) if m.group(2) else 520
        out = out.replace(m.group(0), render_iframe(md_path, path, height))


    return out

def render_iframe(md_path: Path, rel_under_assets: str, height: int = 520) -> str:
    html_path = ASSETS / rel_under_assets
    if not html_path.exists():
        return "_Carte non disponible._\n"
    src = rel_from(md_path, html_path)
    return (f'<div style="margin: 0.5rem 0;">'
            f'<iframe src="{src}" style="width:100%;height:{height}px;border:0" '
            f'loading="lazy" title="Carte"></iframe></div>\n')


def write_md(path: Path, content: str, force: bool) -> None:
    ensure_dir(path)
    if path.exists() and not force:
        return
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)

def build_all(force: bool) -> None:
    latest = max(newest_mtime(ASSETS / "tables"), newest_mtime(ASSETS / "figs"))
    write_md(DOCS / "index.md", INDEX_MD + f"\n**Dernière mise à jour des assets :** {human_dt(latest)}\n", force)

    for section, cfg in SECTIONS.items():
        for fname, tpl in cfg["pages"]:
            md_path = DOCS / section / fname
            write_md(md_path, render_tokens(md_path, tpl), force)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Génère les pages Markdown avec chemins relatifs corrects")
    ap.add_argument("--force", action="store_true", help="Écrase les fichiers existants")
    args = ap.parse_args()
    build_all(force=args.force)
    print("[docs] Pages Markdown générées / mises à jour.")
