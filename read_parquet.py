from pathlib import Path
import pandas as pd

# === Configuration ===
parquet_path = Path(r"H:\Downloads\velib_exports_perf_2025-10-11.parquet")  # 🔸 à modifier

# === Lecture ===
df = pd.read_parquet(parquet_path)

# === Paramètres d'affichage ===
pd.set_option("display.max_columns", None)   # Affiche toutes les colonnes
pd.set_option("display.max_rows", 10)        # N’affiche que 10 lignes max
pd.set_option("display.width", 0)            # Ajuste automatiquement la largeur du terminal
pd.set_option("display.max_colwidth", None)  # Affiche le contenu complet des cellules

# === Échantillonnage aléatoire ===
sample = df.sample(n=10, random_state=None)

print(sample)
