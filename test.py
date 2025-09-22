import io
import requests
import pandas as pd

# URL du parquet (lien brut HuggingFace)
url = "https://huggingface.co/datasets/Adrien97/velib-monitoring-historical/resolve/main/exports/velib.parquet"

# Télécharger et lire le parquet
resp = requests.get(url)
resp.raise_for_status()
df = pd.read_parquet(io.BytesIO(resp.content))

# Forcer pandas à afficher toutes les colonnes
pd.set_option("display.max_columns", None)

# Afficher les 10 dernières lignes avec toutes les colonnes visibles
print(df.tail(10))
