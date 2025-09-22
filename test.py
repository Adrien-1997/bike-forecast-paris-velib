import io
import requests
import pandas as pd

# URL directe du fichier Parquet sur HuggingFace
url = "https://huggingface.co/datasets/Adrien97/velib-monitoring-historical/resolve/main/exports/velib.parquet"

# Télécharger et lire le parquet
resp = requests.get(url)
resp.raise_for_status()
df = pd.read_parquet(io.BytesIO(resp.content))

# Afficher les 10 dernières lignes
print(df.tail(10))
