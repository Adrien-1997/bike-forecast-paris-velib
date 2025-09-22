import io
import requests
import pandas as pd

url = "https://huggingface.co/datasets/Adrien97/velib-monitoring-historical/resolve/main/exports/velib.parquet"
resp = requests.get(url)
resp.raise_for_status()
df = pd.read_parquet(io.BytesIO(resp.content))

# Forcer l'affichage de toutes les colonnes
pd.set_option("display.max_columns", None)

# Dernière date disponible
last_date = pd.to_datetime(df["tbin_utc"]).max()

# Filtrer toutes les lignes correspondant à cette date
df_last = df[df["tbin_utc"] == last_date]

print(f"Dernière date : {last_date}")
print(df_last)
