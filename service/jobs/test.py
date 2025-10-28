import pandas as pd
import pyarrow.parquet as pq

# Lecture du fichier events (chemin GCS exact)
ev = pq.read_table("gs://velib-forecast-472820_cloudbuild/velib/exports/events.parquet").to_pandas()

# Conversion des timestamps
ev["tbin_utc"] = pd.to_datetime(ev["tbin_utc"], errors="coerce")

# Écarte les NaT et mesure le décalage à la grille 5 minutes
off = ((ev["tbin_utc"].dt.minute % 5) * 60 + ev["tbin_utc"].dt.second).fillna(0)

print("share on-grid (exacte 00s) =", (off == 0).mean())
print(off.describe())
