import pandas as pd

# Lecture du fichier parquet local
df = pd.read_parquet("exports/velib.parquet")

# Afficher infos et 10 dernières lignes avec toutes les colonnes
pd.set_option("display.max_columns", None)
print(df.info())
print(df.tail(10))
