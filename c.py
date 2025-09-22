import pandas as pd

# Charger le fichier original
df = pd.read_parquet("velib.parquet")

# Créer un dataframe vide avec mêmes colonnes et dtypes
empty_df = df.iloc[0:0].copy()

# Sauvegarder en parquet
empty_df.to_parquet("velib_empty.parquet", index=False)
