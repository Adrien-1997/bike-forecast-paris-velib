import pandas as pd

# chemin du fichier parquet local
path = r"H:\Documents\2. Perso\github\velib-historical-hf\parquet\velib.parquet"

# lecture du parquet
df = pd.read_parquet(path)

# afficher les 10 dernières lignes avec toutes les colonnes visibles
pd.set_option("display.max_columns", None)
print(df.tail(10))
