import pyarrow.parquet as pq
import pandas as pd

def describe_parquet(path, n_head=5, show_stats=True, stats_columns=None):
    """
    Affiche des informations descriptives sur un fichier Parquet.

    Args:
        path (str): chemin vers le fichier .parquet
        n_head (int): nombre de lignes à afficher en tête
        show_stats (bool): si True, calcul de quelques statistiques simples
        stats_columns (list of str or None): colonnes pour lesquelles on veut les stats (None -> toutes colonnes numériques)
    """
    try:
        pq_file = pq.ParquetFile(path)
    except Exception as e:
        print(f"Erreur à l'ouverture du fichier Parquet : {e}")
        return

    metadata = pq_file.metadata
    schema = pq_file.schema

    print("=== Schéma du fichier Parquet ===")
    print(schema)

    print("\n=== Métadonnées globales ===")
    print(f"Nombre de lignes estimé : {metadata.num_rows}")
    print(f"Nombre de colonnes : {metadata.num_columns}")
    print(f"Nombre de row groups : {metadata.num_row_groups}")
    print(f"Taille (bytes) approximative : {metadata.serialized_size}")

    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        print(f"\n--- Row group {i} ---")
        print(f"  Lignes dans le groupe : {rg.num_rows}")
        print(f"  Taille (bytes) du groupe : {rg.total_byte_size}")
        for j in range(rg.num_columns):
            col = rg.column(j)
            col_name = schema.names[j]
            stats = col.statistics
            print(f"    Colonne {col_name} :")
            if stats is not None:
                print(f"      Nulls      : {stats.null_count}")
                print(f"      Min        : {stats.min}")
                print(f"      Max        : {stats.max}")
            else:
                print("      Pas de statistiques disponibles pour cette colonne")

    if n_head > 0:
        print(f"\n=== Les {n_head} premières lignes ===")
        try:
            # ici on lit juste le début pour ne pas faire exploser la mémoire
            df_head = pq_file.read_row_groups([0]).to_pandas()
            print(df_head.head(n_head))
        except Exception as e:
            print(f"Erreur lors de la lecture des premières lignes: {e}")

    if show_stats:
        try:
            df = pq_file.read().to_pandas()
        except Exception as e:
            print(f"Erreur lors de la lecture complète du fichier: {e}")
            return
        if stats_columns is None:
            numeric = df.select_dtypes(include=["number"]).columns.tolist()
        else:
            numeric = [c for c in stats_columns if c in df.columns]
        print("\n=== Statistiques colonnes numériques ===")
        if not numeric:
            print("  Aucune colonne numérique détectée ou spécifiée pour les stats.")
        else:
            stats = df[numeric].describe()
            print(stats)

if __name__ == "__main__":
    chemin = input("Saisir le chemin complet du fichier Parquet (/chemin/vers/fichier.parquet) : ")
    if not chemin:
        print("Aucun chemin saisi — on arrête.")
    else:
        describe_parquet(chemin)
