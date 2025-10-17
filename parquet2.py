from google.cloud import storage
import joblib, io, os

uri = os.environ["MODEL_URI"]
bucket, key = uri[5:].split("/", 1)
print(f"🔎 Lecture du modèle : {uri}\n")

client = storage.Client()
blob = client.bucket(bucket).blob(key)
buf = io.BytesIO(blob.download_as_bytes())
pack = joblib.load(buf)

print("📦 Clés disponibles :", list(pack.keys()))
print("\n🧩 Nombre de features :", len(pack.get("feat_cols", [])))
print("\n🧾 Liste des features :")
for i, c in enumerate(pack.get("feat_cols", []), 1):
    print(f"{i:>3}. {c}")