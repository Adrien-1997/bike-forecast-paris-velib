from __future__ import annotations

# --- fix chemin d'import ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------------
import numpy as np
import pandas as pd

from typing import Optional

import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.features import build_training_frame

# === Add at the end of src/features.py ===


def prepare_live_features(df_live: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Aligne df_live sur la liste de features attendues par le modèle.
    - Si une feature attendue est absente, on la crée avec une valeur par défaut raisonnable.
    - On reconstruit des features temporelles simples (hour_local, dow, is_weekend) côté live.
    - On force tout en numérique (LightGBM) et on respecte l'ordre de feat_cols.
    """
    df = df_live.copy()
    X = pd.DataFrame(index=df.index)

    # Heure locale Paris (utile pour hour_local, dow, is_weekend)
    try:
        now = pd.Timestamp.now(tz="Europe/Paris")
    except Exception:
        now = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Europe/Paris")

    # Helpers
    def num(s):
        return pd.to_numeric(s, errors="coerce")

    # Valeurs temporelles par défaut
    hour_val = now.hour
    dow_val = now.weekday()
    weekend_val = int(dow_val >= 5)
    holiday_val = 0
    try:
        import holidays
        fr_holidays = holidays.country_holidays("FR", subdiv="75", years=[now.year, now.year-1, now.year+1])
        holiday_val = int(now.date() in fr_holidays)
    except Exception:
        pass

    # Construit chaque feature attendue
    for c in feat_cols:
        if c in df.columns:
            X[c] = num(df[c])
            if X[c].isna().all():
                X[c] = 0
        elif c in ("hour_local", "hour"):
            X[c] = hour_val
        elif c in ("dow", "dayofweek"):
            X[c] = dow_val
        elif c in ("is_weekend", "weekend"):
            X[c] = weekend_val
        elif c in ("is_holiday", "holiday"):
            X[c] = holiday_val
        elif c == "capacity":
            X[c] = num(df.get("capacity")).fillna(0)
        elif c == "numbikesavailable":
            X[c] = num(df.get("numbikesavailable")).fillna(0)
        elif c == "numdocksavailable":
            X[c] = num(df.get("numdocksavailable")).fillna(0)
        elif c in ("temp_C", "precip_mm", "wind_mps"):
            # Si la météo live est absente, on met 0 par défaut
            X[c] = num(df.get(c)).fillna(0)
        else:
            # Fallback neutre
            X[c] = 0

    # S'assure que toutes les colonnes sont numériques
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = num(X[col]).fillna(0)

    # Respecte l'ordre des features attendues
    missing_backfill = [c for c in feat_cols if c not in X.columns]
    for c in missing_backfill:
        X[c] = 0
    return X[feat_cols].copy()


# -----------------------------
# Utils catégorielles / Dataset
# -----------------------------
def _detect_cats(X: pd.DataFrame) -> list[str]:
    cats = []
    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_categorical_dtype(dt):
            cats.append(c)
    return cats


def _harmonize_categories(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xt, Xv = X_train.copy(), X_valid.copy()
    for c in cat_cols:
        if not pd.api.types.is_categorical_dtype(Xt[c]):
            Xt[c] = Xt[c].astype("category")
        if not pd.api.types.is_categorical_dtype(Xv[c]):
            Xv[c] = Xv[c].astype("category")
        lvls = pd.Index(Xt[c].cat.categories).union(pd.Index(Xv[c].cat.categories))
        Xt[c] = Xt[c].cat.set_categories(lvls)
        Xv[c] = Xv[c].cat.set_categories(lvls)
    return Xt, Xv


def _make_lgb_datasets(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    use_cats: bool = True
) -> tuple[lgb.Dataset, lgb.Dataset, list[str], list[str]]:
    feat_cols = list(X_train.columns)

    if use_cats:
        cat_cols = _detect_cats(X_train)
        X_train_h, X_valid_h = _harmonize_categories(X_train, X_valid, cat_cols)
        train_set = lgb.Dataset(X_train_h, label=y_train, feature_name=feat_cols, categorical_feature=cat_cols)
        valid_set = lgb.Dataset(X_valid_h, label=y_valid, feature_name=feat_cols, categorical_feature=cat_cols)
        return train_set, valid_set, feat_cols, cat_cols

    def _to_float(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    X_train_f = _to_float(X_train)
    X_valid_f = _to_float(X_valid)
    train_set = lgb.Dataset(X_train_f, label=y_train, feature_name=feat_cols)
    valid_set = lgb.Dataset(X_valid_f, label=y_valid, feature_name=feat_cols)
    return train_set, valid_set, feat_cols, []


# -----------------------------
# Entraînement + sauvegarde
# -----------------------------
def train(
    horizon_hours: int = 1,
    lookback_days: int = 7,
    use_cats: bool = False,
    model_dir: str | Path = "models"
):
    import numpy as np
    from pathlib import Path
    import joblib
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from src.features import build_training_frame

    # 1) Données
    df, feat_cols = build_training_frame(horizon_hours=horizon_hours, lookback_days=lookback_days)
    if df.empty:
        raise ValueError("Dataset vide — vérifie exports/velib_hourly.parquet / src.aggregate.")

    X = df[feat_cols].copy()
    y = df["y_nb"].astype(float)

    # 2) Gestion des dtypes
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if use_cats:
        # Catégorielles: cast → category
        for c in obj_cols:
            X[c] = X[c].astype("category")
        cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
        print(f"[train] Categorical columns ({len(cat_cols)}): {cat_cols[:8]}{' ...' if len(cat_cols)>8 else ''}")
    else:
        # Numérique only: drop objets
        if obj_cols:
            print(f"[train] Dropping non-numeric cols (use_cats=False): {obj_cols}")
            X.drop(columns=obj_cols, inplace=True, errors="ignore")
            feat_cols = [c for c in feat_cols if c not in obj_cols]
        cat_cols = []

    # 3) Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if X_train.empty or X_valid.empty:
        raise ValueError("Train/valid vides après split — réduis lookback_days.")

    # 4) Datasets LGBM
    train_set = lgb.Dataset(
        X_train, label=y_train,
        feature_name=list(X_train.columns),
        categorical_feature=cat_cols
    )
    valid_set = lgb.Dataset(
        X_valid, label=y_valid,
        feature_name=list(X_valid.columns),
        categorical_feature=cat_cols
    )

    # 5) Params
    params = dict(
        objective="regression",
        metric=["l1", "l2"],
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        seed=42,
        verbose=-1,
    )

    # 6) Entraînement
    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
    )

    # 7) Sauvegarde bundle (modèle + features + cat list)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(model_dir) / f"lgb_nbvelos_T+{horizon_hours}h.joblib"
    bundle = {
        "model": model,
        "features": list(X.columns),   # ordre EXACT des features utilisées
        "categoricals": cat_cols,
        "horizon": horizon_hours,
        "version": "v1.0",
    }
    joblib.dump(bundle, out_path)
    print(f"[train] OK → {out_path} (features={len(bundle['features'])}, cats={len(cat_cols)})")
    return out_path

# -----------------------------
# Chargement modèle + prédiction
# -----------------------------
def load_model_bundle(horizon_hours: int = 1, model_dir: Path | str = "models"):
    path = Path(model_dir) / f"lgb_nbvelos_T+{horizon_hours}h.joblib"
    artefact = joblib.load(path)

    if isinstance(artefact, dict):
        return artefact["model"], artefact.get("features", [])
    return artefact, getattr(artefact, "feature_name_", None)


def prepare_live_features(df_live: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    X = df_live.reindex(columns=feature_list, fill_value=np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def predict_from_bundle(df_live: pd.DataFrame, horizon_hours: int = 1, model_dir: Path | str = "models") -> np.ndarray:
    model, feats = load_model_bundle(horizon_hours, model_dir)
    X = prepare_live_features(df_live, feats)
    best_it = getattr(model, "best_iteration", None)
    y = model.predict(X, num_iteration=best_it)
    return np.asarray(y, dtype=float)