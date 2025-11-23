# api/schemas/forecast.py

"""Schémas Pydantic pour les réponses de prévision.

Ce module définit les modèles utilisés par les routes de prévision
(par exemple `/forecast` et `/forecast/latest`) pour sérialiser proprement
les objets renvoyés par l'API.

Objectif
--------
- Centraliser la structure des réponses de prévision.
- Bénéficier de la validation Pydantic et de la génération automatique
  de schémas OpenAPI.
- Éviter de manipuler des `dict` libres dans les routes : on instancie
  simplement `ForecastItem` avec les enregistrements renvoyés par
  le moteur de forecast.

Usage typique côté route
------------------------
Après avoir récupéré des enregistrements sous forme de dictionnaires
(par exemple à partir d'un DataFrame ou d'un JSON GCS) :

    return [ForecastItem(**rec) for rec in records]

Cela garantit que chaque élément respecte le contrat de l'API
(types, champs obligatoires, etc.).
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ForecastItem(BaseModel):
    """Un élément de prévision pour une station et un horizon donnés.

    Ce modèle correspond à une ligne de forecast produite par le backend
    (pipeline de features + modèle) et renvoyée telle quelle par l'API.

    Champs
    ------
    station_id : int
        Identifiant de la station Vélib' (clé interne).
    tbin_latest : datetime
        Timestamp du dernier bin d'observation utilisé comme base de
        la prévision (UTC, bin 5 minutes).
    horizon_min : int
        Horizon de prévision en minutes (ex: 15, 60).
    bikes_pred : float
        Prédiction "continue" du nombre de vélos disponibles, après clip.
    bikes_pred_int : int
        Prédiction arrondie/entière du nombre de vélos disponibles.
    capacity_bin : int
        Capacité numérique associée à la station sur ce bin (borne
        supérieure pour le clip).
    pred_ts_utc : datetime
        Timestamp UTC auquel la prévision est produite (côté backend).
    model_version : str
        Identifiant/version du modèle utilisé (ex: "2.1.3" ou "latest").
        Défaut : chaîne vide si la version n'est pas renseignée.
    """

    station_id: int
    tbin_latest: datetime
    horizon_min: int
    bikes_pred: float
    bikes_pred_int: int
    capacity_bin: int
    pred_ts_utc: datetime
    model_version: str = Field(default="")

# Rappel d’usage côté route :
#   return [ForecastItem(**rec) for rec in records]
