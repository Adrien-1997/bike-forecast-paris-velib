# Réseau — Présentation générale

Cette section explore **comment le réseau Vélib’ vit dans l’espace et dans le temps**. Elle se compose de trois pages complémentaires :  

- [**Aperçu du réseau**](./overview.md) — KPIs immédiats, carte, tendances.  
- [**Stations & profils**](./stations.md) — table exploratoire, fiches station, **clustering** des comportements.  
- [**Dynamiques spatio-temporelles**](./dynamics.md) — heatmaps h×j, saisonnalité courte/longue, flux intra-urbains.

> **Granularité temporelle**  
> Les analyses sont réalisées à pas **15 minutes** en **UTC** (affichage local possible). Les agrégations (jour, semaine, mois) s’appuient sur ces pas.

> **Jeux de données utilisés**  
> - `events.parquet` : comptages et états par station (bikes/docks), 15 min.  
> - `perf.parquet` : séries cibles `y_true` et, si disponible, `y_pred` (utile pour certaines métriques d’usage).  
> Les deux fichiers sont régénérés régulièrement et reflètent l’état le plus récent du réseau.

---

## 1) Aperçu du réseau (ce que vous trouverez dans `overview.md`)

### Objectif
Donner en un coup d’œil la **santé du réseau aujourd’hui** et la situer par rapport aux semaines précédentes.

### Questions auxquelles la page répond
- Combien de **stations actives** / **hors ligne** actuellement ?  
- Quel est le **niveau de disponibilité** (part de stations avec au moins 1 vélo / 1 place) ?  
- Où sont les **zones sous tension** (pénuries récurrentes, saturation) ?  
- Comment se comparent les **pics/creux** d’aujourd’hui à la médiane des dernières semaines ?

### Indicateurs clés
- **Disponibilité vélo** = `1[velos_disponibles > 0]` (taux de stations offrant au moins 1 vélo).  
- **Disponibilité place** = `1[docks_disponibles > 0]`.  
- **Taux de saturation** = part de stations à `docks_disponibles = 0`.  
- **Taux de pénurie** = part de stations à `velos_disponibles = 0`.  
- **Couverture** = part d’horodatages valides (données présentes).  
- **Volatilité intra-journalière** (écart-type des vélos dispo par station, agrégé).

### Visualisations principales
- **Carte instantanée** (ou la plus récente) avec couches *pénurie* / *saturation*.  
- **KPIs du jour** vs **médiane J-7/J-14/J-21** (barres ou cartes d’étincelles).  
- **Courbe journée type** (médiane horaire des dernières N semaines) superposée à la journée en cours.

### Lecture & limites
- Les indicateurs de disponibilité sont **structurels** (présence/absence) et **indépendants de la capacité**.  
- Une station « saine » peut être très utilisée mais rarement en zéro vélo **ou** zéro place.  
- Les maintenances et coupes réseau peuvent fausser temporairement la perception sur quelques heures.

---

## 2) Stations & profils (ce que vous trouverez dans `stations.md`)

### Objectif
Permettre une **exploration fine station par station**, avec une table filtrable et des **profils comportementaux** (clustering).

### Table exploratoire
- **Colonnes suggérées** : ID, nom, capacité estimée, taux de pénurie/saturation (7 & 30 jours), volatilité, couverture, quartier/arrondissement, distance au centre, cluster (voir ci-dessous).  
- **Tri/Filtre** : top pénuries, top saturations, variabilité, cluster, zone géographique.

### Fiches station (liens depuis la table)
Chaque station dispose d’une fiche synthétique (ouverture dans une page dédiée) comprenant :  
- **Sparkline** 7 jours (vélos disponibles).  
- **Profil 24 h typique** (médiane par quart d’heure).  
- **Heatmap h×j** récente (intensité d’usage).  
- **Indicateurs** : pénurie/saturation (7 & 30 jours), volatilité, couverture, événements anormaux récents.

### Clustering des comportements (expliqué)
Objectif : **regrouper les stations par similarité d’usage** pour comprendre des archétypes (résidentiel, pôle d’emplois, gares, touristique…).

**Variables de description (features)**
- **Profil 24 h** : vecteur de longueur 96 (15 min) = médiane des vélos dispo par pas, sur une fenêtre récente (ex. 28 jours), **centré-réduit**.  
- **Amplitude/variabilité** : écart-type quotidien, plage min-max normalisée.  
- **Asymétries temporelles** : ratios matin/soir, semaine/week-end.  
- **Contexte léger** (optionnel) : capacité, distance centre, altitude (si dispo).  

**Pré-traitement**
- **Normalisation** (StandardScaler) pour rendre comparables les courbes.  
- **Réduction** (PCA 2-3D) **uniquement pour la visualisation** : le clustering se fait sur l’espace complet normalisé.

**Algorithmes**
- **K-Means** (par défaut) avec *k* choisi via coudes + **Silhouette**/**Davies-Bouldin**.  
- **HDBSCAN** (option robuste) quand les densités diffèrent fortement : gère **bruit/outliers** sans imposer *k*.

**Attribution & stabilité**
- Évaluation interne (Silhouette) et stabilité par re-sampling (bootstrap sur semaines).  
- **Centroides typiques** restitués comme *“comportements-types”* interprétables (courbes moyennes).  
- Stations proches des frontières signalées (incertitude).

**Étiquettes interprétables (exemples)**
- **Résidentiel nocturne** : haut la nuit, baisse le matin (départs), remonte le soir (retours).  
- **Pôle d’emplois** : bas la nuit, pics d’arrivée le matin, vidage en fin de journée.  
- **Transport/gares** : fortes oscillations, pics synchronisés aux pointes.  
- **Touristique/loisirs** : activité plus marquée week-end, milieux de journée.

**Sorties affichées**
- **Carte par cluster** (couleur = cluster, taille = capacité).  
- **Courbes typiques** par cluster (centroïdes).  
- **Distribution des clusters** par arrondissement / zone.

> **Limites**  
> - Le clustering **décrit** les usages, il ne **prédit** pas.  
> - Les clusters peuvent évoluer avec la saison ou des travaux ; un recalcul périodique est prévu.

---

## 3) Dynamiques spatio-temporelles (ce que vous trouverez dans `dynamics.md`)

### Objectif
Mettre en évidence les **rythmes** et **déplacements de pression** dans la ville.

### Analyses proposées
- **Heatmaps h×j** (par station et agrégées) : intensité des vélos disponibles ou du taux de pénurie, par heure du jour et jour de semaine.  
- **Saisonnalité courte/longue** :  
  - **Intra-semaine** (lundi→dimanche) : comparaison des profils typiques.  
  - **Intra-année** (si historique suffisant) : effet météo/vacances (qualitatif), glissements de pics.  
- **Cartes temporelles animées** (ou séquences d’instantanés) pour suivre la **vague de saturation → pénurie** sur une journée type.  
- **Flux intra-urbains (qualitatif)** : lecture conjointe des zones qui passent de saturation à pénurie avec un décalage horaire (indication de “courant” de déplacement).

### Indicateurs & méthodes
- **Indice de tension** par station : `penurie_rate + saturation_rate` (7 jours).  
- **Score de régularité** : corrélation de la journée en cours à la journée type (90 derniers jours).  
- **Détection d’épisodes** : séquences ≥ X pas en pénurie/saturation (morphologie binaire).  
- **Agrégations spatiales** : par arrondissement/quartier (moyennes pondérées par capacité).

### Lecture & limites
- Les heatmaps mettent à nu la **récurrence** des phénomènes ; elles n’expliquent pas la cause (météo, événements).  
- Les “flux” sont déduits **visuellement** par co-évolution des zones ; ils ne sont pas des trajectoires individuelles.

---

## Valeur analytique de la section “Réseau”
- **Opérationnel** : repérer rapidement les zones à surveiller (redispatch).  
- **Stratégique** : comprendre les **archétypes d’usage** et leur évolution (clustering).  
- **Communication** : visualisations pédagogiques pour le grand public (profil 24 h, cartes).

### Bonnes pratiques de lecture
- Toujours croiser **pénurie** et **saturation** (les deux faces d’un déséquilibre).  
- Un **taux de disponibilité élevé** ne signifie pas faible tension : regarder la **volatilité**.  
- Les **clusters** aident à comparer *des stations comparables* entre elles.

---

## Aller plus loin
- Page suivante : [**Aperçu du réseau**](./overview.md)  
- Ou explorez directement : [**Stations & profils**](./stations.md) • [**Dynamiques spatio-temporelles**](./dynamics.md)
