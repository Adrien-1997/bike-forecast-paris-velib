# Interface Utilisateur — Documentation Complète (UI)

Ce document décrit l’architecture, les objectifs et les mécanismes internes de l’interface utilisateur du projet **Vélib’ Forecast Paris**.  
Il est destiné aux développeurs, reviewers et recruteurs souhaitant comprendre comment fonctionne la partie **Next.js / React** du front-end.

---

## 1. 🎯 Objectifs de l’UI

L’interface utilisateur a trois missions principales :

1. **Fournir une application grand public** permettant d’explorer les stations Vélib’ en temps réel.
2. **Exposer un module Monitoring avancé**, destiné à illustrer les capacités Data/MLOps du projet :
   - Analyse réseau
   - Santé des données
   - Dérive
   - Performance du modèle
   - Explicabilité
3. **Servir de vitrine technique** pour démontrer :
   - maîtrise de React / Next.js,
   - intégration Plotly,
   - visualisations avancées,
   - Leaflet,
   - architecture propre et industrialisée.

---

## 2. 🏗️ Architecture Générale

```
ui/
 ├─ components/          → composants réutilisables (UI, cartes, KPI, nav…)
 ├─ lib/                 → services HTTP, loaders, helpers, typage
 │   └─ services/        → services du Monitoring + pages /app
 ├─ pages/               → pages Next.js (monitoring, app, landing)
 ├─ public/              → assets statiques (data, images, favicon…)
 ├─ styles/              → CSS globaux + CSS contextuels
 ├─ types/               → d.ts spécifiques (react-plotly, Leaflet…)
 ├─ netlify/             → fonctions serverless proxy API
 ├─ next.config.js       → config Next.js
 ├─ tsconfig.json        → config TS
 └─ package.json         → dépendances & scripts
```

---

## 3. ⚙️ Fonctionnement Global

### 3.1 Routage Contextuel
La page `_app.tsx` identifie trois contextes :
- `landing`
- `app` (la carte)
- `monitoring`

Cela permet :
- de charger automatiquement les feuilles CSS adaptées,
- de garder un design system clair,
- d'activer/désactiver le header / footer selon le contexte.

### 3.2 Mode embed / nochrome
Certaines pages peuvent être affichées sans chrome (pas de header/footer/halo).  
Ce mode s’active via :
- props `noChrome`
- querystring `?embed=1`
- détection automatique d’un iframe

Très utile pour intégration externe.

---

## 4. 🌐 Services HTTP & Caching

Les services utilisent `fetchJsonWithEtag` :
- gestion transparente des ETags,
- revalidation automatique,
- fallback si nécessaire,
- réduction drastique de la bande passante.

Les services sont typés :  
exemple : `/monitoring/model/performance` → `model_performance.ts`

Tous suivent le même modèle propre et lisible.

---

## 5. 📊 Visualisations (Plotly)

L’application utilise **react-plotly.js** (charge en dynamic import, sans SSR) :

- Graphiques de performances : MAE, lift, biais
- Séries temporelles 24h
- Découpes par heure / jour
- Heatmaps 7×24
- Barplots comparatifs J / J−7 / J−14 / J−21

Un thème Plotly dédié est fourni dans `lib/plotlyTheme.ts`.

---

## 6. 🗺️ Cartographie (Leaflet)

L’UI propose des cartes pour :

- le réseau en instantané,
- les clusters,
- les dynamiques (pénuries/saturations),
- les stations top/bottom lift.

Les cartes utilisent un fallback automatique **Carto Light → OSM**.

Chaque carte est encapsulée dans un composant React autonome.

---

## 7. 🧩 Monitoring : Structure des Pages

### `/monitoring`
Vue d’ensemble :
- KPIs réseau
- liens rapides
- statut des sous-systèmes
- conseils

### `/monitoring/network/*`
- **overview** : snapshot global + courbes J−1
- **stations** : clusters, distributions
- **dynamics** : heatmaps, profils, épisodes, tension

### `/monitoring/model/*`
- **performance** : MAE, lift, cartes, stations
- **explainability** : SHAP & dépendance

### `/monitoring/data/*`
- **health** : schéma, complétude, fraîcheur
- **drift** : PSI et dérive

Toutes les pages utilisent :
- `MonitoringNav` pour la navigation interne
- `LoadingBar` pour l’état des fetchs
- `KpiBar` pour l’affichage compact

---

## 8. 📁 Données Locales

### 8.1 Stations Index
Généré via :  
`scripts/buildStationsIndex.ts`

Produit :
```
public/data/stations.index.json
```

Ce fichier compact fournit :
- station_id
- nom
- lat/lon nettoyés
→ très utile pour toutes les cartes Monitoring.

---

## 9. 🚀 Déploiement & Build

### Netlify
Le fichier `netlify.toml` gère :
- build Next.js,
- plugin officiel,
- configuration CSP,
- mapping proxy → API backend Cloud Run.

### Next.js
Configuration via `next.config.js` :
- compilation SWC,
- optimisation images,
- support React strict mode.

### Typescript
Configuration via `tsconfig.json`.

---

## 10. 📦 Dépendances Principales

Extrait depuis `package.json` :
- React / Next.js
- react-plotly.js
- plotly.js
- leaflet / react-leaflet
- classnames

---

## 11. 🛠️ Développement Local

### Installer
```
npm install
```

### Lancer le serveur
```
npm run dev
```

### Build production
```
npm run build
npm run start
```

### Regénérer l’index des stations
```
npx ts-node scripts/buildStationsIndex.ts
```

---

## 12. 🔒 Sécurité

- CSP renforcée (frame-ancestors whitelist)
- ETag partout
- suppression auto des styles dynamiques
- mode embed sécurisé

---

## 13. 📚 Notes Design

- Polices : Urbanist (next/font)
- KPI bars avec color-mix (ok/warn/down)
- Layouts fluides → mobile compatible

---

## 14. 🤝 Contribution

1. Créer une branche `feature/...`
2. Documenter systématiquement les nouveaux services
3. Tenir les pages alignées sur le monitoring.css
4. Garder cohérence des noms : kpi-bar, map-block, plot-card…

---

## 15. 📎 Licence
Projet personnel utilisé comme démonstration technique.  
Non destiné à un usage commercial tiers.
