// scripts/buildStationsIndex.ts
//
// -----------------------------------------------------------------------------
// Script Node pour construire un index des stations consommable par l’UI.
//
// Rôle :
//   - Lire le JSON GBFS brut `station_information.json` (format Vélib’ / GBFS).
//   - Extraire un tableau de stations minimal (code, nom, lat, lon).
//   - Normaliser le code station (string, sans zéros de tête).
//   - Filtrer les enregistrements sans code ou sans nom.
//   - Écrire un fichier JSON compact `stations.index.json` utilisé côté front.
//
// Entrée (chemin relatif au repo) :
//   - src = "ui/public/data/station_information.json"
//       • Attendu : objet de forme { data: { stations: [...] } } (GBFS)
//         ou structure équivalente exposant un tableau via .value.
//
// Sortie :
//   - dst = "ui/public/data/stations.index.json"
//       • Format : { "value": [ { stationcode, name, lat, lon }, ... ] }
//       • Utilisé par l’UI pour indexer / afficher les métadonnées station.
//
// Notes d’implémentation :
//   - stationcode :
//       • pris depuis station_id ou stationcode,
//       • converti en string et nettoyé des zéros en tête (ex: "00012" → "12").
//   - name : trim() systématique pour éviter les espaces parasites.
//   - lat / lon : forcés en Number pour éviter les types ambigus côté front.
//   - Les enregistrements sans code ou sans name sont rejetés par le filter().
//
// Exemple d’exécution (TS/Node) :
//   - npx ts-node scripts/buildStationsIndex.ts
//   - ou transpilation préalable puis : node dist/scripts/buildStationsIndex.js
// -----------------------------------------------------------------------------

import fs from "fs";

const src = "ui/public/data/station_information.json"; // JSON GBFS source
const dst = "ui/public/data/stations.index.json";      // fichier de sortie

const raw = JSON.parse(fs.readFileSync(src, "utf-8"));
const arr = Array.isArray(raw.data?.stations) ? raw.data.stations : raw.value ?? [];

const out = arr
  .map((s: any) => ({
    stationcode: String(s.station_id ?? s.stationcode ?? "").replace(/^0+/, ""),
    name: String(s.name ?? "").trim(),
    lat: Number(s.lat),
    lon: Number(s.lon),
  }))
  .filter((s: any) => s.stationcode && s.name); // garde seulement les noms non vides

fs.writeFileSync(dst, JSON.stringify({ value: out }, null, 2), "utf-8");
console.log(`✅ ${out.length} stations écrites dans ${dst}`);
