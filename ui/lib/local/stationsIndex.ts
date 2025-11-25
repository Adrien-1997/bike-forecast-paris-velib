// ui/lib/local/stationsIndex.ts
//
// =============================================================================
// Helpers pour charger un index local de stations Vélib’ depuis un JSON.
//
// Rôle :
// - Lire un fichier JSON (array d’objets) via `fetch(path)` côté client.
// - Construire un index `Record<string, StationMeta>` clé → métadonnées station.
// - Gérer plusieurs conventions de noms de colonnes pour l’identifiant de station
//   (station_id, stationcode, stationCode, code…).
// - Ajouter des aliases d’identifiants sans zéros de tête (ex: "00012" → "12").
//
// Utilisation typique :
// - précharger un petit JSON de métadonnées de stations côté frontend
//   (nom, lat/lon, capacité) pour alimenter cartes / listes.
// =============================================================================

export type StationMeta = {
  /** Nom lisible de la station. */
  name: string;
  /** Latitude (optionnelle). */
  lat?: number;
  /** Longitude (optionnelle). */
  lon?: number;
  /** Capacité totale de la station (nombre de bornes). */
  capacity?: number | null;
};

/**
 * Charge un index de stations depuis un JSON "array-like".
 *
 * Paramètres :
 * - path : URL ou chemin relatif vers le JSON, ex:
 *     "/static/stations_index.json"
 *
 * Format attendu :
 * - le JSON est soit :
 *     • un tableau : [{ station_id, name, lat, lon, capacity, ... }, ...]
 *     • ou un objet avec une clé `value` contenant ce tableau.
 *
 * Normalisation :
 * - l’identifiant de station est pioché dans l’une des clés suivantes :
 *     station_id, stationcode, stationCode, code
 * - pour chaque station, on crée deux entrées dans l’index :
 *     • la clé brute (rawId)
 *     • la même clé sans zéros de tête (noZeros), si différente
 */
export async function loadStationsIndexFromArrayJson(path: string) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fail fetch ${path}`);

  const raw = await res.json();
  const arr: any[] = Array.isArray(raw) ? raw : raw.value ?? [];
  const index: Record<string, StationMeta> = {};

  for (const s of arr) {
    // Identifiant brut, en essayant les différentes conventions possibles
    const rawId = String(s.station_id ?? s.stationcode ?? s.stationCode ?? s.code ?? "");
    if (!rawId) continue;

    const meta: StationMeta = {
      name: typeof s.name === "string" ? s.name : "",
      lat: s.lat,
      lon: s.lon,
      capacity: s.capacity ?? null,
    };

    // 1) indexé sous l’ID tel quel
    index[rawId] = meta;

    // 2) indexé sous l’ID sans zéros de tête (si différent)
    const noZeros = rawId.replace(/^0+/, "");
    if (!(noZeros in index)) index[noZeros] = meta;
  }

  return index;
}
