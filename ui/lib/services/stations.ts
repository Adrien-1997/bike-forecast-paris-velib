// ui/lib/services/stations.ts
//
// =============================================================================
// Service frontend pour récupérer la **liste des stations** et leurs
// **métadonnées + état courant** depuis l’API.
//
// Rôle :
// - Appeler l’endpoint `/stations` (backend unifié).
// - Normaliser la forme des objets pour coller au type `Station` partagé
//   (`ui/lib/types.ts`).
// - Renvoyer un tableau typé `Station[]` prêt à être consommé par :
//     • la carte (Leaflet),
//     • la liste principale des stations,
//     • les hooks comme `useAppData`,
//     • les modules de monitoring.
//
// Contraintes :
// - Ce module ne fait **aucune** transformation métier au-delà de la
//   normalisation des types.
// - Aucune dépendance à React : service pur.
// - Tolérance maximale : en cas d’erreur réseau, renvoie `[]` plutôt que de
//   casser l’UI. Les champs non convertibles sont ramenés à `undefined`,
//   jamais `null`, pour rester compatibles avec le type `Station`.
// =============================================================================

import { getJSON } from "@/lib/http";
import type { Station } from "@/lib/types/types";

/* ------------------------------------------------------------------------- */
/* Utils                                                                     */
/* ------------------------------------------------------------------------- */

/**
 * Convertit une valeur inconnue en `number | undefined`.
 *
 * - Accepte un `number` fini.
 * - Accepte une string numérique convertible proprement.
 * - Retourne `undefined` sinon (jamais `null`) pour rester compatible
 *   avec les champs optionnels de `Station` (ex: `lat?: number`).
 */
function toNum(x: unknown): number | undefined {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string") {
    const n = Number(x);
    return Number.isFinite(n) ? n : undefined;
  }
  return undefined;
}

/* ------------------------------------------------------------------------- */
/* Service principal                                                         */
/* ------------------------------------------------------------------------- */

/**
 * Récupère la liste complète des stations depuis l’API `/stations`.
 *
 * Schéma backend typique (exemple simplifié) :
 * [
 *   {
 *     "station_id": "10042",
 *     "name": "République",
 *     "lat": 48.8674,
 *     "lon": 2.3632,
 *     "capacity": 32,
 *     "num_bikes_available": 12,
 *     "num_docks_available": 20
 *   },
 *   ...
 * ]
 *
 * Tolérance :
 * - `station_id` ou `stationcode` sont acceptés comme identifiants.
 * - Les valeurs non numériques (lat/lon/capacity/availability) sont
 *   converties en `undefined` plutôt qu’en `null`.
 *
 * Retour :
 * - Un tableau `Station[]` toujours défini (jamais `null`).
 * - En cas d’erreur réseau / JSON invalide : `[]`.
 */
export async function getStations(): Promise<Station[]> {
  const arr = await getJSON<any[]>("/stations").catch(() => []);
  const rows: Station[] = [];

  for (const raw of arr) {
    const r: any = raw ?? {};

    // Identifiant : station_id prioritaire, sinon fallback sur stationcode
    const sid = r.station_id ?? r.stationId ?? r.stationcode ?? r.code;
    const station_id = sid != null ? String(sid) : "";
    if (!station_id) continue; // on ignore les entrées sans id exploitable

    rows.push({
      station_id,
      name: r.name ?? undefined,
      lat: toNum(r.lat),
      lon: toNum(r.lon),
      capacity: toNum(r.capacity),
      num_bikes_available: toNum(r.num_bikes_available),
      num_docks_available: toNum(r.num_docks_available),
    });
  }

  return rows;
}
