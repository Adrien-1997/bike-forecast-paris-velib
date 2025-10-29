// ui/lib/local/stationsIndex.ts
export type StationMeta = { name: string; lat?: number; lon?: number; capacity?: number | null };

export async function loadStationsIndexFromArrayJson(path: string) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fail fetch ${path}`);
  const raw = await res.json();
  const arr: any[] = Array.isArray(raw) ? raw : raw.value ?? [];
  const index: Record<string, StationMeta> = {};

  for (const s of arr) {
    const rawId = String(s.station_id ?? s.stationcode ?? s.stationCode ?? s.code ?? "");
    if (!rawId) continue;
    const meta: StationMeta = {
      name: typeof s.name === "string" ? s.name : "",
      lat: s.lat,
      lon: s.lon,
      capacity: s.capacity ?? null,
    };
    // id tel quel
    index[rawId] = meta;
    // id sans zéros de tête (au cas où)
    const noZeros = rawId.replace(/^0+/, "");
    if (!(noZeros in index)) index[noZeros] = meta;
  }
  return index;
}
