// scripts/buildStationsIndex.ts
import fs from "fs";

const src = "ui/public/data/station_information.json"; // ton JSON GBFS source
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
