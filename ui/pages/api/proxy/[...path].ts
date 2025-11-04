// /ui/pages/api/proxy/[...path].ts
import type { NextApiRequest, NextApiResponse } from "next";

/*─────────────────────────────── Config privée (serveur) ───────────────────────────────*/
const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/+$/, "");
const API_TOKEN = process.env.API_TOKEN || "";
const AUTH_HEADER = process.env.API_AUTH_HEADER || "Authorization";
const TOKEN_PREFIX = process.env.API_TOKEN_PREFIX ?? "Bearer ";
const DEFAULT_TIMEOUT_MS = Number(process.env.API_PROXY_TIMEOUT_MS ?? "30000");

// Désactive le bodyParser (on forward le corps tel quel)
export const config = {
  api: {
    bodyParser: false,
    externalResolver: true,
  },
};

/*─────────────────────────────── Helpers ───────────────────────────────*/
function readRawBody(req: NextApiRequest): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function buildUpstreamUrl(req: NextApiRequest) {
  const segs = Array.isArray(req.query.path)
    ? req.query.path
    : [req.query.path].filter(Boolean) as string[];
  const path = segs.join("/");
  const q = req.url?.split("?")[1];
  return `${BASE}/${path}${q ? `?${q}` : ""}`;
}

function filterIncomingHeaders(req: NextApiRequest): Record<string, string> {
  const h: Record<string, string> = {};

  // Types acceptés
  const ct = req.headers["content-type"];
  if (ct) h["Content-Type"] = String(ct);
  h["Accept"] = req.headers["accept"] ? String(req.headers["accept"]) : "application/json";

  // Caching conditionnel safe
  if (req.headers["if-none-match"]) h["If-None-Match"] = String(req.headers["if-none-match"]);
  if (req.headers["if-modified-since"])
    h["If-Modified-Since"] = String(req.headers["if-modified-since"]);

  // Pas d'Authorization entrant (évite l'injection)
  // Ajoute le jeton privé serveur si dispo
  if (API_TOKEN) {
    if (AUTH_HEADER.toLowerCase() === "authorization") {
      h["Authorization"] = `${TOKEN_PREFIX}${API_TOKEN}`;
    } else {
      h[AUTH_HEADER] = `${TOKEN_PREFIX}${API_TOKEN}`;
    }
  }

  return h;
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = setTimeout(() => reject(new Error("Upstream timeout")), ms);
    p.then((v) => { clearTimeout(id); resolve(v); })
     .catch((e) => { clearTimeout(id); reject(e); });
  });
}

/*─────────────────────────────── Handler principal ───────────────────────────────*/
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Config manquante → réponse générique (ne pas exposer les valeurs)
  if (!BASE) {
    res.status(500).json({ error: "Upstream not configured" });
    return;
  }

  const method = (req.method || "GET").toUpperCase();
  const url = buildUpstreamUrl(req);

  let body: Buffer | undefined;
  if (!["GET", "HEAD"].includes(method)) {
    try {
      body = await readRawBody(req);
    } catch {
      res.status(400).json({ error: "Invalid request body" });
      return;
    }
  }

  const headers = filterIncomingHeaders(req);
  const bodyInit = body ? new Uint8Array(body) : undefined;

  try {
    const upstream = await withTimeout(
      fetch(url, { method, headers, body: bodyInit, redirect: "manual" }),
      DEFAULT_TIMEOUT_MS
    );

    // Propage code + entêtes utiles, sans exposer quoi que ce soit de sensible
    res.status(upstream.status);
    const ct = upstream.headers.get("content-type");
    if (ct) res.setHeader("Content-Type", ct);

    const etag = upstream.headers.get("etag");
    if (etag) res.setHeader("ETag", etag);

    const lastMod = upstream.headers.get("last-modified");
    if (lastMod) res.setHeader("Last-Modified", lastMod);

    // Pas de cache côté CDN/Client via ce proxy
    res.setHeader("Cache-Control", "no-store, private, max-age=0");

    if (method === "HEAD" || upstream.status === 304) {
      res.end();
      return;
    }

    const buf = Buffer.from(await upstream.arrayBuffer());
    res.setHeader("Content-Length", String(buf.length));
    res.send(buf);
  } catch {
    // Réponse minimale, sans echo d’URL ni de détails serveur
    res.status(502).json({ error: "Bad gateway" });
  }
}
