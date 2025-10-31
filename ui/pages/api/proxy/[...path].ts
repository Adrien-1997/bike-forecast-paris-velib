// /ui/pages/api/proxy/[...path].ts
import type { NextApiRequest, NextApiResponse } from "next";

/*─────────────────────────────── Config depuis l'env ───────────────────────────────*/

const API_BASE = process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "")!;
if (!API_BASE) {
  throw new Error("NEXT_PUBLIC_API_BASE is not set");
}

// Jeton privé, non exposé au client
const API_TOKEN = process.env.API_TOKEN;
if (!API_TOKEN) {
  console.warn("[proxy] API_TOKEN is not set — calls will be unauthenticated");
}

// Personnalisation optionnelle
const AUTH_HEADER = process.env.API_AUTH_HEADER || "Authorization";
const TOKEN_PREFIX = process.env.API_TOKEN_PREFIX ?? "Bearer ";
const DEFAULT_TIMEOUT_MS = Number(process.env.API_PROXY_TIMEOUT_MS ?? "30000");

// Désactive le bodyParser pour pouvoir lire le corps brut
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
  const query = req.url?.split("?")[1];
  return `${API_BASE}/${path}${query ? `?${query}` : ""}`;
}

function filterIncomingHeaders(req: NextApiRequest): Record<string, string> {
  const h: Record<string, string> = {};

  if (req.headers["content-type"]) h["Content-Type"] = String(req.headers["content-type"]);
  h["Accept"] = req.headers["accept"]
    ? String(req.headers["accept"])
    : "application/json";

  if (req.headers["if-none-match"]) h["If-None-Match"] = String(req.headers["if-none-match"]);
  if (req.headers["if-modified-since"])
    h["If-Modified-Since"] = String(req.headers["if-modified-since"]);

  // Ajoute le token privé côté serveur
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
    p.then((v) => {
      clearTimeout(id);
      resolve(v);
    }).catch((e) => {
      clearTimeout(id);
      reject(e);
    });
  });
}

/*─────────────────────────────── Handler principal ───────────────────────────────*/

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const url = buildUpstreamUrl(req);
  const method = (req.method || "GET").toUpperCase();

  let body: Buffer | undefined;
  if (!["GET", "HEAD"].includes(method)) {
    try {
      body = await readRawBody(req);
    } catch (e: any) {
      res.status(400).json({ error: "Invalid request body", detail: String(e?.message || e) });
      return;
    }
  }

  const headers = filterIncomingHeaders(req);
  const bodyInit = body ? new Uint8Array(body) : undefined; // ✅ conversion propre

  try {
    const upstream = await withTimeout(
      fetch(url, {
        method,
        headers,
        body: bodyInit,
        redirect: "manual",
      }),
      DEFAULT_TIMEOUT_MS
    );

    res.status(upstream.status);

    // Headers utiles
    const ct = upstream.headers.get("content-type");
    if (ct) res.setHeader("Content-Type", ct);

    const etag = upstream.headers.get("etag");
    if (etag) res.setHeader("ETag", etag);

    const lastMod = upstream.headers.get("last-modified");
    if (lastMod) res.setHeader("Last-Modified", lastMod);

    res.setHeader("Cache-Control", "no-store, private, max-age=0");

    // Réponses sans corps
    if (method === "HEAD" || upstream.status === 304) {
      res.end();
      return;
    }

    const buf = Buffer.from(await upstream.arrayBuffer());
    res.setHeader("Content-Length", String(buf.length));
    res.send(buf);
  } catch (e: any) {
    console.error("[proxy] upstream error →", e);
    res.status(502).json({
      error: "Bad gateway",
      detail: String(e?.message || e),
      upstream: url,
    });
  }
}
