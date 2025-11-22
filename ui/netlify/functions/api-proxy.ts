// netlify/functions/api-proxy.ts
import type { Handler, HandlerEvent } from "@netlify/functions";

const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/$/, "");
const API_PREFIX = (process.env.API_PREFIX || "").replace(/^\/|\/$/g, "");
const BEARER = process.env.API_GLOBAL_TOKEN ? `Bearer ${process.env.API_GLOBAL_TOKEN}` : "";

function buildForwardHeaders(src: Record<string, string | undefined> | undefined): Headers {
  const h = new Headers();
  if (!src) return h;

  for (const [k, v] of Object.entries(src)) {
    if (!v) continue;
    const kl = k.toLowerCase();
    // retire hop-by-hop / host
    if (kl === "host" || kl === "connection" || kl === "transfer-encoding") continue;
    h.set(k, v);
  }

  // Toujours demander du JSON
  if (!h.has("accept")) h.set("accept", "application/json");

  // ⚠️ Empêche la double compression
  h.set("accept-encoding", "identity");

  // 🔐 Injecte le token si absent côté client
  if (!h.has("authorization") && BEARER) {
    h.set("authorization", BEARER);
  }

  return h;
}

export const handler: Handler = async (event: HandlerEvent) => {
  try {
    if (!BASE) {
      return { statusCode: 500, body: JSON.stringify({ error: "CLOUD_RUN_BASE not set" }) };
    }

    const fnPrefix = "/.netlify/functions/api-proxy";
    const rawPath = event.path || "/";
    const passthrough = rawPath.startsWith(fnPrefix) ? rawPath.slice(fnPrefix.length) : rawPath;
    const qs = event.rawQuery ? `?${event.rawQuery}` : "";
    const upstreamUrl = `${BASE}${API_PREFIX ? `/${API_PREFIX}` : ""}${passthrough}${qs}`;

    const method = (event.httpMethod || "GET").toUpperCase();
    const headers = buildForwardHeaders(event.headers);

    const body =
      event.body && method !== "GET" && method !== "HEAD"
        ? (event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body)
        : undefined;

    console.log("[proxy] →", method, upstreamUrl);

    const res = await fetch(upstreamUrl, { method, headers, body: body as any, redirect: "manual" });
    const ab = await res.arrayBuffer();
    const buf = Buffer.from(ab);

    // Entêtes propres pour le client
    const out: Record<string, string> = {};
    res.headers.forEach((v, k) => {
      const kl = k.toLowerCase();
      if (kl === "transfer-encoding" || kl === "connection" || kl === "content-encoding") return;
      if (kl === "content-length") return;
      out[k] = v;
    });
    out["Content-Length"] = String(buf.length);

    if (!res.ok) {
      const peek = buf.toString("utf8").slice(0, 1000);
      console.error(`[proxy] upstream ${res.status} ${res.statusText} — body:`, peek);
    }

    return {
      statusCode: res.status,
      headers: out,
      body: buf.toString("base64"),
      isBase64Encoded: true,
    };
  } catch (e: any) {
    console.error("[proxy] ERROR", e);
    return { statusCode: 502, body: JSON.stringify({ error: "proxy_error", detail: String(e?.message || e) }) };
  }
};
