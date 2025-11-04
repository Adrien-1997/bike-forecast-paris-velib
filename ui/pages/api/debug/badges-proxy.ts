// pages/api/debug/badges-proxy.ts
import type { NextApiRequest, NextApiResponse } from "next";

const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/+$/, "");
const API_TOKEN = process.env.API_TOKEN || ""; // injecté côté Netlify, jamais exposé

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (!BASE) {
    // Ne divulgue pas le nom de la variable manquante
    return res.status(500).json({ error: "Upstream not configured" });
  }

  const url = `${BASE}/badges?mode=latest&_ts=${Date.now()}`;

  try {
    const upstream = await fetch(url, {
      method: "GET",
      headers: {
        accept: "application/json",
        ...(API_TOKEN ? { authorization: `Bearer ${API_TOKEN}` } : {}),
      },
      // on force le fetch frais pour un endpoint de debug
      cache: "no-store",
    });

    // On forward le status et le JSON sans écho de l’URL ni des headers
    const text = await upstream.text();
    let body: any;
    try {
      body = JSON.parse(text);
    } catch {
      // Si ce n’est pas du JSON, renvoie un wrapper minimal (toujours sans secrets)
      return res
        .status(upstream.status)
        .json({ ok: upstream.ok, status: upstream.status, body: text.slice(0, 2000) });
    }

    return res.status(upstream.status).json(body);
  } catch (e: any) {
    // Message générique pour éviter toute fuite d’info
    return res.status(502).json({ error: "Upstream fetch failed" });
  }
}
