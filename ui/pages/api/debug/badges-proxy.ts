// pages/api/debug/badges-proxy.ts
import type { NextApiRequest, NextApiResponse } from 'next';

const UPSTREAM = `${process.env.NEXT_PUBLIC_API_BASE}/badges?mode=latest`;

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (!process.env.NEXT_PUBLIC_API_BASE) {
    return res.status(500).json({ error: 'NEXT_PUBLIC_API_BASE not set' });
  }
  const url = `${UPSTREAM}&_ts=${Date.now()}`; // anti-cache
  try {
    const upstream = await fetch(url, {
      // Force no cache pour voir la vérité du serveur
      cache: 'no-store',
      headers: {
        'accept': 'application/json',
        // ajoute ici éventuels tokens: Authorization: `Bearer ${process.env.API_TOKEN}`,
      },
    });

    const rawText = await upstream.text();
    let parsed: any = null;
    try { parsed = JSON.parse(rawText); } catch { /* pas JSON */ }

    const headersObj = Object.fromEntries(upstream.headers.entries());

    res.status(200).json({
      probe: {
        requested_url: url,
        status: upstream.status,
        statusText: upstream.statusText,
        headers: headersObj,
        is_json: parsed !== null,
      },
      body: parsed ?? rawText,
    });
  } catch (e: any) {
    res.status(500).json({ error: String(e) });
  }
}
