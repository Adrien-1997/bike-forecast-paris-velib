// ui/pages/monitoring/_probe.tsx
import React, { useEffect, useState } from "react";

type Json = any;

const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8081";

// ── util: résume les clés/typos d'un objet ou d'un tableau d'objets
function summarizeSchema(v: Json): Record<string, string> | string {
  if (Array.isArray(v)) {
    const sample = v[0];
    if (!sample || typeof sample !== "object") return "array";
    return summarizeSchema(sample) as Record<string, string>;
  }
  if (v && typeof v === "object") {
    const out: Record<string, string> = {};
    for (const k of Object.keys(v)) {
      const val = v[k];
      const t = Array.isArray(val) ? "array" : typeof val;
      out[k] = t;
    }
    return out;
  }
  return typeof v;
}

function PrettyJSON({ data }: { data: Json }) {
  return (
    <pre className="text-xs leading-relaxed overflow-auto p-3 rounded-xl bg-slate-900/50 ring-1 ring-slate-700 text-slate-200">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

export default function MonitoringProbe() {
  const [manifest, setManifest] = useState<Json | null>(null);
  const [perfDaily15, setPerfDaily15] = useState<Json | null>(null);
  const [perfDaily60, setPerfDaily60] = useState<Json | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    (async () => {
      const errs: string[] = [];
      async function fetchJson(url: string) {
        try {
          const r = await fetch(url, { headers: { "cache-control": "no-cache" } });
          if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
          return await r.json();
        } catch (e: any) {
          errs.push(`${url} → ${e.message || e}`);
          return null;
        }
      }
      const m = await fetchJson(`${base}/monitoring/manifest`);
      const p15 = await fetchJson(`${base}/monitoring/model/perf/daily?h=15`);
      const p60 = await fetchJson(`${base}/monitoring/model/perf/daily?h=60`);

      setManifest(m);
      setPerfDaily15(p15);
      setPerfDaily60(p60);
      setErrors(errs);
      // Log console pour copier-coller si besoin
      // eslint-disable-next-line no-console
      console.groupCollapsed("Monitoring Probe");
      // eslint-disable-next-line no-console
      console.log("manifest", m);
      // eslint-disable-next-line no-console
      console.log("perf/daily?h=15", p15);
      // eslint-disable-next-line no-console
      console.log("perf/daily?h=60", p60);
      // eslint-disable-next-line no-console
      console.groupEnd();
    })();
  }, []);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold text-slate-100">Monitoring — Probe</h1>
      <p className="text-slate-400 text-sm">
        Base API: <code className="px-1 py-0.5 rounded bg-slate-800 ring-1 ring-slate-700">{base}</code>
      </p>

      {errors.length > 0 && (
        <div className="rounded-xl bg-red-900/30 ring-1 ring-red-700 p-3 text-sm text-red-200">
          <div className="font-medium mb-1">Erreurs de fetch</div>
          <ul className="list-disc pl-5 space-y-1">
            {errors.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </div>
      )}

      <section className="space-y-2">
        <h2 className="text-lg font-medium text-slate-100">/monitoring/manifest</h2>
        {manifest && (
          <>
            <div className="text-xs text-slate-300">Schema (sample):</div>
            <PrettyJSON data={summarizeSchema(manifest)} />
            <div className="text-xs text-slate-300">Payload:</div>
            <PrettyJSON data={manifest} />
          </>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium text-slate-100">/monitoring/model/perf/daily?h=15</h2>
        {perfDaily15 && (
          <>
            <div className="text-xs text-slate-300">Schema (sample):</div>
            <PrettyJSON data={summarizeSchema(perfDaily15)} />
            <div className="text-xs text-slate-300">Payload (first 5):</div>
            <PrettyJSON data={Array.isArray(perfDaily15) ? perfDaily15.slice(0, 5) : perfDaily15} />
          </>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium text-slate-100">/monitoring/model/perf/daily?h=60</h2>
        {perfDaily60 && (
          <>
            <div className="text-xs text-slate-300">Schema (sample):</div>
            <PrettyJSON data={summarizeSchema(perfDaily60)} />
            <div className="text-xs text-slate-300">Payload (first 5):</div>
            <PrettyJSON data={Array.isArray(perfDaily60) ? perfDaily60.slice(0, 5) : perfDaily60} />
          </>
        )}
      </section>

      <p className="text-xs text-slate-400">
        Une fois validé, on dérive des **types précis** (TS) à partir de ces schémas réels, puis on remplace les `any` dans les services.
      </p>
    </div>
  );
}
