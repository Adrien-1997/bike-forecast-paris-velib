// ui/pages/monitoring/pipeline.tsx
import Head from "next/head";
import type { GetStaticProps } from "next";
import { getManifest } from "@/lib/services/monitoring";
import type { MonitoringManifest, MonitoringManifestItem } from "@/lib/types";

type Job = {
  id: string;
  label: string;
  desc: string;
  updated: string | null; // <-- jamais undefined
};

type Props = {
  manifest: MonitoringManifest | null;
  jobs: Job[];
  generatedAt: string;
};

export const revalidate = 600;

export const getStaticProps: GetStaticProps<Props> = async () => {
  const manifest = await getManifest();

  const jobsBase: Omit<Job, "updated">[] = [
    { id: "ingest",     label: "Ingest",        desc: "Snapshots 5 min (bronze)" },
    { id: "compact",    label: "Compact daily", desc: "Compactage vers silver" },
    { id: "features",   label: "Features 4h",   desc: "Fenêtres de features (serving)" },
    { id: "train",      label: "Training",      desc: "Entraînement modèles LightGBM" },
    { id: "forecast",   label: "Forecast",      desc: "Prédictions T+15 / T+60" },
    { id: "monitoring", label: "Monitoring",    desc: "Indicateurs perf / drift" },
  ];

  const items: MonitoringManifestItem[] = manifest?.items ?? [];

  const mapUpdated = (needle: string): string | null => {
    const matches = items.filter((it) => it.path?.includes(needle));
    if (!matches.length) return null;
    matches.sort((a, b) => {
      const va = Number(a.updated || 0);
      const vb = Number(b.updated || 0);
      if (va !== vb) return vb - va;
      return String(b.updated_iso).localeCompare(String(a.updated_iso));
    });
    return matches[0]?.updated_iso ?? (matches[0]?.updated ? String(matches[0].updated) : null);
  };

  const jobs: Job[] = jobsBase.map((j) => ({
    ...j,
    updated: mapUpdated(j.id) ?? null, // <-- force null
  }));

  return {
    props: {
      manifest: manifest ?? null,
      jobs,
      generatedAt: new Date().toISOString(),
    },
    revalidate,
  };
};

function toParis(iso?: string | null) {
  if (!iso) return "—";
  const s = iso.endsWith("Z") ? iso : `${iso}Z`;
  const d = new Date(s);
  return new Intl.DateTimeFormat("fr-FR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/Paris",
  }).format(d);
}

export default function PipelinePage({ jobs, manifest, generatedAt }: Props) {
  return (
    <>
      <Head>
        <title>Monitoring — Pipeline</title>
        <meta
          name="description"
          content="Suivi des tâches du pipeline (ingest, compact, build features, train, forecast, monitoring)."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1000, margin: "0 auto" }}>
        <h1>Pipeline</h1>
        <p style={{ opacity: 0.7 }}>
          Manifest généré : {manifest?.generated_at ? toParis(manifest.generated_at) : "—"} · Page ISR : {toParis(generatedAt)}
        </p>

        <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 16 }}>
          <thead>
            <tr>
              <th align="left">Étape</th>
              <th align="left">Description</th>
              <th align="left">Dernière exécution (UTC)</th>
              <th align="left">Heure Paris</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td style={{ fontWeight: 600 }}>{job.label}</td>
                <td>{job.desc}</td>
                <td>{job.updated ?? "—"}</td>
                <td>{toParis(job.updated)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <section style={{ marginTop: 28 }}>
          <details>
            <summary style={{ cursor: "pointer" }}>Voir le manifest brut</summary>
            <pre
              style={{
                background: "#0b1220",
                color: "#d1d5db",
                padding: "1rem",
                borderRadius: 8,
                overflowX: "auto",
                maxHeight: "50vh",
                marginTop: 8,
              }}
            >
              {JSON.stringify(manifest ?? {}, null, 2)}
            </pre>
          </details>
        </section>
      </main>
    </>
  );
}
