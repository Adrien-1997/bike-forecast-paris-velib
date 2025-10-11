// ui/pages/monitoring/index.tsx
import Head from "next/head";
import Link from "next/link";
import type { GetStaticProps } from "next";
import { getManifest, getPerfDaily } from "@/lib/services/monitoring";
import type { MonitoringManifest, PerfDailyResponse } from "@/lib/types";

type Props = {
  manifest: MonitoringManifest | null;
  daily15: PerfDailyResponse | null;
  daily60: PerfDailyResponse | null;
  generatedAt: string;
};

export const getStaticProps: GetStaticProps<Props> = async () => {
  // Manifest sécurisé
  let manifest = null;
  try {
    manifest = await getManifest(); // peut revenir null si timeout/404/500
  } catch {
    manifest = null;
  }

  // Daily h=15 sécurisé
  let daily15 = null;
  try {
    daily15 = await getPerfDaily(15);
  } catch {
    daily15 = null;
  }

  // Ne PAS charger 60 ici
  return {
    props: {
      manifest,
      daily15,
      daily60: null,
      generatedAt: new Date().toISOString(),
    },
    revalidate: 600,
  };
};

export default function MonitoringHome({
  manifest,
  daily15,
  daily60,
  generatedAt,
}: Props) {
  const h60Available = !!daily60;
  const h15Days = daily15?.metrics?.length ?? 0;

  // Helpers lecture manifest (tolérant au manifest null)
  const findLatest = (
  cat: keyof NonNullable<MonitoringManifest["summary"]>
  ) => {
  if (!manifest?.summary) return [] as { base: string; latest_updated: string }[];
  const group = (manifest.summary as any)?.[cat] as
      | { resources: { base: string; latest_updated: string; latest: string }[] }
      | undefined;
  return group?.resources?.map((r) => ({
      base: r.base,
      latest_updated: r.latest_updated,
    })) ?? [];
  };

  const modelPerf = findLatest("model_perf");
  const drift = findLatest("drift");
  const network = findLatest("network");

  return (
    <>
      <Head>
        <title>Monitoring — Overview</title>
        <meta
          name="description"
          content="Vue d’ensemble du monitoring Vélib’ : manifest, données disponibles, et accès rapide aux modules."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Monitoring — Overview</h1>
                <div className="small" style={{ opacity: 0.7 }}>
                Manifest généré: {manifest?.generated_at
                    ? new Date(manifest.generated_at).toLocaleString("fr-FR")
                    : "indisponible"}
                {" "}— UI build: {new Date(generatedAt).toLocaleString("fr-FR")}
                </div>
                <div className="small" style={{ opacity: 0.7 }}>
                Root: <code>{manifest?.root ?? "—"}</code> — items: {manifest?.total_items ?? 0}
                </div>
          </div>

          <nav style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Link
              href="/monitoring/perf"
              style={pill(true)}
              title="Performance du modèle (daily)"
            >
              Model Perf
            </Link>
            <Link
              href="/monitoring/drift"
              style={pill(!!drift?.length)}
              aria-disabled={!drift?.length}
              onClick={(e) => {
                if (!drift?.length) e.preventDefault();
              }}
              title={!drift?.length ? "Drift indisponible" : "Distribution & PSI/KS"}
            >
              Drift
            </Link>
            <Link
              href="/monitoring/data-health"
              style={pill(true)}
            >
              Data Health
            </Link>
            <Link
              href="/monitoring/network/overview"
              style={pill(!!network?.length)}
              aria-disabled={!network?.length}
              onClick={(e) => {
                if (!network?.length) e.preventDefault();
              }}
              title={!network?.length ? "Réseau indisponible" : "KPIs réseau"}
            >
              Network
            </Link>
            <Link href="/monitoring/explain" style={pill(false)} aria-disabled title="À venir">
              Explain
            </Link>
          </nav>
        </header>

        {/* KPIs rapides */}
        <section style={{ marginTop: 24 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
              gap: 12,
            }}
          >
            <Kpi label="Daily h=15 — jours" value={h15Days} />
            <Kpi
              label="Daily h=60 — status"
              value={h60Available ? "disponible" : "absent"}
              intent={h60Available ? "success" : "warning"}
            />
            <Kpi
              label="Model Perf — fichiers"
              value={modelPerf?.length ?? 0}
            />
            <Kpi
              label="Network — fichiers"
              value={network?.length ?? 0}
            />
          </div>
        </section>

        {/* Résumé Manifest par catégorie */}
        <section style={{ marginTop: 28 }}>
          <h2 style={{ margin: "12px 0" }}>Dernières mises à jour</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 16 }}>
            <Card title="Model Perf" items={modelPerf} />
            <Card title="Drift" items={drift} />
            <Card title="Network" items={network} />
          </div>
        </section>

        {/* Lien outil probe */}
        <section style={{ marginTop: 28 }}>
          <div
            style={{
              border: "1px dashed #374151",
              padding: 12,
              borderRadius: 12,
              background: "rgba(15,23,42,0.3)",
            }}
          >
            <div style={{ fontSize: 14, color: "#93C5FD", marginBottom: 6 }}>
              Dev tool
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <Link href="/monitoring/_probe" style={anchorLike()}>
                Ouvrir le Probe (manifest & daily)
              </Link>
              <span style={{ opacity: 0.6, fontSize: 12 }}>
                Vérifie le schéma réel avant d’ajouter de nouvelles pages.
              </span>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}

/* ───────────── UI helpers ───────────── */

function pill(enabled: boolean): React.CSSProperties {
  return {
    padding: "6px 10px",
    borderRadius: 999,
    border: "1px solid #d1d5db",
    background: enabled ? "#111827" : "white",
    color: enabled ? "white" : "#111827",
    pointerEvents: enabled ? "auto" : "none",
    opacity: enabled ? 1 : 0.5,
    textDecoration: "none",
    fontSize: 14,
  };
}

function anchorLike(): React.CSSProperties {
  return {
    color: "#60A5FA",
    textDecoration: "underline",
    cursor: "pointer",
  };
}

function Kpi({
  label,
  value,
  intent = "default",
}: {
  label: string;
  value: number | string | null | undefined;
  intent?: "default" | "success" | "warning" | "danger";
}) {
  const ring =
    intent === "success"
      ? "#10B981"
      : intent === "warning"
      ? "#F59E0B"
      : intent === "danger"
      ? "#EF4444"
      : "#334155";
  return (
    <div
      style={{
        border: `1px solid ${ring}`,
        background: "rgba(15,23,42,0.5)",
        borderRadius: 12,
        padding: "10px 12px",
      }}
    >
      <div style={{ fontSize: 12, color: "#9CA3AF", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 22, color: "#E5E7EB", fontWeight: 600, marginTop: 2 }}>
        {value ?? "—"}
      </div>
    </div>
  );
}

function Card({
  title,
  items,
}: {
  title: string;
  items?: { base: string; latest_updated: string }[];
}) {
  return (
    <div
      style={{
        border: "1px solid #374151",
        borderRadius: 12,
        padding: 12,
        background: "rgba(15,23,42,0.4)",
        minHeight: 140,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 6 }}>{title}</div>
      {items && items.length > 0 ? (
        <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 1.6 }}>
          {items.slice(0, 5).map((r) => (
            <li key={r.base} style={{ fontSize: 14 }}>
              {r.base} —{" "}
              <span style={{ opacity: 0.7 }}>
                {new Date(r.latest_updated).toLocaleString("fr-FR")}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <div style={{ opacity: 0.6, fontSize: 14 }}>Aucun fichier récent</div>
      )}
    </div>
  );
}
