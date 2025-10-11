// ui/pages/monitoring/explain.tsx
import Head from "next/head";
import dynamic from "next/dynamic";
import type { GetStaticProps } from "next";
import { getModelExplainability } from "@/lib/services/monitoring";

// Plotly chargé côté client
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type ExplainFeature = {
  feature: string;
  importance?: number;
  shap_mean?: number;
  shap_std?: number;
};

type Props = {
  features: ExplainFeature[];
  generatedAt: string;
};

export const revalidate = 600;

export const getStaticProps: GetStaticProps<Props> = async () => {
  const explain = (await getModelExplainability()) ?? [];
  const features: ExplainFeature[] = Array.isArray(explain) ? explain : [];
  return {
    props: {
      features,
      generatedAt: new Date().toISOString(),
    },
    revalidate,
  };
};

function fmt(v?: number | null, d = 3) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "—";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d }).format(x);
}

export default function ExplainPage({ features, generatedAt }: Props) {
  const hasData = Array.isArray(features) && features.length > 0;
  const top10 = hasData
    ? [...features]
        .sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0))
        .slice(0, 10)
    : [];

  return (
    <>
      <Head>
        <title>Monitoring — Explainability</title>
        <meta
          name="description"
          content="Analyse de l’importance des features (global SHAP)."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1000, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Explainability</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Généré : {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>
        </header>

        {!hasData ? (
          <div style={{ marginTop: 20, opacity: 0.7 }}>
            <p>
              Aucune donnée d’explicabilité n’a encore été publiée.
              <br />
              Cette page se branchera sur l’endpoint{" "}
              <code>/monitoring/model/explainability</code> dès qu’il sera exposé.
            </p>
          </div>
        ) : (
          <>
            <section style={{ marginTop: 24 }}>
              <h2 style={{ margin: "12px 0" }}>Top features (importance moyenne)</h2>
              <Plot
                data={[
                  {
                    type: "bar",
                    orientation: "h",
                    x: top10.map((f) => f.importance),
                    y: top10.map((f) => f.feature),
                    marker: { color: "rgb(31, 119, 180)" },
                  } as any,
                ]}
                layout={{
                  autosize: true,
                  height: 400,
                  margin: { l: 140, r: 20, t: 20, b: 40 },
                  xaxis: { title: { text: "Importance moyenne (|SHAP|)" } }, // ← FIX
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
            </section>

            <section style={{ marginTop: 32 }}>
              <h2 style={{ margin: "12px 0" }}>Table complète</h2>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th align="left">Feature</th>
                      <th align="right">Importance</th>
                      <th align="right">SHAP mean</th>
                      <th align="right">SHAP std</th>
                    </tr>
                  </thead>
                  <tbody>
                    {features.map((f) => (
                      <tr key={f.feature}>
                        <td>{f.feature}</td>
                        <td align="right">{fmt(f.importance, 4)}</td>
                        <td align="right">{fmt(f.shap_mean, 4)}</td>
                        <td align="right">{fmt(f.shap_std, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </main>
    </>
  );
}
