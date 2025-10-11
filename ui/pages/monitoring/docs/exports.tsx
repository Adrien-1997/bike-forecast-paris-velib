// ui/pages/docs/exports.tsx
import Head from "next/head";
import type { GetStaticProps } from "next";
import Link from "next/link";
import { getDoc } from "@/lib/services/monitoring";

type DocBlock =
  | { type: "heading"; level?: 1 | 2 | 3 | 4; text: string }
  | { type: "paragraph"; text: string }
  | { type: "list"; ordered?: boolean; items: string[] }
  | { type: "link"; text: string; href: string }
  | { type: "table"; headers: string[]; rows: (string | number | null)[][] }
  | { [k: string]: any };

type Props = {
  blocks: DocBlock[];
  generatedAt: string;
};

export const revalidate = 600;

export const getStaticProps: GetStaticProps<Props> = async () => {
  const payload = (await getDoc("exports")) ?? {};
  const blocks: DocBlock[] = Array.isArray((payload as any).blocks)
    ? (payload as any).blocks
    : Array.isArray(payload)
    ? (payload as any)
    : [];
  return {
    props: { blocks, generatedAt: new Date().toISOString() },
    revalidate,
  };
};

export default function ExportsPage({ blocks, generatedAt }: Props) {
  return (
    <>
      <Head>
        <title>Docs — Exports</title>
        <meta
          name="description"
          content="Liste des exports (fichiers, destinations GCS/BigQuery, schémas)."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1000, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Exports</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Page ISR : {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>
          <Link
            href="/monitoring"
            style={{
              padding: "8px 12px",
              borderRadius: 8,
              border: "1px solid #e5e7eb",
              background: "white",
              textDecoration: "none",
              color: "inherit",
            }}
          >
            ← Monitoring
          </Link>
        </header>

        <section style={{ marginTop: 20 }}>
          {(!blocks || !blocks.length) && (
            <div className="small" style={{ opacity: 0.7 }}>
              Aucun contenu n’a été renvoyé par <code>/monitoring/docs/exports</code>.
            </div>
          )}

          {blocks?.map((b, i) => <Block key={i} block={b} />)}
        </section>
      </main>
    </>
  );
}

/* ─────────────── Renderer commun aux pages docs ─────────────── */

function Block({ block }: { block: any }) {
  if (!block || typeof block !== "object") return null;

  switch (block.type) {
    case "heading": {
      const L = Number(block.level ?? 2);
      const txt = String(block.text ?? "");
      if (L <= 1) return <h1 style={{ marginTop: 24 }}>{txt}</h1>;
      if (L === 2) return <h2 style={{ marginTop: 22 }}>{txt}</h2>;
      if (L === 3) return <h3 style={{ marginTop: 20 }}>{txt}</h3>;
      return <h4 style={{ marginTop: 18 }}>{txt}</h4>;
    }

    case "paragraph": {
      return (
        <p style={{ lineHeight: 1.6, marginTop: 10 }}>
          {String(block.text ?? "")}
        </p>
      );
    }

    case "list": {
      const items: any[] = Array.isArray(block.items) ? block.items : [];
      if (block.ordered) {
        return (
          <ol style={{ paddingLeft: 22, marginTop: 10 }}>
            {items.map((it, i) => (
              <li key={i} style={{ marginBottom: 4 }}>
                {String(it)}
              </li>
            ))}
          </ol>
        );
      }
      return (
        <ul style={{ paddingLeft: 22, marginTop: 10 }}>
          {items.map((it, i) => (
            <li key={i} style={{ marginBottom: 4 }}>
              {String(it)}
            </li>
          ))}
        </ul>
      );
    }

    case "link": {
      const href = String(block.href ?? "#");
      const text = String(block.text ?? href);
      return (
        <p style={{ marginTop: 10 }}>
          <a href={href} target="_blank" rel="noopener noreferrer">
            {text}
          </a>
        </p>
      );
    }

    case "table": {
      const headers: string[] = Array.isArray(block.headers) ? block.headers : [];
      const rows: any[][] = Array.isArray(block.rows) ? block.rows : [];
      return (
        <div style={{ overflowX: "auto", marginTop: 12 }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {headers.map((h, i) => (
                  <th key={i} align="left">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {r.map((c, j) => (
                    <td key={j}>{cell(c)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    default: {
      return (
        <pre
          style={{
            background: "#0b1220",
            color: "#d1d5db",
            padding: "1rem",
            borderRadius: 8,
            overflowX: "auto",
            marginTop: 12,
          }}
        >
          {JSON.stringify(block, null, 2)}
        </pre>
      );
    }
  }
}

function cell(v: unknown) {
  if (v == null) return "—";
  if (typeof v === "string" || typeof v === "number") return String(v);
  return JSON.stringify(v);
}
