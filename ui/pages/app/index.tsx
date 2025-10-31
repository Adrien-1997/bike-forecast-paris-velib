// ui/pages/app/index.tsx
import Head from "next/head";

export default function AppIndexPage() {
  return (
    <>
      <Head>
        <title>Vélo Paris App — Disponibilités et prévisions</title>
        <meta
          name="description"
          content="Consultez les disponibilités et prévisions courtes des stations Vélib’ à Paris."
        />
      </Head>

      {/* ✅ Iframe interne : charge la version sans chrome */}
      <main
        className="main app-wrapper"
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "stretch",
          padding: "0",
          height: "calc(100vh - 120px)", // ajuste selon la hauteur de ton header/footer
        }}
      >
        <iframe
          src="/app/embed"
          title="Vélo Paris App"
          style={{
            flex: 1,
            border: "none",
            width: "100%",
            height: "100%",
          }}
          loading="lazy"
          referrerPolicy="no-referrer"
        />
      </main>
    </>
  );
}
