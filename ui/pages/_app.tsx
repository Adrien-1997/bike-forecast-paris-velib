// ui/pages/_app.tsx
import type { AppProps } from "next/app";
import Head from "next/head";

// Only keep a minimal global reset/tokens here.
// DO NOT import app.css or monitoring.css globally.
import "@/styles/globals.css";

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        {/* keep meta light and generic; page titles live in each page */}
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="color-scheme" content="dark light" />
        <meta name="theme-color" content="#0b1220" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}