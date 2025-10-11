// ui/pages/_app.tsx
import type { AppProps } from 'next/app';
import Head from 'next/head';
import '@/styles/globals.css';
import 'leaflet/dist/leaflet.css';

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="color-scheme" content="dark light" />
        <meta name="theme-color" content="#0b1220" />
        <title>Vélib’ Paris — Forecast</title>
      </Head>
      <Component {...pageProps} />
    </>
  );
}
