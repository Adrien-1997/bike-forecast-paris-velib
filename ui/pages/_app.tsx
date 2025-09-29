import type { AppProps } from "next/app";
import "@/styles.css"; // optionnel
export default function App({ Component, pageProps }: AppProps) { return <Component {...pageProps} />; }
