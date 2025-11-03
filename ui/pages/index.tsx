// ui/pages/index.tsx
import Script from "next/script";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import Head from "next/head";
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import { getMonitoringIntro, type IntroDoc } from "@/lib/services/monitoring/intro";
import { getOverviewSnapshotMap, type OverviewSnapshotMap } from "@/lib/services/monitoring/network_overview";
import { chartLayout, chartConfig } from "@/lib/plotlyTheme";
import { getDataHealthCoverageByHour, type CoverageByHourRow } from "@/lib/services/monitoring/data_health";
import type * as Plotly from "plotly.js";
import { getPerformanceLiftCurve, type LiftCurve } from "@/lib/services/monitoring/model_performance";


/* Plotly (client only) */
const Plot = dynamic(() => import("react-plotly.js").then(m => m.default), {
  ssr: false,
  loading: () => <div className="empty">Chargement…</div>,
});


export default function LandingPage() {
  // ────────────────────────────────────────────────────────────────────────────
  // Refs
  const demoIframeRef = useRef<HTMLIFrameElement | null>(null);
  const embedWrapRef = useRef<HTMLDivElement | null>(null);

  const year = useMemo(() => new Date().getFullYear(), []);

  // 🔐 Liens de paiement
  const STRIPE_DON_5 = process.env.NEXT_PUBLIC_STRIPE_DON_5 ?? "";
  const STRIPE_DON_10 = process.env.NEXT_PUBLIC_STRIPE_DON_10 ?? "";
  const STRIPE_DON_20 = process.env.NEXT_PUBLIC_STRIPE_DON_20 ?? "";
  const STRIPE_MONTHLY_5 = process.env.NEXT_PUBLIC_STRIPE_MONTHLY_5 ?? "";

  function getCssVar(name: string, fallback: string) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch {
      return fallback;
    }
  }

  // ────────────────────────────────────────────────────────────────────────────
  // LoadingBar (même logique que monitoring, simple succès)
  const [loading] = useState<boolean>(false);
  const [error] = useState<string | null>(null);
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  // ────────────────────────────────────────────────────────────────────────────
  // KPIs intro (réels via API monitoring)
  const [intro, setIntro] = useState<IntroDoc | null>(null);
  const [introError, setIntroError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const doc = await getMonitoringIntro();
        if (!alive) return;
        setIntro(doc ?? null);
      } catch (e: any) {
        if (!alive) return;
        setIntroError(String(e?.message ?? e));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const fmtDateTime = (iso?: string | null) => (iso ? new Date(iso).toLocaleString("fr-FR") : null);
  const generatedAt = fmtDateTime(intro?.generated_at) ?? null;
  const modelVersions = intro?.kpis?.model_versions ?? "h15 / h60";

  // Injection des valeurs dans la KPI bar animée
  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    const coverage = intro?.kpis?.coverage_7d_pct ?? 98;
    const freshP95 = intro?.kpis?.freshness_p95_min ?? 5;
    const stations = intro?.kpis?.stations_active ?? 1400;
    const psi = intro?.kpis?.psi_global ?? 0.3;

    const nodes = document.querySelectorAll<HTMLElement>(".kpi-card .kpi__value");
    if (nodes[0]) nodes[0].dataset.count = `${Number(coverage).toFixed(0)}%`;
    if (nodes[1]) nodes[1].dataset.count = `${Number(freshP95).toFixed(0)}`;
    if (nodes[2]) nodes[2].dataset.count = `${Number(stations).toFixed(0)}`;
    if (nodes[3]) nodes[3].dataset.count = `${Number(psi).toFixed(2)}`;

    if (prefersReduced) {
      nodes.forEach((el) => el.dataset.count && (el.textContent = el.dataset.count));
      return;
    }

    const ease = (t: number) => 1 - Math.pow(1 - t, 4);
    const animateCount = (el: HTMLElement, to: number, suffix = "") => {
      const start = performance.now();
      const dur = 1100 + Math.random() * 600;
      const from = 0;
      const step = (now: number) => {
        const p = Math.min(1, (now - start) / dur);
        const v = Math.round((from + (to - from) * ease(p)) * 10) / 10;
        el.textContent = suffix ? v + suffix : String(v);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };

    document.querySelectorAll<HTMLElement>(".kpi-card .kpi__value").forEach((el) => {
      const raw = el.dataset.count;
      if (!raw) return;
      const isPct = raw.trim().endsWith("%");
      const to = parseFloat(raw);
      if (Number.isFinite(to)) animateCount(el, to, isPct ? "%" : "");
    });
  }, [intro]);

  // Deuxième passe d'animation (fallback robustesse)
  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    const ease = (t: number) => 1 - Math.pow(1 - t, 4);
    const animateCount = (el: HTMLElement, to: number, suffix = "") => {
      const start = performance.now();
      const dur = 1100 + Math.random() * 600;
      const from = 0;
      const step = (now: number) => {
        const p = Math.min(1, (now - start) / dur);
        const v = Math.round((from + (to - from) * ease(p)) * 10) / 10;
        el.textContent = suffix ? v + suffix : String(v);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };

    document.querySelectorAll<HTMLElement>(".kpi-card .kpi__value").forEach((el) => {
      const raw = el.dataset.count;
      if (!raw) return;

      if (prefersReduced) {
        el.textContent = raw;
        return;
      }

      const isPct = raw.trim().endsWith("%");
      const to = parseFloat(raw);
      if (Number.isFinite(to)) animateCount(el, to, isPct ? "%" : "");
    });
  }, []);

  // ────────────────────────────────────────────────────────────────────────────
  // Snapshot map (Overview) — state + load
  const [snapMap, setSnapMap] = useState<OverviewSnapshotMap | null>(null);

  // Couverture par heure (Data Health)
  const [covHour, setCovHour] = useState<CoverageByHourRow[] | null>(null);

  // Lift quotidien (h=60)
  const [lift60, setLift60] = useState<LiftCurve | null>(null);

  useEffect(() => {
    let alive = true;
    getOverviewSnapshotMap()
      .then((doc) => {
        if (!alive) return;
        setSnapMap(doc ?? null);
      })
      .catch(() => {
        if (!alive) return;
        setSnapMap(null);
      });
    return () => {
      alive = false;
    };
  }, []);

  // Fetch couverture par heure
    useEffect(() => {
      let alive = true;
      getDataHealthCoverageByHour()
        .then((doc) => { if (alive) setCovHour(doc ?? null); })
        .catch(() => { if (alive) setCovHour(null); });
      return () => { alive = false; };
    }, []);

    const coverageBarData = useMemo(() => {
    const arr = covHour ?? [];
    if (!arr.length) return [];
    return [{
      x: arr.map(r => `${String(r.hour).padStart(2,"0")}h`),
      y: arr.map(r => Number(r.coverage_pct)),
      type: "bar" as const,
      name: "Couverture (%)",
      hovertemplate: "%{x} — %{y:.1f}%<extra></extra>",
    }];
  }, [covHour]);


  // Fetch lift quotidien (h=60)
  useEffect(() => {
    let alive = true;
    getPerformanceLiftCurve(60)
      .then((doc) => { if (alive) setLift60(doc ?? null); })
      .catch(() => { if (alive) setLift60(null); });
    return () => { alive = false; };
  }, []);

  const liftDailyData = useMemo(() => {
    const pts = lift60?.points ?? [];
    if (!pts.length) return [];
    return [{
      x: pts.map(p => p.date),
      y: pts.map(p => Number.isFinite(Number(p.lift_vs_baseline)) ? Number(p.lift_vs_baseline) * 100 : null),
      type: "scatter" as const,
      mode: "lines",
      name: "Lift vs baseline (%)",
      connectgaps: false,
      hovertemplate: "%{x} — %{y:.1f}%<extra></extra>",
    }];
  }, [lift60]);

  // ────────────────────────────────────────────────────────────────────────────
  // Démo (iframe) : lancement manuel + plein écran wrapper + skeleton piloté par state
  const [demoLaunched, setDemoLaunched] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [showSkeleton, setShowSkeleton] = useState<boolean>(false);

  // Au load de l’iframe, on coupe le skeleton via state (pas de remove())
  useEffect(() => {
    const frame = demoIframeRef.current;
    if (!frame) return;
    const onLoad = () => setShowSkeleton(false);
    frame.addEventListener("load", onLoad);
    return () => frame.removeEventListener("load", onLoad);
  }, []);

  // Suivre les changements de plein écran sur le wrapper
  useEffect(() => {
    const onFs = () => setIsFullscreen(document.fullscreenElement === embedWrapRef.current);
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  // Actions
  const handleLaunch = () => {
    if (demoLaunched) return;
    setShowSkeleton(true);
    setDemoLaunched(true); // l'iframe reçoit src via JSX (pas de mutation DOM directe)
  };

  const handleReload = () => {
    const frame = demoIframeRef.current;
    if (!frame || !demoLaunched) return;
    const url = frame.src || "/app/embed";
    setShowSkeleton(true);
    frame.src = "";
    setTimeout(() => {
      frame.src = url;
    }, 60);
  };

  const handleEnterFullscreen = () => {
    const wrap = embedWrapRef.current;
    if (!wrap) return;
    wrap.classList.add("is-fs"); // active le plein écran “interne”
    setIsFullscreen(true);
  };

  const handleExitFullscreen = () => {
    const wrap = embedWrapRef.current;
    if (!wrap) return;
    wrap.classList.remove("is-fs");
    setIsFullscreen(false);
  };

  // Header (ancres + liens app/monitoring)
  const headerItems = [
    { label: "Démo", href: "#demo" },
    { label: "Fonctions", href: "#features" },
    { label: "Monitoring", href: "#monitoring" },
    { label: "Architecture", href: "#how" },
    { label: "FAQ", href: "#faq" },
    { label: "Soutenir", href: "#support" },
  ];

  // Auto-hide header
  useEffect(() => {
    const header = document.querySelector<HTMLElement>(".site-header");
    if (!header) return;

    header.classList.add("autohide");
    let prev = window.scrollY;

    const onScroll = () => {
      const y = window.scrollY;
      const goingDown = y > prev && y > 10;
      header.classList.toggle("is-hidden", goingDown);
      prev = y;
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Auto-slide KPI bar
  useEffect(() => {
    const root = document.querySelector<HTMLElement>(".kpi-bar.kpi-bar--auto");
    const track = root?.querySelector<HTMLElement>(".kpi-track");
    if (!root || !track) return;

    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) return;

    const items = Array.from(track.children) as HTMLElement[];
    const N = items.length;
    if (N === 0) return;

    const firstClone = items[0].cloneNode(true) as HTMLElement;
    track.appendChild(firstClone);

    const getOffsets = () => {
      const rects = Array.from(track.children).map((el) => (el as HTMLElement).offsetLeft);
      const base = rects[0] || 0;
      return rects.map((x) => x - base);
    };

    let offsets = getOffsets();
    let index = 0;
    let holdMs = 3000;
    let slideMs = 380;
    let timer: number | null = null;

    (root.style as any).overflow = "hidden";
    track.style.willChange = "transform";

    const applyTransform = (i: number, withTransition: boolean) => {
      track.style.transition = withTransition ? `transform ${slideMs}ms ease` : "none";
      const x = offsets[Math.min(i, offsets.length - 1)] || 0;
      track.style.transform = `translateX(${-x}px)`;
    };

    const onVis = () => {
      if (document.hidden) {
        if (timer) window.clearTimeout(timer);
        timer = null;
      } else {
        scheduleNext();
      }
    };

    const ro = new ResizeObserver(() => {
      const currentX = offsets[Math.min(index, offsets.length - 1)] || 0;
      offsets = getOffsets();
      track.style.transition = "none";
      track.style.transform = `translateX(${-currentX}px)`;
    });
    ro.observe(track);

    const onEnter = () => {
      if (timer) window.clearTimeout(timer);
      timer = null;
    };
    const onLeave = () => scheduleNext();

    const goNext = () => {
      applyTransform(index + 1, true);

      const onEnd = () => {
        track.removeEventListener("transitionend", onEnd);
        if (index + 1 >= N) {
          index = 0;
          applyTransform(0, false);
        } else {
          index += 1;
        }
        scheduleNext();
      };

      track.addEventListener("transitionend", onEnd, { once: true });
    };

    const scheduleNext = () => {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(goNext, holdMs) as unknown as number;
    };

    applyTransform(0, false);
    scheduleNext();

    document.addEventListener("visibilitychange", onVis);
    root.addEventListener("mouseenter", onEnter);
    root.addEventListener("mouseleave", onLeave);

    return () => {
      if (timer) window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVis);
      root.removeEventListener("mouseenter", onEnter);
      root.removeEventListener("mouseleave", onLeave);
      ro.disconnect();
      track.style.transition = "";
      track.style.transform = "";
      track.style.willChange = "";
      try {
        track.lastElementChild === firstClone && track.removeChild(firstClone);
      } catch {}
    };
  }, []);

  return (
    <>
      <Head>
        {/* ===== Base meta ===== */}
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Vélo Paris — Carte en direct & Prévisions +15 / +60 min</title>
        <meta
          name="description"
          content="Vélo Paris anticipe la disponibilité des stations Vélib’ à +15 et +60 minutes grâce à un pipeline Cloud Run / FastAPI / Next.js."
        />
        <meta name="theme-color" content="#0b1220" />
        <meta name="color-scheme" content="dark light" />

        {/* ===== Canonical / robots ===== */}
        <link rel="canonical" href="https://example.com/velib-forecast/" />
        <meta name="robots" content="index,follow,max-image-preview:large" />

        {/* ===== OpenGraph / Twitter ===== */}
        <meta property="og:title" content="Vélo Paris — Carte en direct & Prévisions" />
        <meta
          property="og:description"
          content="Anticipez la disponibilité des stations à +15 et +60 minutes. Live map, monitoring réseau, data & modèle."
        />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://example.com/velib-forecast/" />
        <meta property="og:image" content="https://example.com/velib-forecast/cover.jpg" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Vélib’ Forecast Paris — Prévisions +15/+60" />
        <meta name="twitter:description" content="Carte temps réel & prévisions, pipelines Cloud Run." />
        <meta name="twitter:image" content="https://example.com/velib-forecast/cover.jpg" />

        {/* ===== Perf ===== */}
        <link
          rel="preconnect"
          href="https://velib-ui-160046094975.europe-west1.run.app"
          crossOrigin=""
        />
        <link rel="dns-prefetch" href="https://velib-ui-160046094975.europe-west1.run.app" />

        {/* ===== Leaflet CSS (pour SnapshotMap) ===== */}
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          crossOrigin=""
        />

        {/* ===== JSON-LD ===== */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebSite",
              name: "Vélo Paris",
              url: "https://example.com/velib-forecast/",
              description:
                "Carte temps réel des stations Vélib’ avec prévisions +15/+60 et monitoring (réseau/data/modèle).",
              inLanguage: "fr-FR",
              publisher: { "@type": "Organization", name: "Vélo PAris" },
              potentialAction: {
                "@type": "SearchAction",
                target: "https://example.com/velib-forecast/?q={query}",
                "query-input": "required name=query",
              },
            }),
          }}
        />

        {/* Z-index de sécurité pour que le widget soit au-dessus */}
        <style jsx global>{`
          .floatingchat-container { z-index: 10000 !important; }
        `}</style>
      </Head>

      {/* ===== A11y skip link ===== */}
      <a href="#demo" className="sr-only">Aller au contenu principal</a>

      {/* Header global */}
      <GlobalHeader items={headerItems} brandHref="/" />

      {/* ====================== CONTENT ====================== */}
      <div className="monitoring">
        <main id="top" className="page" style={{ paddingTop: "var(--header-offset, 10px)" }}>
          {/* Loading bar homogène */}
          <LoadingBar status={barStatus} />
          {error && <div className="banner banner--error mt-2">{error}</div>}

          {/* ====================== HERO ====================== */}
          <section className="panel hero" aria-labelledby="hero-title">
            <div className="container hero-grid">
              <div>
                <div className="eyebrow">
                  <span className="ping" aria-hidden="true" />
                  <span className="chip" aria-label="Horizon de prévision">Prévisions +15 / +60 min • Paris</span>
                  <span className="chip" aria-label="Actualisation">Données live 5 min</span>
                </div>

                <h1 id="hero-title">
                  Vélo Paris
                  <br />
                  <span
                    style={{
                      background: "linear-gradient(90deg,var(--primary),var(--primary-2))",
                      WebkitBackgroundClip: "text",
                      backgroundClip: "text",
                      color: "transparent",
                    }}
                  >
                    Cartographie, prévisions et monitoring
                  </span>
                  .
                </h1>

                <p className="lead">
                  Vélo Paris anticipe la disponibilité des stations Vélib’ à +15 et +60 minutes grâce à un pipeline complet :
                  ingestion en continu des flux GBFS dans Cloud Storage, enrichissement météo et temporel, modélisation XGBoost,
                  API FastAPI et interface Next.js déployées sur Cloud Run Jobs.
                  Un projet pensé pour la lisibilité urbaine, la fiabilité des données et la performance technique.
                </p>

                <ul className="text-muted" style={{ margin: "10px 0 0", paddingLeft: 18 }}>
                  <li>Carte interactive optimisée pour la fluidité et la lisibilité, même à grande échelle.</li>
                  <li>Prévisions issues d’un modèle XGBoost calibré sur les tendances horaires, la météo et les profils de station.</li>
                  <li>Monitoring complet du réseau, des données et du modèle : fraîcheur, complétude, dérive (PSI) et stabilité des prédictions.</li>
                </ul>

                <div className="cta">
                  <a className="btn" href="#demo" aria-label="Aller à la démo en direct">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <path d="M7 7h10v10H7z" stroke="white" strokeWidth="1.8" />
                      <path d="M3 3v6M3 3h6M21 21v-6M21 21h-6" stroke="white" strokeWidth="1.8" />
                    </svg>
                    Voir la démo
                  </a>
                  <a className="btn outline" href="/monitoring">Monitoring</a>
                  <a className="btn outline" href="/app">Lancer l’app</a>
                </div>

                <div className="tech-chips">
                  <span className="chip">Cloud Storage</span>
                  <span className="chip">Cloud Run Jobs</span>
                  <span className="chip">FastAPI</span>
                  <span className="chip">Next.js</span>
                  <span className="chip">XGBoost</span>
                </div>
              </div>

              <aside className="glass hero-card" aria-label="Indicateurs clés">
                <h3>En chiffres</h3>

                {/* KPI BAR — auto-slide */}
                <div className="kpi-bar-wrap">
                  <div className="kpi-bar kpi-bar--scroll kpi-bar--auto kpi-bar--dense" role="list">
                    <div className="kpi-track">
                      <div className="kpi-card" role="listitem">
                        <div className="kpi__label">Observations couvertes</div>
                        <div className="kpi__row">
                          <div className="kpi__value" data-count="98%">0%</div>
                        </div>
                      </div>

                      <div className="kpi-card" role="listitem">
                        <div className="kpi__label">Fraîcheur (p95)</div>
                        <div className="kpi__row">
                          <div className="kpi__value" data-count="5">0</div>
                          <span className="kpi__hint">min</span>
                        </div>
                      </div>

                      <div className="kpi-card" role="listitem">
                        <div className="kpi__label">Stations suivies</div>
                        <div className="kpi__row">
                          <div className="kpi__value" data-count="1400">0</div>
                        </div>
                      </div>

                      <div className="kpi-card is-muted" role="listitem">
                        <div className="kpi__label">Drift (7j)</div>
                        <div className="kpi__row">
                          <div className="kpi__value" data-count="0.3">0</div>
                          <span className="kpi-delta kpi-delta--ok">stable</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {!introError && generatedAt && (
                  <p className="meta-line" aria-live="polite">
                    Mise à jour monitoring : {generatedAt}
                  </p>
                )}

                <div className="embed">
                  <div className="ratio">
                    <img src="/img/preview-map.webp" alt="Carte Vélo Paris – aperçu statique" loading="lazy" />
                  </div>
                </div>

                <ul className="text-muted" style={{ margin: "12px 0 0", paddingLeft: 18 }}>
                  <li>MAE vs baseline persistance — par station et tranche horaire.</li>
                  <li>Exports JSON réutilisables : kpis, maps, résidus, calibration, importance features.</li>
                </ul>
              </aside>
            </div>
          </section>

          {/* ====================== DEMO (iframe) ====================== */}
          <section id="demo" className="panel" aria-labelledby="demo-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="demo-title">Démo en direct</h2>
                  <p>
                    Application React embarquée : carte en direct, recherche de stations, et prévisions à +15 minutes.
                    Le premier accès peut prendre quelques secondes (cold start Cloud Run).
                  </p>
                </div>
                <div className="actions-row" style={{ gap: "0.5rem" }}>
                  {!demoLaunched ? (
                    <button className="btn" type="button" onClick={handleLaunch} aria-label="Lancer la démo">
                      Lancer la démo
                    </button>
                  ) : (
                    <>
                      <button className="btn" type="button" onClick={handleEnterFullscreen} aria-label="Plein écran">
                        Plein écran
                      </button>
                      <button className="btn outline" type="button" onClick={handleReload} aria-label="Recharger la démo">
                        Recharger
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Wrapper en plein écran (inclut l’iframe + la croix) */}
              <div className="embed" aria-live="polite" ref={embedWrapRef} style={{ position: "relative" }}>
                {isFullscreen && (
                <button
                  type="button"
                  onClick={handleExitFullscreen}
                  aria-label="Quitter le plein écran"
                  title="Quitter le plein écran"
                  style={{
                    position: "absolute",
                    top: "max(12px, env(safe-area-inset-top) + 15px)",
                    left: "max(12px, env(safe-area-inset-left) + 15px)",
                    zIndex: 3,
                    border: "none",
                    borderRadius: "50%",
                    width: "42px",
                    height: "42px",
                    background: "color-mix(in srgb, var(--panel) 70%, black)",
                    color: "var(--text, #fff)",
                    cursor: "pointer",
                    fontSize: "2rem",
                    lineHeight: 1,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: "0 4px 16px rgba(0,0,0,.3)",
                    backdropFilter: "blur(4px)",
                  }}
                >
                  ×
                </button>
                )}

                {showSkeleton && (
                  <div
                    className="skeleton"
                    id="skeleton"
                    role="status"
                    aria-live="polite"
                    aria-atomic="true"
                    style={{ position: "absolute", inset: 0 }}
                  >
                    {demoLaunched ? "Initialisation de la démo…" : "Cliquez sur « Lancer la démo » pour démarrer"}
                  </div>
                )}

                <iframe
                  ref={demoIframeRef}
                  title="Vélo Paris — Application"
                  src={demoLaunched ? "/app/embed" : ""}
                  loading="lazy"
                  allow="fullscreen; clipboard-read; clipboard-write"
                  referrerPolicy="no-referrer-when-downgrade"
                  aria-hidden={demoLaunched ? undefined : true}
                  style={{ width: "100%", height: "100%", border: 0, display: "block" }}
                />

              </div>

              {/* Actions sous le frame */}
              <div className="actions-row">
                <a className="btn" href="/app" target="_blank" rel="noopener">Ouvrir dans un onglet</a>
                <a className="btn outline" href="#features">Découvrir les fonctions</a>
              </div>

              <div className="glass prose mt-2">
                <h3>Pourquoi c’est fluide ?</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Cloud Run UI/API séparés, connexions keep-alive et caches HTTP.</li>
                  <li>Préchargement DNS, lazy-loading et fragmentation maîtrisée.</li>
                  <li>RUM (web-vitals) & métriques UX pour piloter l’expérience perçue.</li>
                </ul>
              </div>
            </div>
          </section>

          {/* ====================== FEATURES ====================== */}
          <section id="features" className="panel" aria-labelledby="features-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="features-title">Du live à la décision</h2>
                  <p>Repérez les stations utiles, anticipez à +15/+60, comparez à l’historique, basculez en monitoring.</p>
                </div>
                <a className="btn outline" href="/app">Essayer maintenant</a>
              </div>

              <div className="features">
                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M12 7v10M7 12h10" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Carte lisible & rapide</h3>
                  <p>Couleurs sobres, étiquettes claires, clustering équilibré, recherche instantanée et focus proximité.</p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M3 12a9 9 0 1018 0A9 9 0 003 12z" stroke="currentColor" strokeWidth="2" />
                      <path d="M12 7v6l4 2" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Prévisions +15/+60</h3>
                  <p>LightGBM avec features calendrier/météo, calibration légère et segments horaires pour limiter les biais.</p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M4 4h16v16H4z" stroke="currentColor" strokeWidth="2" />
                      <path d="M4 9h16M9 4v16" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Comparaisons utiles</h3>
                  <p>“Aujourd’hui vs médiane” et profils horaires par station pour comprendre les dynamiques locales.</p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M4 12h16" stroke="currentColor" strokeWidth="2" />
                      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Monitoring intégré</h3>
                  <p>KPIs fraîcheur/complétude, dérive simple (PSI), résidus, QQ/ACF, calibration & incertitude.</p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M7 7h10v10H7z" stroke="currentColor" strokeWidth="2" />
                      <path d="M3 3v6M3 3h6M21 21v-6M21 21h-6" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Intégrable partout</h3>
                  <p>
                    Un simple <code>&lt;iframe&gt;</code> suffit (sous-domaine/app proxy). Thème auto (clair/sombre), navigation
                    clavier, CORS & headers sûrs.
                  </p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Pensé pour évoluer</h3>
                  <p>Nouveaux horizons, nouvelles villes, nouvelles sources — sans refonte : pipeline modulaire & contrats JSON.</p>
                </article>
              </div>

              <div className="glass prose mt-2">
                <h3>Cas d’usage rapides</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Info voyageurs (entreprises/quartiers) : carte intégrée.</li>
                  <li>Immobilier/événementiel : repérer zones sous-/sur-servies.</li>
                  <li>Mobilité individuelle : planifier avec station d’arrivée fiable.</li>
                </ul>
              </div>
            </div>
          </section>

          {/* ====================== MONITORING ====================== */}
          <section id="monitoring" className="panel" aria-labelledby="monitoring-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="monitoring-title">Monitoring & Qualité (réseau, data, modèle)</h2>
                  <p>
                    Exports JSON versionnés sur Cloud Storage : kpis.json, snapshot_map.json, station_health.json,
                    drift_summary.json, residuals.json, calibration.json, uncertainty.json, feature_importance.json…
                  </p>
                </div>
                <a className="btn outline" href="/monitoring">Ouvrir le monitoring</a>
              </div>

              <div className="monitoring-showcase">
                {/* Carte principale */}
                <figure className="card card--map">
                  <figcaption className="cap" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
                      <strong>Snapshot réseau</strong>
                      {snapMap?.generated_at && (
                        <span>
                          {snapMap.generated_at
                            .replace("T", " ")
                            .replace("Z", "")
                            .replace(/\.\d+/, "")}
                        </span>
                      )}
                    </div>

                    <a
                      className="btn sm outline"
                      href="/monitoring/network/overview"
                      aria-label="Voir les détails du réseau"
                    >
                      Détails du réseau
                    </a>
                  </figcaption>

                  <div className="map-wrap">
                    {snapMap?.rows?.length ? (
                      <>
                        <SnapshotMap rows={snapMap.rows} />
                        <div className="cluster-legend" aria-hidden>
                          <div className="title">État du réseau</div>
                          <div className="row"><span className="dot dot--pen" /> <span>Pénurie</span></div>
                          <div className="row"><span className="dot dot--sat" /> <span>Saturation</span></div>
                          <div className="row"><span className="dot dot--ok"  /> <span>OK</span></div>
                        </div>
                      </>
                    ) : (
                      <div className="empty" style={{ height: "100%", display: "grid", placeItems: "center" }}>
                        Snapshot indisponible.
                      </div>
                    )}
                  </div>
                </figure>

                {/* Couverture par heure (moyenne) */}
                <figure className="card plot-card">
                  <figcaption className="cap" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <div>
                      <strong>Couverture par heure</strong>
                      <span style={{ marginLeft: 8 }}>Moyenne récente (%)</span>
                    </div>

                    <a
                      className="btn sm outline"
                      href="/monitoring/data/health"
                      aria-label="Voir les détails de la qualité des données"
                    >
                      Détails qualité des données
                    </a>
                  </figcaption>

                  {coverageBarData.length ? (
                    <div className="plot-wrap">
                      <Plot
                        data={coverageBarData as Plotly.Data[]}
                        layout={chartLayout({
                          autosize: true,
                          height: 280,
                          margin: { l: 56, r: 10, t: 6, b: 40 },
                          yaxis: { title: { text: "Couverture (%)" }, range: [0, 100], ticksuffix: "%" },
                          xaxis: { title: { text: "Heure (locale)" } },
                          hovermode: "x unified",
                        })}
                        config={{ ...chartConfig, responsive: true }}
                        useResizeHandler
                        style={{ width: "100%" }}
                        className="plot"
                      />
                    </div>
                  ) : (
                    <div className="empty">Pas de données de couverture horaire.</div>
                  )}
                </figure>

                {/* Lift quotidien (h=60) */}
                <figure className="card plot-card">
                  <figcaption
                    className="cap"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 8,
                    }}
                  >
                    <div>
                      <strong>Lift quotidien</strong>
                      <span style={{ marginLeft: 8 }}>Amélioration vs baseline (h=60)</span>
                    </div>

                    <a
                      className="btn sm outline"
                      href="/monitoring/model/performance"
                      aria-label="Voir les détails de performance du modèle"
                    >
                      Détails performance
                    </a>
                  </figcaption>

                  {liftDailyData.length ? (
                    <div className="plot-wrap">
                      <Plot
                        data={liftDailyData as Plotly.Data[]}
                        layout={chartLayout({
                          autosize: true,
                          height: 280,
                          margin: { l: 56, r: 10, t: 6, b: 40 },
                          yaxis: { title: { text: "Lift (%)" }, zeroline: true },
                          xaxis: { title: { text: "Date (locale)" } },
                          hovermode: "x unified",
                        })}
                        config={{ ...chartConfig, responsive: true }}
                        useResizeHandler
                        style={{ width: "100%" }}
                        className="plot"
                      />
                    </div>
                  ) : (
                    <div className="empty">Pas de courbe de lift disponible.</div>
                  )}
                </figure>

              </div>

              <div className="glass prose mt-2">
                <h3>Ce que l’on suit</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li><strong>Freshness</strong> : p50/p95, trous, hors-plage.</li>
                  <li><strong>Coverage</strong> : % lignes valides, champs critiques, NaN sûrs.</li>
                  <li><strong>Stability</strong> : dérive simple (PSI/KS) sur features clés.</li>
                  <li><strong>Alerts</strong> : pénurie/saturation anormales, outliers horaires.</li>
                </ul>
              </div>
            </div>
          </section>

          {/* ====================== HOW ====================== */}
          <section id="how" className="panel" aria-labelledby="how-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="how-title">Sous le capot</h2>
                  <p>Pipeline reproductible, artefacts versionnés, UI et API découplées.</p>
                </div>
              </div>

              <div className="steps" role="list">
                <div className="step" role="listitem">
                  <span className="chip">1 · Ingestion</span>
                  <strong>GBFS → Cloud Storage (bronze)</strong>
                  <p>Snapshots toutes 5 min, parquet 5-min, compactage journalier, schéma strict station_id+tbin.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Organisation GCS : <code>velib/daily</code>, <code>velib/exports</code>, <code>velib/monitoring</code>.</li>
                    <li>Nettoyage NaN/Inf et timestamps sûrs (UTC/local).</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">2 · Enrichissement</span>
                  <strong>Features calendrier & météo</strong>
                  <p>Sin/cos horaires, jours/semaine/vacances, lags & rollings (1–4 h), tendances & ratios.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Exports JSON prêts-API (sanitization NaN→null).</li>
                    <li>Contrats de schéma versionnés.</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">3 · Modélisation</span>
                  <strong>LightGBM (h15/h60)</strong>
                  <p>Évaluation MAE/WAPE vs baseline persistance, calibration légère, Optuna HPO (GPU Kaggle).</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Artefacts <code>.joblib</code> versionnés (latest + timestamps).</li>
                    <li>Manifests & métriques au format JSON.</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">4 · API</span>
                  <strong>FastAPI (Cloud Run)</strong>
                  <p>Endpoints stations/prévisions/monitoring, ETag/Last-Modified, TTLs et <code>/latest</code> vs <code>?at=</code>.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>CORS limité, headers de sécurité, health checks <code>/ready</code>.</li>
                    <li>Réponses JSON compactes & cacheables.</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">5 · UI</span>
                  <strong>Next.js</strong>
                  <p>Pages App/Monitoring (Leaflet/Plotly), thème auto, UX responsive, animations respect RDM.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Composants : KPI bars, nav sticky, cartes/graphes, tables triables.</li>
                    <li>Perf : lazy, suspense, préconnect/prefetch ciblés.</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">6 · Jobs</span>
                  <strong>Cloud Run Jobs</strong>
                  <p>Jobs Python (build_*), logs Cloud Build, env vars explicites, mémoire/CPU/timeout adaptés.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Images dédiées (pipeline/api/ui) via Artifact Registry.</li>
                    <li>Exports versionnés sous <code>monitoring/*/latest</code> + datés.</li>
                  </ul>
                </div>
              </div>

              <div className="glass prose mt-2">
                <h3>Fiabilité</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Contrats de schéma + tests parsing/horodatage.</li>
                  <li>Sanitization systématique (NaN→null, bornes, types).</li>
                  <li>Monitoring indépendant et réutilisable (JSON-first).</li>
                </ul>
              </div>
            </div>
          </section>

          {/* ====================== FAQ ====================== */}
          <section id="faq" className="panel" aria-labelledby="faq-title">
            <div className="container">
              <div className="sec-head">
                <h2 id="faq-title">FAQ</h2>
              </div>

              <div className="faq-grid">
                {/* Colonne gauche : questions */}
                <div className="faq-questions">
                  <details>
                    <summary>La démo met quelques secondes à démarrer, normal ?</summary>
                    <p>Oui : cold start Cloud Run. Une instance minimale supprime le délai.</p>
                  </details>

                  <details>
                    <summary>Puis-je intégrer l’app dans mon site ?</summary>
                    <p>
                      Oui, via un simple <code>&lt;iframe&gt;</code>. Responsive, thème clair/sombre, navigation clavier et
                      headers de sécurité compatibles.
                    </p>
                  </details>

                  <details>
                    <summary>Comment sont calculées les prévisions ?</summary>
                    <p>
                      Modèles LightGBM avec signaux calendrier/météo, lissages et calibration. Baseline persistance pour
                      mesurer le vrai gain (MAE/WAPE).
                    </p>
                  </details>

                  <details>
                    <summary>Et la qualité des données ?</summary>
                    <p>
                      KPIs fraîcheur/complétude, dérive simple (PSI/KS), résidus & calibration. Exports JSON pour vos
                      tableaux de bord.
                    </p>
                  </details>
                </div>

                {/* Colonne droite : intégration */}
                <aside className="glass prose">
                  <h3>Intégration (Cloud Run)</h3>
                  <p className="text-muted">
                    Intégrez la carte directement dans votre site avec un simple iframe&nbsp;:
                  </p>
                  <pre style={{ whiteSpace: "pre", overflowX: "auto" }}>
                    <code>{`<iframe src="https://velo-paris.fr/app/embed" width="100%" height="68svh" style="border:0"></iframe>`}</code>
                  </pre>
                  <p className="text-muted" style={{ fontSize: ".95rem" }}>
                    Vous pouvez aussi héberger l’app sur un sous-domaine dédié
                    (<em>app.votredomaine.fr</em>) via Cloud Run ou Netlify.
                  </p>
                </aside>
              </div>
            </div>
          </section>

          {/* ====================== SUPPORT ====================== */}
          <section id="support" className="panel" aria-labelledby="support-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="support-title">Soutenir le projet</h2>
                </div>
              </div>

              <div className="grid-2">
                {/* Bio courte */}
                <article className="glass prose">
                  <h3>À propos</h3>
                  <p className="text-muted">
                    Je m’appelle <strong>Adrien</strong>. Ingénieur en mathématiques appliquées, je conçois des outils fondés sur la
                    donnée pour mieux comprendre et anticiper les dynamiques urbaines.  
                    J’ai créé <em>Vélo Paris</em> pour rendre la mobilité plus lisible et accessible, à travers une approche claire,
                    transparente et méthodique.
                  </p>

                  <p className="text-muted">
                    Ce projet indépendant vise à montrer ce que la donnée peut apporter de concret à la vie quotidienne : une vision
                    plus fiable du réseau, une meilleure compréhension des usages, et une capacité à anticiper plutôt qu’à subir.
                    Tout a été pensé pour rester simple à utiliser et solide dans le fond.
                  </p>

                  <p className="text-muted">
                    Je crois qu’une donnée bien traitée peut devenir un repère, une aide à la décision, voire un outil de confiance.
                    C’est cet équilibre entre rigueur et utilité que je cherche à construire, jour après jour, à travers Vélo Paris.
                  </p>

                  <p className="text-muted" style={{ fontSize: ".95rem" }}>
                    Votre soutien contribue à maintenir le projet en ligne, à améliorer la qualité des prévisions et à garantir son
                    indépendance.  
                    <strong>Merci pour votre appui, il fait réellement la différence.</strong>
                  </p>
                </article>


                {/* Cartes de paiement */}
                <div className="support-cards">
                  <figure className="card">
                    <figcaption className="cap">
                      <strong>Soutenir Vélo Paris</strong>
                    </figcaption>

                    <div className="prose text-muted" style={{ marginBottom: "1rem" }}>
                      <p>
                        Votre contribution aide à financer le <strong>développement continu</strong> du projet,
                        l’<strong>hébergement cloud</strong>, la <strong>supervision des données</strong> et la
                        maintenance des modèles de prévision.  
                        Chaque soutien, ponctuel ou régulier, contribue à garder Vélo Paris
                        <strong> libre, fiable et performant</strong>.
                      </p>
                    </div>

                    <div className="actions-col" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                      {/* Ligne 1 : dons uniques */}
                      <div className="actions-row" style={{ flexWrap: "wrap", gap: "0.5rem" }}>
                        <span style={{ flex: "1 1 100%", fontSize: "0.95rem", color: "var(--text-dim)" }}>
                          Contribution unique :
                        </span>
                        <a className="btn" href={STRIPE_DON_5} target="_blank" rel="noopener">
                          5 €
                        </a>
                        <a className="btn outline" href={STRIPE_DON_10} target="_blank" rel="noopener">
                          10 €
                        </a>
                        <a className="btn outline" href={STRIPE_DON_20} target="_blank" rel="noopener">
                          20 €
                        </a>
                      </div>

                      {/* Ligne 2 : soutien mensuel */}
                      <div className="actions-row" style={{ flexWrap: "wrap", gap: "0.5rem" }}>
                        <span style={{ flex: "1 1 100%", fontSize: "0.95rem", color: "var(--text-dim)" }}>
                          Soutien mensuel :
                        </span>
                        <a className="btn" href={STRIPE_MONTHLY_5} target="_blank" rel="noopener">
                          5 €/mois
                        </a>
                      </div>
                    </div>

                    <small className="text-muted" style={{ display: "block", marginTop: "1rem" }}>
                      Paiements gérés par Stripe. Transactions sécurisées et reçus automatiques par e-mail.
                    </small>
                  </figure>
                </div>
              </div>

              {/* Encadré fiscalité / contact */}
              <div className="glass prose mt-2">
                <h3>Transparence & contact</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Les contributions financent l’hébergement, la supervision et l’amélioration continue.</li>
                  <li>Pas de déduction fiscale (sauf mention contraire).</li>
                  <li>
                    Besoin d’un reçu, d’une facture ou d’un partenariat ? Écrivez-moi : <em>adrien.morel@gmail.com</em>.
                  </li>
                </ul>
              </div>
            </div>
          </section>
        </main>
      </div>

      {/* Footer global */}
      <GlobalFooter />
    </>
  );
}

/* ───────────────────────── Mini Map (snapshot) ───────────────────────── */
type MapRow = OverviewSnapshotMap["rows"][number];

const SnapshotMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const { useEffect, useMemo, useState } = await import("react");

  function FitBounds({ rows }: { rows: MapRow[] }) {
    const map = useMap();
    useEffect(() => {
      const pts = rows.filter(
        (r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))
      );
      if (!pts.length) return;
      let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
      for (const r of pts) {
        const la = Number(r.lat), lo = Number(r.lon);
        if (la < minLat) minLat = la;
        if (la > maxLat) maxLat = la;
        if (lo < minLon) minLon = lo;
        if (lo > maxLon) maxLon = lo;
      }
      if (minLat <= maxLat && minLon <= maxLon) {
        map.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [20, 20] });
      }
    }, [rows, map]);
    return null;
  }

  function MapInner({ rows }: { rows: MapRow[] }) {
    const valid = useMemo(
      () => rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))),
      [rows]
    );

    const latMed = valid.length
      ? valid.map((r) => Number(r.lat)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 48.8566;
    const lonMed = valid.length
      ? valid.map((r) => Number(r.lon)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 2.3522;

    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        <MapContainer center={[latMed, lonMed]} zoom={12} style={{ height: "100%", width: "100%", background: "#fff" }}>
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina
          />
          <FitBounds rows={valid} />
          {valid.map((r) => {
            const pen = r.is_penury === 1;
            const sat = r.is_saturation === 1;
            const col = pen ? "#ef4444" : sat ? "#3b82f6" : "#10b981";
            const rad = Math.max(3, Math.min(9, Math.sqrt(Math.max(0, Number(r.bikes ?? 0))) + (sat ? 2 : 0)));
            return (
              <CircleMarker
                key={r.station_id}
                center={[Number(r.lat), Number(r.lon)]}
                radius={rad}
                pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.85 }}
              >
              <Tooltip>
                <div style={{ fontSize: "12.5px", lineHeight: 1.4 }}>
                  <div style={{ fontWeight: 700 }}>{r.name}</div>
                  <div>Vélos : {r.bikes ?? "—"}</div>
                  <div>Bornes : {r.docks_avail ?? "—"}</div>
                  {pen && <div style={{ color: "#ef4444", fontWeight: 600 }}>⚠️ Pénurie</div>}
                  {sat && <div style={{ color: "#3b82f6", fontWeight: 600 }}>⚠️ Saturation</div>}
                </div>
              </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
    );
  }

  return MapInner;
}, { ssr: false });
