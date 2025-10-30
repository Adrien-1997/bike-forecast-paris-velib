// ui/pages/index.tsx
import Script from "next/script";
import { useEffect, useMemo, useRef, useState } from "react";
import Head from "next/head";
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import { getMonitoringIntro, type IntroDoc } from "@/lib/services/monitoring/intro";

export default function LandingPage() {
  // ────────────────────────────────────────────────────────────────────────────
  // Refs
  const demoIframeRef = useRef<HTMLIFrameElement | null>(null);
  const demoSkeletonRef = useRef<HTMLDivElement | null>(null);

  const year = useMemo(() => new Date().getFullYear(), []);

  // 🔐 Liens de paiement (remplace par tes URLs Stripe/Ko-fi/Sponsors)
  const SUPPORT_ONE_TIME =
    process.env.NEXT_PUBLIC_SUPPORT_ONE_TIME ?? "https://buy.stripe.com/test_123"; // Don unique
  const SUPPORT_MONTHLY =
    process.env.NEXT_PUBLIC_SUPPORT_MONTHLY ?? "https://buy.stripe.com/test_monthly_123"; // Abonnement
  const SUPPORT_SPONSORS =
    process.env.NEXT_PUBLIC_SUPPORT_SPONSORS ?? "https://github.com/sponsors/Adrien-1997"; // Sponsors
  const SUPPORT_KOFI =
    process.env.NEXT_PUBLIC_SUPPORT_KOFI ?? "https://ko-fi.com/adrien61942"; // Ko-fi

  // 🔑 Ko-fi username (déduit de l'URL ou via env)
  const KOFI_USERNAME =
    process.env.NEXT_PUBLIC_KOFI_USERNAME ??
    (() => {
      try {
        const u = new URL(SUPPORT_KOFI);
        const seg = u.pathname.split("/").filter(Boolean);
        return seg[0] || "adrien61942";
      } catch {
        return "adrien61942";
      }
    })();

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
  // ➕ Ajout: lecture des KPIs intro (sans toucher à tes états ci-dessus)
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

  // Formats locaux
  const fmtDateTime = (iso?: string | null) => (iso ? new Date(iso).toLocaleString("fr-FR") : null);
  const generatedAt = fmtDateTime(intro?.generated_at) ?? null;
  const modelVersions = intro?.kpis?.model_versions ?? "h15 / h60";

  // Injection des valeurs dans la barre KPI (en conservant ton animation)
  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    // Valeurs issues d’intro, sinon fallback démo existant
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

  // KPI counters (respect reduced motion) — CONSERVÉ tel quel, gardé pour robustesse
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

  // Iframe: remove DEMO skeleton on load
  useEffect(() => {
    const frame = demoIframeRef.current;
    const onLoad = () => demoSkeletonRef.current?.remove();
    if (!frame) return;
    frame.addEventListener("load", onLoad);
    return () => frame.removeEventListener("load", onLoad);
  }, []);

  // Actions
  const handleReload = () => {
    const frame = demoIframeRef.current;
    if (!frame) return;
    const url = frame.src;
    frame.src = "";
    setTimeout(() => {
      frame.src = url;
    }, 60);
  };

  const handleFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        await demoIframeRef.current?.requestFullscreen?.();
      } else {
        await document.exitFullscreen?.();
      }
    } catch {
      /* noop */
    }
  };

  // Header (ancres internes)
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

  // ───────────────────────────────────────────────────────────────
  // Ko-fi : ouverture via le bouton "Ko-fi" (branché au widget)
  const openKoFi = () => {
    try {
      const api = (window as any).kofiWidgetOverlay;
      const primary = getCssVar("--primary", "#ff6a00");

      if (api && typeof api.draw === "function") {
        // Dessiner une seule fois le widget, aux couleurs du site
        if (!(window as any).__kofiDrawn) {
          api.draw(KOFI_USERNAME, {
            type: "floating-chat",
            "floating-chat.donateButton.text": "Soutenez-moi",
            "floating-chat.donateButton.background-color": primary,
            "floating-chat.donateButton.text-color": "#ffffff",
          });
          (window as any).__kofiDrawn = true;
        }

        // Petit délai pour laisser le DOM du widget apparaître, puis ouverture
        setTimeout(() => {
          const btn =
            document.querySelector<HTMLButtonElement>(
              ".floatingchat-container button, .floatingchat-container [role='button']"
            );
          if (btn) btn.click();
          else window.open(SUPPORT_KOFI, "_blank", "noopener,noreferrer"); // fallback
        }, 60);
        return;
      }
    } catch {
      // ignore
    }
    // Fallback si la lib n'est pas dispo (CSP/adblock)
    window.open(SUPPORT_KOFI, "_blank", "noopener,noreferrer");
  };

  return (
    <>
      <Head>
        {/* ===== Base meta ===== */}
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Vélib’ Forecast Paris — Carte en direct & Prévisions +15 min</title>
        <meta
          name="description"
          content="Carte temps réel des stations Vélib’ avec prévisions à +15 minutes, et monitoring de la qualité des données. Application React embarquée, déployée sur Cloud Run."
        />
        <meta name="theme-color" content="#0b1220" />
        <meta name="color-scheme" content="dark light" />

        {/* ===== Canonical / robots ===== */}
        <link rel="canonical" href="https://example.com/velib-forecast/" />
        <meta name="robots" content="index,follow,max-image-preview:large" />

        {/* ===== OpenGraph / Twitter ===== */}
        <meta
          property="og:title"
          content="Vélib’ Forecast Paris — Carte en direct & Prévisions +15 min"
        />
        <meta
          property="og:description"
          content="Anticipez la disponibilité des stations Vélib’ à +15 min. Carte en direct, comparaisons et monitoring qualité."
        />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://example.com/velib-forecast/" />
        <meta property="og:image" content="https://example.com/velib-forecast/cover.jpg" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Vélib’ Forecast Paris — Prévisions +15 min" />
        <meta name="twitter:description" content="Carte temps réel & prévisions à +15 minutes." />
        <meta name="twitter:image" content="https://example.com/velib-forecast/cover.jpg" />

        {/* ===== Perf ===== */}
        <link
          rel="preconnect"
          href="https://velib-ui-160046094975.europe-west1.run.app"
          crossOrigin=""
        />
        <link rel="dns-prefetch" href="https://velib-ui-160046094975.europe-west1.run.app" />

        {/* ===== JSON-LD ===== */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebSite",
              name: "Vélib’ Forecast Paris",
              url: "https://example.com/velib-forecast/",
              description:
                "Carte temps réel des stations Vélib’ avec prévisions à +15 minutes et monitoring.",
              inLanguage: "fr-FR",
              publisher: { "@type": "Organization", name: "Vélib’ Forecast" },
              potentialAction: {
                "@type": "SearchAction",
                target: "https://example.com/velib-forecast/?q={query}",
                "query-input": "required name=query",
              },
            }),
          }}
        />

        {/* Ko-fi overlay script */}
        <Script
          id="kofi-overlay"
          src="https://storage.ko-fi.com/cdn/scripts/overlay-widget.js"
          strategy="afterInteractive"
        />

        {/* Z-index de sécurité pour que le widget soit au-dessus */}
        <style jsx global>{`
          .floatingchat-container {
            z-index: 10000 !important;
          }
        `}</style>
      </Head>

      {/* ===== A11y skip link ===== */}
      <a href="#demo" className="sr-only">
        Aller au contenu principal
      </a>

      {/* Header global */}
      <GlobalHeader items={headerItems} brandHref="/" />

      {/* ====================== CONTENT ====================== */}
      {/* 👇 Wrapper .monitoring pour hériter des tokens/fond/containers */}
      <div className="monitoring">
        <main id="top" className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
          {/* Loading bar homogène */}
          <LoadingBar status={barStatus} />
          {error && <div className="banner banner--error mt-2">{error}</div>}
          {/* ➕ Ajout léger : ligne méta sans rien retirer */}
          {!introError && generatedAt && (
            <div className="kpi-bar-meta" style={{ marginTop: 6 }}>
              Mise à jour monitoring : {generatedAt} · Modèle : {modelVersions}
            </div>
          )}

          {/* ====================== HERO ====================== */}
          <section className="panel hero" aria-labelledby="hero-title">
            <div className="container hero-grid">
              <div>
                <div className="eyebrow">
                  <span className="ping" aria-hidden="true" />
                  <span className="chip" aria-label="Horizon de prévision">
                    Prévisions +15,+60 min • Paris
                  </span>
                  <span className="chip" aria-label="Actualisation">
                    Données live 5 min
                  </span>
                </div>

                <h1 id="hero-title">
                  Anticipez les stations Vélib’
                  <br />
                  avec une{" "}
                  <span
                    style={{
                      background: "linear-gradient(90deg,var(--primary),var(--primary-2))",
                      WebkitBackgroundClip: "text",
                      backgroundClip: "text",
                      color: "transparent",
                    }}
                  >
                    UX taillée pour la ville
                  </span>
                  .
                </h1>

                <p className="lead">
                  Carte temps réel, prédictions à +15 min par station, comparaison aux comportements historiques, et
                  monitoring natif. Conçu pour fiabilité, vitesse et clarté — même aux heures de pointe.
                </p>

                <ul className="text-muted" style={{ margin: "10px 0 0", paddingLeft: 18 }}>
                  <li>Filtres quartier, recherche suggérée, focus proximité.</li>
                  <li>Mises à jour live, transitions fluides, lisibilité renforcée.</li>
                  <li>Prévisions calibrées, médiane historique et profils horaires.</li>
                </ul>

                <div className="cta">
                  <a className="btn" href="#demo" aria-label="Ouvrir la démo en direct">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <path d="M7 7h10v10H7z" stroke="white" strokeWidth="1.8" />
                      <path d="M3 3v6M3 3h6M21 21v-6M21 21h-6" stroke="white" strokeWidth="1.8" />
                    </svg>
                    Ouvrir la démo
                  </a>
                  <a className="btn outline" href="#how">
                    Architecture
                  </a>
                  <a className="btn outline" href="#monitoring">
                    Monitoring
                  </a>
                </div>

                <div className="tech-chips">
                  <span className="chip">Cloud Run</span>
                  <span className="chip">Next.js</span>
                  <span className="chip">React-Leaflet</span>
                  <span className="chip">Cloud Storage</span>
                  <span className="chip">LightGBM</span>
                </div>
              </div>

              <aside className="glass hero-card" aria-label="Indicateurs clés">
                <h3>En chiffres — 7 derniers jours</h3>

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

                  <div className="kpi-bar-meta">Démo · valeurs illustratives</div>
                </div>

                <div className="embed">
                  <div className="ratio">
                    <small>Prévisualisation statique — lancez la démo ci-dessous</small>
                  </div>
                </div>

                <ul className="text-muted" style={{ margin: "12px 0 0", paddingLeft: 18 }}>
                  <li>MAE baseline vs modèle, par station et par segments.</li>
                  <li>Défaut tolérant : trous comblés, horodatage strict, NaN sûrs.</li>
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
                    Application React embarquée : carte en direct, recherche de stations, et prévisions à +15 minutes. Le
                    premier accès peut prendre quelques secondes (cold start Cloud Run).
                  </p>
                </div>
                <div>
                  <span className="kbd" aria-hidden="true">Alt</span> +{" "}
                  <span className="kbd" aria-hidden="true">Clique</span>{" "}
                  <span className="sr-only">Astuce :</span> pour plein écran
                </div>
              </div>

              <div className="embed" aria-live="polite">
                <div className="skeleton" id="skeleton" ref={demoSkeletonRef}>
                  Initialisation de la démo…
                </div>
                <iframe
                  ref={demoIframeRef}
                  title="Vélib’ Forecast — Application"
                  src="http://localhost:3000/app"
                  loading="lazy"
                  allow="fullscreen; clipboard-read; clipboard-write"
                  referrerPolicy="no-referrer-when-downgrade"
                />
                <button
                  className="btn btn-fs"
                  type="button"
                  onClick={handleFullscreen}
                  aria-label="Plein écran"
                >
                  Plein écran
                </button>
              </div>

              <div className="actions-row">
                <a
                  className="btn"
                  href="http://localhost:3000/app"
                  target="_blank"
                  rel="noopener"
                >
                  Ouvrir dans un onglet
                </a>
                <button className="btn outline" type="button" onClick={handleReload}>
                  Recharger la démo
                </button>
                <a className="btn outline" href="#features">
                  Découvrir les fonctions
                </a>
              </div>

              <div className="glass prose mt-2">
                <h3>Pourquoi c’est fluide ?</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Préchargement DNS et connexions persistantes.</li>
                  <li>Découpage UI, caches navigateur et CDN.</li>
                  <li>Metrics RUM pour piloter l’expérience réelle.</li>
                </ul>
              </div>
            </div>
          </section>

          {/* ====================== FEATURES ====================== */}
          <section id="features" className="panel" aria-labelledby="features-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="features-title">Plein usage, du matin au soir</h2>
                  <p>
                    Repérez les stations utiles, visualisez l’évolution à +15 min, comparez à la médiane, puis basculez en
                    mode monitoring si besoin.
                  </p>
                </div>
                <a className="btn outline" href="#demo">
                  Essayer maintenant
                </a>
              </div>

              <div className="features">
                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M12 7v10M7 12h10" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Carte lisible & rapide</h3>
                  <p>
                    Couleurs travaillées, légende compacte, recherche instantanée, focus quartier. Affichage pensé pour 1–2
                    infos clés par station (vélos/capacité + tendance).
                  </p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M3 12a9 9 0 1018 0A9 9 0 003 12z" stroke="currentColor" strokeWidth="2" />
                      <path d="M12 7v6l4 2" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Prévisions à +15 min</h3>
                  <p>
                    Modèle entraîné sur l’historique et enrichi météo (vents, pluie, saisonnalités). Calibrage par segments
                    horaires et stations pour limiter les biais.
                  </p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M4 4h16v16H4z" stroke="currentColor" strokeWidth="2" />
                      <path d="M4 9h16M9 4v16" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Comparaisons utiles</h3>
                  <p>
                    “Aujourd’hui vs médiane” et profils horaires par station pour comprendre les dynamiques locales (heures
                    de pointe, zones de reports, anomalies).
                  </p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M4 12h16" stroke="currentColor" strokeWidth="2" />
                      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Monitoring intégré</h3>
                  <p>
                    KPIs fraîcheur/complétude, alertes simples (saturation/pénurie), suivi de stabilité des features — pour
                    des décisions fiables.
                  </p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M7 7h10v10H7z" stroke="currentColor" strokeWidth="2" />
                      <path d="M3 3v6M3 3h6M21 21v-6M21 21h-6" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Accessible partout</h3>
                  <p>
                    Un simple <code>&lt;iframe&gt;</code> suffit (Cloud Run, proxy, sous-domaine), avec thème auto
                    (clair/sombre) et navigation clavier.
                  </p>
                </article>

                <article className="feature">
                  <div className="icon" aria-hidden="true">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </div>
                  <h3>Pensé pour évoluer</h3>
                  <p>
                    Code modulaire : nouveaux horizons (T+60), nouvelles villes, nouvelles sources — sans refonte complète.
                  </p>
                </article>
              </div>

              <div className="glass prose mt-2">
                <h3>Cas d’usage rapides</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Communication et info voyageurs : carte intégrée à un site de quartier/entreprise.</li>
                  <li>Immobilier/événementiel : repérer les zones sous- ou sur-servies à l’instant T.</li>
                  <li>Mobilité individuelle : planifier un trajet avec station d’arrivée fiable.</li>
                </ul>
              </div>
            </div>
          </section>

          {/* ====================== MONITORING ====================== */}
          <section id="monitoring" className="panel" aria-labelledby="monitoring-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="monitoring-title">Monitoring & Qualité des données</h2>
                  <p>
                    Surveille en continu la fraîcheur, la couverture et les anomalies pour préserver la fiabilité des
                    prévisions. Export des KPIs en JSON pour alimenter d’autres vues.
                  </p>
                </div>
                <a className="btn outline" href="#faq">
                  En savoir plus
                </a>
              </div>

              <div className="showcase">
                <figure className="card">
                  <figcaption className="cap">
                    <strong>Data Health Dashboard</strong>
                    <span>Détails techniques & KPIs</span>
                  </figcaption>
                  <div className="ratio">
                    <small>Prévisualisation — insérez vos captures générées</small>
                  </div>
                </figure>

                <div className="kpi-row">
                  <figure className="card">
                    <figcaption className="cap">
                      <strong>Fraîcheur</strong>
                      <span>Objectif ≤ 5 min</span>
                    </figcaption>
                    <div className="ratio" />
                  </figure>
                  <figure className="card">
                    <figcaption className="cap">
                      <strong>Complétude</strong>
                      <span>Stations × heures</span>
                    </figcaption>
                    <div className="ratio" />
                  </figure>
                </div>
              </div>

              <div className="glass prose mt-2">
                <h3>Ce que l’on suit</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>
                    <strong>Freshness</strong> : p50/p95, hors-plage, trous de capture.
                  </li>
                  <li>
                    <strong>Coverage</strong> : % lignes valides, champs critiques, NaN sûrs.
                  </li>
                  <li>
                    <strong>Stability</strong> : dérive simple (KS/PSI) sur features clés.
                  </li>
                  <li>
                    <strong>Alerts</strong> : pénurie/saturation anormales, outliers horaires.
                  </li>
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
                  <p>
                    Un pipeline robuste de l’ingestion à la mise en prod, avec des composants simples à maintenir et des
                    points de contrôle clairs.
                  </p>
                </div>
              </div>

              <div className="steps" role="list">
                <div className="step" role="listitem">
                  <span className="chip">1 · Ingestion</span>
                  <strong>GBFS → Cloud Storage</strong>
                  <p>Snapshots toutes les 5 minutes, consolidation journalière, schéma strict.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Parquet shardé (daily/weekly) pour IO efficaces.</li>
                    <li>Clés station_id + tbin_utc, zones horaires UTC/locale.</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">2 · Enrichissement</span>
                  <strong>Features calendrier & météo</strong>
                  <p>Jour/semaine, vacances, sin/cos horaires, pluie/vent.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Rollings (lags, fenêtres 1–4 h) et indicateurs de tendance.</li>
                    <li>Sanitization JSON (NaN→null) pour APIs propres.</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">3 · Modélisation</span>
                  <strong>LightGBM (T+15)</strong>
                  <p>Évaluation MAE/WAPE vs baseline persistance par segments.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Calibration légère, contrôle des sur-/sous-estimations.</li>
                    <li>Artifacts versionnés (joblib) et manifest JSON.</li>
                  </ul>
                </div>
                <div className="step" role="listitem">
                  <span className="chip">4 · App & Docs</span>
                  <strong>Next.js + APIs</strong>
                  <p>Carte interactive, pages Réseau/Modèle/Monitoring/Data.</p>
                  <ul className="text-muted" style={{ marginTop: 8, paddingLeft: 18 }}>
                    <li>Déploiement Cloud Run, CORS maîtrisé, headers sûrs.</li>
                    <li>Static props + lazy pour une UX perçue plus rapide.</li>
                  </ul>
                </div>
              </div>

              <div className="glass prose mt-2">
                <h3>Pourquoi c’est fiable ?</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Tests unitaires sur parsing/horodatage et contrats de schéma.</li>
                  <li>Nettoyage systématique des valeurs infinies/NaN avant export.</li>
                  <li>Monitoring indépendant et exports JSON réutilisables.</li>
                </ul>
              </div>
            </div>
          </section>

          {/* ====================== FAQ ====================== */}
          <section id="faq" className="panel" aria-labelledby="faq-title">
            <div className="container grid-2">
              <div>
                <div className="sec-head">
                  <h2 id="faq-title">FAQ</h2>
                </div>

                <details>
                  <summary>La démo met quelques secondes à démarrer, normal ?</summary>
                  <p>
                    Oui, c’est le cold start de Cloud Run. Les accès suivants sont instantanés. Vous pouvez configurer une
                    instance minimum pour éviter ce délai.
                  </p>
                </details>

                <details>
                  <summary>Puis-je intégrer l’app dans mon site ?</summary>
                  <p>
                    Oui, via un simple <code>&lt;iframe&gt;</code>. La page gère le responsive, le thème clair/sombre et la
                    navigation clavier.
                  </p>
                </details>

                <details>
                  <summary>Comment sont calculées les prévisions ?</summary>
                  <p>
                    Entraînement station-par-station avec signaux calendrier/météo. Une baseline de persistance permet de
                    mesurer l’amélioration réelle et d’éviter les gains artificiels.
                  </p>
                </details>

                <details>
                  <summary>Et la qualité des données ?</summary>
                  <p>
                    Contrôles de fraîcheur (p50/p95), complétude des champs critiques, dérive simple des features, et
                    alertes sur pénurie/saturation. Exports JSON pour vos propres tableaux de bord.
                  </p>
                </details>
              </div>

              <aside className="glass prose">
                <h3>Intégration Cloud Run</h3>
                <p className="text-muted">Remplacez l’URL ci-dessous par l’endpoint public de votre service.</p>
                <pre>
                  <code>{`<iframe
  src="https://velib-ui-160046094975.europe-west1.run.app/"
  width="100%" height="68vh" style="border:0"
  allow="fullscreen"></iframe>`}</code>
                </pre>
                <p className="text-muted" style={{ fontSize: ".95rem" }}>
                  Vous pouvez aussi placer l’app derrière un sous-domaine (ex. <em>app.votredomaine.fr</em>), avec un
                  enregistrement CNAME et des headers de sécurité adaptés.
                </p>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>CORS restreint, CSP stricte, cookies “None; Secure”.</li>
                  <li>Build reproductible, image minimale, endpoint de santé /ready.</li>
                </ul>
              </aside>
            </div>
          </section>

          {/* ====================== SUPPORT ====================== */}
          <section id="support" className="panel" aria-labelledby="support-title">
            <div className="container">
              <div className="sec-head">
                <div>
                  <h2 id="support-title">Soutenir le projet</h2>
                  <p>
                    Ce projet est développé et maintenu indépendamment pour proposer une expérience fluide de la mobilité à
                    Paris. Votre soutien permet de couvrir l’hébergement, la supervision et le temps de R&D.
                  </p>
                </div>
              </div>

              <div className="grid-2">
                {/* Bio courte */}
                <article className="glass prose">
                  <h3>À propos</h3>
                  <p className="text-muted">
                    Je m’appelle <strong>Adrien</strong>, ingénieur en mathématiques appliquées spécialisé en analyse,
                    modélisation statistique et machine learning. J’aime transformer des données réelles en outils utiles,
                    fiables et élégants – ici, pour anticiper la disponibilité des vélos en ville.
                  </p>
                  <ul className="text-muted" style={{ paddingLeft: 18 }}>
                    <li>Pipeline temps réel (GBFS + météo) et modèles LightGBM.</li>
                    <li>App Next.js avec carte interactive et monitoring dédié.</li>
                    <li>Hébergement sur Cloud Run, coûts optimisés.</li>
                  </ul>
                  <p className="text-muted" style={{ fontSize: ".95rem" }}>
                    Vous pouvez contribuer une fois, vous abonner mensuellement, ou devenir sponsor. Merci 🙏
                  </p>
                </article>

                {/* Cartes de paiement */}
                <div className="support-cards">
                  <figure className="card">
                    <figcaption className="cap">
                      <strong>Don unique</strong>
                      <span>Rapide et sans compte</span>
                    </figcaption>
                    <div className="ratio" />
                    <div className="actions-row">
                      <a className="btn" href={SUPPORT_ONE_TIME} target="_blank" rel="noopener">
                        Contribuer une fois
                      </a>

                      {/* BOUTON KO-FI BRANCHÉ SUR LE WIDGET */}
                      <button
                        className="btn outline"
                        type="button"
                        onClick={openKoFi}
                        aria-label="Soutenez-moi sur Ko-fi"
                      >
                        Ko-fi
                      </button>
                    </div>
                    <small className="text-muted" style={{ display: "block", marginTop: 8 }}>
                      Géré par Stripe/Ko-fi. Les frais de plateforme s’appliquent.
                    </small>
                  </figure>

                  <figure className="card">
                    <figcaption className="cap">
                      <strong>Mensuel</strong>
                      <span>Annulable à tout moment</span>
                    </figcaption>
                    <div className="ratio" />
                    <div className="actions-row">
                      <a className="btn" href={SUPPORT_MONTHLY} target="_blank" rel="noopener">
                        Soutien mensuel
                      </a>
                      <a className="btn outline" href={SUPPORT_SPONSORS} target="_blank" rel="noopener">
                        GitHub Sponsors
                      </a>
                    </div>
                    <small className="text-muted" style={{ display: "block", marginTop: 8 }}>
                      Abonnements sécurisés. Reçus automatiques par e-mail.
                    </small>
                  </figure>
                </div>
              </div>

              {/* Encadré fiscalité / contact */}
              <div className="glass prose mt-2">
                <h3>Transparence & contact</h3>
                <ul className="text-muted" style={{ paddingLeft: 18 }}>
                  <li>Les contributions financent l’hébergement, la supervision et l’amélioration continue.</li>
                  <li>Pas de contreparties fiscales particulières (don non-déductible), sauf mention contraire.</li>
                  <li>
                    Besoin d’un reçu, d’une facture ou d’un partenariat ? Écrivez-moi : <em>contact@votredomaine.fr</em>.
                  </li>
                </ul>
                <p className="small muted" style={{ marginTop: 8 }}>
                  © {year} • Vélib’ Forecast Paris
                </p>
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
