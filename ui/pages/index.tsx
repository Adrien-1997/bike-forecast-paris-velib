// ui/pages/index.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import Head from "next/head";

export default function LandingPage() {
  // Mobile menu state
  const [menuOpen, setMenuOpen] = useState(false);

  // Refs for behavior
  const headerRef = useRef<HTMLElement | null>(null);
  const navRef = useRef<HTMLDivElement | null>(null);
  const navListRef = useRef<HTMLUListElement | null>(null);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const skeletonRef = useRef<HTMLDivElement | null>(null);
  const year = useMemo(() => new Date().getFullYear(), []);

  // Sections & scroll active link
  useEffect(() => {
    const header = headerRef.current!;
    const onScroll = () => {
      if (!header) return;
      header.classList.toggle("scrolled", window.scrollY > 20);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Mobile menu backdrop + Esc close
  useEffect(() => {
    if (!menuOpen) {
      // remove backdrop if present
      document.querySelector(".nav-backdrop")?.remove();
      return;
    }
    // add backdrop
    const b = document.createElement("div");
    b.className = "nav-backdrop";
    b.addEventListener("click", () => setMenuOpen(false), { passive: true });
    document.body.appendChild(b);

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.querySelector(".nav-backdrop")?.remove();
    };
  }, [menuOpen]);

  // Active nav link on scroll
  useEffect(() => {
    const linkNodes = Array.from(
      (navListRef.current?.querySelectorAll("a") ?? []) as NodeListOf<HTMLAnchorElement>
    );

    const targets = linkNodes
      .map((a) => document.querySelector<HTMLElement>(a.getAttribute("href") || ""))
      .filter(Boolean) as HTMLElement[];

    const setActive = () => {
      const y = window.scrollY + 120;
      let active: HTMLAnchorElement | null = linkNodes[0] ?? null;
      targets.forEach((sec, i) => {
        if (sec.offsetTop <= y) active = linkNodes[i] ?? active;
      });
      linkNodes.forEach((a) => a.classList.remove("active"));
      active?.classList.add("active");
    };

    document.addEventListener("scroll", setActive, { passive: true });
    setActive();
    return () => document.removeEventListener("scroll", setActive);
  }, []);

  // KPI counters (respect reduced motion)
  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const ease = (t: number) => 1 - Math.pow(1 - t, 4);
    const animateCount = (el: HTMLElement, to: number, suffix = "") => {
      const start = performance.now();
      const dur = 1100 + Math.random() * 600;
      const from = 0;
      const step = (now: number) => {
        const p = Math.min(1, (now - start) / dur);
        const v = Math.round((from + (to - from) * ease(p)) * 10) / 10;
        el.textContent = suffix
          ? v + suffix
          : el.dataset.count?.includes("%")
          ? v + "%"
          : String(v);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };

    document.querySelectorAll<HTMLElement>(".kpi .value").forEach((el) => {
      const raw = el.dataset.count;
      if (!raw) return;
      if (prefersReduced) {
        el.textContent = raw;
        return;
      }
      const isPct = raw.includes("%");
      const to = parseFloat(raw);
      if (Number.isFinite(to)) animateCount(el, to, isPct ? "%" : "");
    });
  }, []);

  // Iframe: remove skeleton on load
  useEffect(() => {
    const frame = iframeRef.current;
    const skel = skeletonRef.current;
    if (!frame) return;
    const onLoad = () => skel?.remove();
    frame.addEventListener("load", onLoad);
    return () => frame.removeEventListener("load", onLoad);
  }, []);

  // Handlers
  const handleReload = () => {
    const frame = iframeRef.current;
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
        await iframeRef.current?.requestFullscreen?.();
      } else {
        await document.exitFullscreen?.();
      }
    } catch {
      /* noop */
    }
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
        <link
          rel="icon"
          href={
            "data:image/svg+xml," +
            encodeURIComponent(
              '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="0.9em" font-size="90">🚲</text></svg>'
            )
          }
        />

        {/* ===== Canonical / robots ===== */}
        <link rel="canonical" href="https://example.com/velib-forecast/" />
        <meta name="robots" content="index,follow,max-image-preview:large" />

        {/* ===== OpenGraph / Twitter ===== */}
        <meta property="og:title" content="Vélib’ Forecast Paris — Carte en direct & Prévisions +15 min" />
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
        <link rel="preconnect" href="https://velib-ui-160046094975.europe-west1.run.app" crossOrigin="" />
        <link rel="dns-prefetch" href="https://velib-ui-160046094975.europe-west1.run.app" />

        {/* ===== Styles (scoped landing) ===== */}
        <link rel="stylesheet" href="/css/landing.css" />

        {/* ===== JSON-LD ===== */}
        <script
          type="application/ld+json"
          // eslint-disable-next-line react/no-danger
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
      </Head>

      {/* ===== A11y skip link ===== */}
      <a href="#demo" className="sr-only">
        Aller au contenu principal
      </a>

      {/* ====================== NAV ====================== */}
      <header ref={headerRef}>
        <div
          ref={navRef}
          className={`container nav${menuOpen ? " is-open" : ""}`}
          role="navigation"
          aria-label="Navigation principale"
        >
          <a className="brand" href="#top" aria-label="Accueil Vélib’ Forecast">
            <span className="logo" aria-hidden="true">
              🚲
            </span>
            <span>Vélib’ Forecast</span>
          </a>

          <nav aria-label="Sections">
            <button
              className="nav__burger"
              aria-label={menuOpen ? "Fermer le menu" : "Ouvrir le menu"}
              aria-expanded={menuOpen}
              aria-controls="navList"
              onClick={() => setMenuOpen((v) => !v)}
            >
              <span className="bar" aria-hidden="true" />
            </button>
            <ul id="navList" ref={navListRef}>
              <li>
                <a href="#demo" className="active">
                  Démo
                </a>
              </li>
              <li>
                <a href="#features">Fonctions</a>
              </li>
              <li>
                <a href="#monitoring">Monitoring</a>
              </li>
              <li>
                <a href="#how">Architecture</a>
              </li>
              <li>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      {/* ====================== HERO ====================== */}
      <main id="top">
        <section className="hero" aria-labelledby="hero-title">
          <div className="container hero-grid">
            <div>
              <div className="eyebrow">
                <span className="ping" aria-hidden="true" />
                <span className="chip" aria-label="Horizon de prévision">
                  Prévisions +15 min • Paris
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
                Carte temps réel, prédictions à +15 minutes par station, et monitoring qualité pour une
                fraîcheur de données constante. Pensé pour la fiabilité au quotidien.
              </p>

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
              </div>

              <div className="tech-chips">
                <span className="chip">Cloud Run</span>
                <span className="chip">Next.js</span>
                <span className="chip">React-Leaflet</span>
                <span className="chip">DuckDB</span>
                <span className="chip">LightGBM</span>
              </div>
            </div>

            <aside className="glass hero-card" aria-label="Indicateurs clés">
              <h3>En chiffres — 7 derniers jours</h3>
              <div className="kpis" role="list">
                <div className="kpi" role="listitem">
                  <div className="value" data-count="98%">
                    0%
                  </div>
                  <div className="label">Observations couvertes</div>
                </div>
                <div className="kpi" role="listitem">
                  <div className="value" data-count="5">
                    0
                  </div>
                  <div className="label">Fraîcheur (min)</div>
                </div>
                <div className="kpi" role="listitem">
                  <div className="value" data-count="1400">
                    0
                  </div>
                  <div className="label">Stations suivies</div>
                </div>
              </div>

              <div className="embed">
                <div className="skeleton" aria-hidden="true">
                  Mini-aperçu
                </div>
                <div className="ratio">
                  <small>Prévisualisation statique — lancez la démo ci-dessous</small>
                </div>
              </div>
            </aside>
          </div>
        </section>

        {/* ====================== DEMO (iframe) ====================== */}
        <section id="demo" aria-labelledby="demo-title">
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
                <span className="kbd">Alt</span> + <span className="kbd">Clique</span> pour plein écran
              </div>
            </div>

            <div className="embed" aria-live="polite">
              <div className="skeleton" id="skeleton" ref={skeletonRef}>
                Initialisation de la démo…
              </div>
              <iframe
                ref={iframeRef}
                title="Vélib’ Forecast — Application"
                src="https://velib-ui-160046094975.europe-west1.run.app/"
                loading="lazy"
                allow="fullscreen; clipboard-read; clipboard-write"
                referrerPolicy="no-referrer-when-downgrade"
              />
              <button className="btn btn-fs" type="button" onClick={handleFullscreen} aria-label="Plein écran">
                Plein écran
              </button>
            </div>

            <div className="actions-row">
              <a
                className="btn"
                href="https://velib-ui-160046094975.europe-west1.run.app/"
                target="_blank"
                rel="noopener"
              >
                Ouvrir dans un onglet
              </a>
              <button className="btn outline" type="button" onClick={handleReload}>
                Recharger la démo
              </button>
            </div>
          </div>
        </section>

        {/* ====================== FEATURES ====================== */}
        <section id="features" aria-labelledby="features-title">
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
                <p>Couleurs travaillées, légende compacte, recherche instantanée, focus quartier.</p>
              </article>

              <article className="feature">
                <div className="icon" aria-hidden="true">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M3 12a9 9 0 1018 0A9 9 0 003 12z" stroke="currentColor" strokeWidth="2" />
                    <path d="M12 7v6l4 2" stroke="currentColor" strokeWidth="2" />
                  </svg>
                </div>
                <h3>Prévisions à +15 min</h3>
                <p>Modèle entraîné sur l’historique et enrichi météo pour anticiper disponibilité et saturation.</p>
              </article>

              <article className="feature">
                <div className="icon" aria-hidden="true">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M4 4h16v16H4z" stroke="currentColor" strokeWidth="2" />
                    <path d="M4 9h16M9 4v16" stroke="currentColor" strokeWidth="2" />
                  </svg>
                </div>
                <h3>Comparaisons utiles</h3>
                <p>“Aujourd’hui vs médiane” et profils horaires par station pour comprendre les dynamiques.</p>
              </article>

              <article className="feature">
                <div className="icon" aria-hidden="true">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M4 12h16" stroke="currentColor" strokeWidth="2" />
                    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
                  </svg>
                </div>
                <h3>Monitoring intégré</h3>
                <p>KPIs fraîcheur, complétude, anomalies & résilience du pipeline pour une qualité constante.</p>
              </article>

              <article className="feature">
                <div className="icon" aria-hidden="true">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M7 7h10v10H7z" stroke="currentColor" strokeWidth="2" />
                    <path d="M3 3v6M3 3h6M21 21v-6M21 21h-6" stroke="currentColor" strokeWidth="2" />
                  </svg>
                </div>
                <h3>Responsive & accessible</h3>
                <p>
                  Design mobile-first, contrastes conformes, navigation clavier, respect des préférences système.
                </p>
              </article>

              <article className="feature">
                <div className="icon" aria-hidden="true">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" />
                  </svg>
                </div>
                <h3>Intégrable partout</h3>
                <p>
                  Un simple <code>&lt;iframe&gt;</code> suffit (Cloud Run, proxy, sous-domaine, etc.).
                </p>
              </article>
            </div>
          </div>
        </section>

        {/* ====================== MONITORING ====================== */}
        <section id="monitoring" aria-labelledby="monitoring-title">
          <div className="container">
            <div className="sec-head">
              <div>
                <h2 id="monitoring-title">Monitoring & Qualité des données</h2>
                <p>
                  Surveille en continu la fraîcheur, la couverture et les anomalies pour préserver la fiabilité des
                  prévisions.
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

              <div className="cols-2">
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
          </div>
        </section>

        {/* ====================== HOW ====================== */}
        <section id="how" aria-labelledby="how-title">
          <div className="container">
            <div className="sec-head">
              <div>
                <h2 id="how-title">Sous le capot</h2>
                <p>Un pipeline robuste de l’ingestion à la mise en prod, avec des composants simples à maintenir.</p>
              </div>
            </div>

            <div className="steps" role="list">
              <div className="step" role="listitem">
                <span className="chip">1 · Ingestion</span>
                <strong>GBFS → DuckDB</strong>
                <p>Snapshots toutes les 5 minutes, consolidation, indexation spatio-temporelle.</p>
              </div>
              <div className="step" role="listitem">
                <span className="chip">2 · Enrichissement</span>
                <strong>Features calendrier & météo</strong>
                <p>Jour de semaine, heure, vacances, pluviométrie, vents, etc.</p>
              </div>
              <div className="step" role="listitem">
                <span className="chip">3 · Modélisation</span>
                <strong>LightGBM (T+15)</strong>
                <p>
                  Prévisions station-par-station à +15 min, évaluation MAE/WAPE et baseline persistance.
                </p>
              </div>
              <div className="step" role="listitem">
                <span className="chip">4 · App & Docs</span>
                <strong>Next.js + MkDocs</strong>
                <p>
                  Carte interactive, pages “Réseau/Modèle/Monitoring/Data” et export automatique des KPIs.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ====================== FAQ ====================== */}
        <section id="faq" aria-labelledby="faq-title">
          <div className="container grid-2">
            <div>
              <div className="sec-head">
                <h2 id="faq-title">FAQ</h2>
              </div>

              <details>
                <summary>La démo met quelques secondes à démarrer, normal ?</summary>
                <p>
                  Oui, c’est le cold start de Cloud Run. Les accès suivants sont instantanés. Vous pouvez augmenter
                  l’instance minimum pour éviter ce délai.
                </p>
              </details>

              <details>
                <summary>Puis-je intégrer l’app dans mon site ?</summary>
                <p>
                  Oui, via un simple <code>&lt;iframe&gt;</code>. La page gère déjà le responsive, le thème et le focus
                  clavier.
                </p>
              </details>

              <details>
                <summary>Comment sont calculées les prévisions ?</summary>
                <p>
                  Le modèle apprend les dynamiques horaires par station et incorpore des signaux calendrier/météo. Une
                  baseline de persistance sert de repère pour mesurer l’amélioration.
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
                Vous pouvez aussi placer l’app derrière un sous-domaine (ex. <em>app.votredomaine.fr</em>).
              </p>
            </aside>
          </div>
        </section>
      </main>

      <footer role="contentinfo">
        <div className="container actions-row" style={{ justifyContent: "space-between" }}>
          <div>
            © <span>{year}</span> Vélib’ Forecast — Fait avec ♥ à Paris
          </div>
          <div className="actions-row" style={{ marginTop: 0 }}>
            <a className="chip" href="#demo">
              Démo
            </a>
            <a className="chip" href="#monitoring">
              Monitoring
            </a>
            <a className="chip" href="#how">
              Architecture
            </a>
          </div>
        </div>
      </footer>
    </>
  );
}
