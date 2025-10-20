// ui/pages/index.tsx
import { useEffect, useMemo, useRef } from "react";
import Head from "next/head";
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";

export default function LandingPage() {
  // Refs
  const demoIframeRef = useRef<HTMLIFrameElement | null>(null);
  const demoSkeletonRef = useRef<HTMLDivElement | null>(null);

  const year = useMemo(() => new Date().getFullYear(), []);

  // KPI counters (respect reduced motion)
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
  ];

  // Auto-hide header (ajoute .autohide et toggle .is-hidden selon scroll)
  useEffect(() => {
    const header = document.querySelector<HTMLElement>(".site-header");
    if (!header) return;

    header.classList.add("autohide"); // active la mécanique CSS
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
        <meta
          property="og:image"
          content="https://example.com/velib-forecast/cover.jpg"
        />
        <meta name="twitter:card" content="summary_large_image" />
        <meta
          name="twitter:title"
          content="Vélib’ Forecast Paris — Prévisions +15 min"
        />
        <meta
          name="twitter:description"
          content="Carte temps réel & prévisions à +15 minutes."
        />
        <meta
          name="twitter:image"
          content="https://example.com/velib-forecast/cover.jpg"
        />

        {/* ===== Perf ===== */}
        <link
          rel="preconnect"
          href="https://velib-ui-160046094975.europe-west1.run.app"
          crossOrigin=""
        />
        <link
          rel="dns-prefetch"
          href="https://velib-ui-160046094975.europe-west1.run.app"
        />

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

      {/* Header global */}
      <GlobalHeader items={headerItems} brandHref="/" />

      {/* ====================== CONTENT ====================== */}
      <main id="top">
        {/* ====================== HERO ====================== */}
        <section className="hero" aria-labelledby="hero-title">
          <div className="container hero-grid">
            <div>
              <div className="eyebrow">
                <span className="ping" aria-hidden="true" />
                <span className="chip" aria-label="Horizon de prévision">
                  Prévisions +15 min • Paris
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
                    background:
                      "linear-gradient(90deg,var(--primary),var(--primary-2))",
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
                Carte temps réel, prédictions à +15 min par station, comparaison aux
                comportements historiques, et monitoring natif. Conçu pour fiabilité,
                vitesse et clarté — même aux heures de pointe.
              </p>

              <ul className="text-muted" style={{ margin: "10px 0 0", paddingLeft: 18 }}>
                <li>Filtres quartier, recherche suggérée, focus proximité.</li>
                <li>Mises à jour live, transitions fluides, lisibilité renforcée.</li>
                <li>Prévisions calibrées, médiane historique et profils horaires.</li>
              </ul>

              <div className="cta">
                {/* Principal : reste en couleur */}
                <a className="btn" href="#demo" aria-label="Ouvrir la démo en direct">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M7 7h10v10H7z" stroke="white" strokeWidth="1.8" />
                    <path d="M3 3v6M3 3h6M21 21v-6M21 21h-6" stroke="white" strokeWidth="1.8" />
                  </svg>
                  Ouvrir la démo
                </a>
                {/* Secondaires : harmonisés en outline */}
                <a className="btn outline" href="#how">Architecture</a>
                <a className="btn outline" href="#monitoring">Monitoring</a>
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
                  <div className="value" data-count="98%">0%</div>
                  <div className="label">Observations couvertes</div>
                </div>
                <div className="kpi" role="listitem">
                  <div className="value" data-count="5">0</div>
                  <div className="label">Fraîcheur (min)</div>
                </div>
                <div className="kpi" role="listitem">
                  <div className="value" data-count="1400">0</div>
                  <div className="label">Stations suivies</div>
                </div>
              </div>

              {/* ✅ Aperçu statique sans double écriture (plus de skeleton ici) */}
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
        <section id="demo" aria-labelledby="demo-title">
          <div className="container">
            <div className="sec-head">
              <div>
                <h2 id="demo-title">Démo en direct</h2>
                <p>
                  Application React embarquée : carte en direct, recherche de stations, et
                  prévisions à +15 minutes. Le premier accès peut prendre quelques secondes
                  (cold start Cloud Run).
                </p>
              </div>
              <div>
                <span className="kbd" aria-hidden="true">Alt</span> + <span className="kbd" aria-hidden="true">Clique</span>{" "}
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
                src="https://velib-ui-160046094975.europe-west1.run.app/"
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
                href="https://velib-ui-160046094975.europe-west1.run.app/"
                target="_blank"
                rel="noopener"
              >
                Ouvrir dans un onglet
              </a>
              <button className="btn outline" type="button" onClick={handleReload}>
                Recharger la démo
              </button>
              <a className="btn outline" href="#features">Découvrir les fonctions</a>
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
        <section id="features" aria-labelledby="features-title">
          <div className="container">
            <div className="sec-head">
              <div>
                <h2 id="features-title">Plein usage, du matin au soir</h2>
                <p>
                  Repérez les stations utiles, visualisez l’évolution à +15 min, comparez à
                  la médiane, puis basculez en mode monitoring si besoin.
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
                  Couleurs travaillées, légende compacte, recherche instantanée, focus quartier.
                  Affichage pensé pour 1–2 infos clés par station (vélos/capacité + tendance).
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
                  Modèle entraîné sur l’historique et enrichi météo (vents, pluie, saisonnalités).
                  Calibrage par segments horaires et stations pour limiter les biais.
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
                  “Aujourd’hui vs médiane” et profils horaires par station pour comprendre
                  les dynamiques locales (heures de pointe, zones de reports, anomalies).
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
                  KPIs fraîcheur/complétude, alertes simples (saturation/pénurie), suivi
                  de stabilité des features — pour des décisions fiables.
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
                  Un simple <code>&lt;iframe&gt;</code> suffit (Cloud Run, proxy, sous-domaine),
                  avec thème auto (clair/sombre) et navigation clavier.
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
                  Code modulaire : nouveaux horizons (T+60), nouvelles villes, nouvelles
                  sources — sans refonte complète.
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
        <section id="monitoring" aria-labelledby="monitoring-title">
          <div className="container">
            {/* ────── En-tête de section ────── */}
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

            {/* ────── Showcase : carte principale + sous-cartes ────── */}
            <div className="showcase">
              {/* Carte principale */}
              <figure className="card">
                <figcaption className="cap">
                  <strong>Data Health Dashboard</strong>
                  <span>Détails techniques & KPIs</span>
                </figcaption>
                <div className="ratio">
                  <small>Prévisualisation — insérez vos captures générées</small>
                </div>
              </figure>

              {/* Deux sous-cartes côte à côte */}
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

            {/* ────── Détails suivis ────── */}
            <div className="glass prose mt-2">
              <h3>Ce que l’on suit</h3>
              <ul className="text-muted" style={{ paddingLeft: 18 }}>
                <li><strong>Freshness</strong> : p50/p95, hors-plage, trous de capture.</li>
                <li><strong>Coverage</strong> : % lignes valides, champs critiques, NaN sûrs.</li>
                <li><strong>Stability</strong> : dérive simple (KS/PSI) sur features clés.</li>
                <li><strong>Alerts</strong> : pénurie/saturation anormales, outliers horaires.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* ====================== HOW ====================== */}
        <section id="how" aria-labelledby="how-title">
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
                <strong>GBFS → DuckDB</strong>
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
        <section id="faq" aria-labelledby="faq-title">
          <div className="container grid-2">
            <div>
              <div className="sec-head">
                <h2 id="faq-title">FAQ</h2>
              </div>

              <details>
                <summary>La démo met quelques secondes à démarrer, normal ?</summary>
                <p>
                  Oui, c’est le cold start de Cloud Run. Les accès suivants sont instantanés.
                  Vous pouvez configurer une instance minimum pour éviter ce délai.
                </p>
              </details>

              <details>
                <summary>Puis-je intégrer l’app dans mon site ?</summary>
                <p>
                  Oui, via un simple <code>&lt;iframe&gt;</code>. La page gère le responsive,
                  le thème clair/sombre et la navigation clavier.
                </p>
              </details>

              <details>
                <summary>Comment sont calculées les prévisions ?</summary>
                <p>
                  Entraînement station-par-station avec signaux calendrier/météo. Une baseline
                  de persistance permet de mesurer l’amélioration réelle et d’éviter
                  les gains artificiels.
                </p>
              </details>

              <details>
                <summary>Et la qualité des données ?</summary>
                <p>
                  Contrôles de fraîcheur (p50/p95), complétude des champs critiques, dérive
                  simple des features, et alertes sur pénurie/saturation. Exports JSON
                  pour vos propres tableaux de bord.
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
                Vous pouvez aussi placer l’app derrière un sous-domaine (ex. <em>app.votredomaine.fr</em>),
                avec un enregistrement CNAME et des headers de sécurité adaptés.
              </p>
              <ul className="text-muted" style={{ paddingLeft: 18 }}>
                <li>CORS restreint, CSP stricte, cookies “None; Secure”.</li>
                <li>Build reproductible, image minimale, endpoint de santé /ready.</li>
              </ul>
            </aside>
          </div>
        </section>
      </main>

      {/* Footer global (remplace l’ancien footer inline) */}
      <GlobalFooter />
    </>
  );
}
