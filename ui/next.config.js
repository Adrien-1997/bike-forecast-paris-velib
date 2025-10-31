/** @type {import('next').NextConfig} */

// Helper: construit la directive frame-ancestors
function buildFrameAncestors() {
  // Allowlist venant d'une env var CSV (ex: "https://velo-paris.fr,https://partner.example.com")
  const csv = process.env.EMBED_ALLOWLIST ?? "";
  const origins = csv
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  // Ajoute toujours 'self' (ton propre domaine)
  const base = ["'self'"];

  // En dev, autorise localhost pour tests externes si utile
  if (process.env.NODE_ENV !== "production") {
    base.push("http://localhost:3000");
  }

  return [...base, ...origins].join(" ");
}

const nextConfig = {
  reactStrictMode: true,

  // Webpack (prod par défaut)
  webpack: (config) => {
    config.resolve.fallback = {
      ...(config.resolve.fallback || {}),
      buffer: require.resolve("buffer/"),
    };
    return config;
  },

  // Turbopack (dev)
  turbopack: {
    resolveAlias: {
      buffer: "buffer",
    },
  },

  // 🔐 Headers de sécurité
  async headers() {
    const frameAncestors = buildFrameAncestors();

    return [
      // 1) Par défaut (tout le site) : interdiction d'être embarqué ailleurs
      {
        source: "/:path*",
        headers: [
          // On protège globalement
          { key: "Content-Security-Policy", value: `frame-ancestors 'self';` },
          // ⚠️ N'ajoute PAS X-Frame-Options ici (déprécié et peut entrer en conflit)
        ],
      },

      // 2) Exception pour l'embed : autorise seulement les ancêtres de l'allowlist
      {
        source: "/app/embed",
        headers: [
          // Remplace la directive globale par une plus permissive et contrôlée
          { key: "Content-Security-Policy", value: `frame-ancestors ${frameAncestors};` },
          // Optionnel : Permissions-Policy minimale (laisse le parent décider de la géoloc)
          // { key: "Permissions-Policy", value: "geolocation=()" },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
