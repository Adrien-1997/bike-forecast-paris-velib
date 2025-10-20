/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Webpack (prod par défaut) — OK d'utiliser require.resolve ici
  webpack: (config) => {
    config.resolve.fallback = {
      ...(config.resolve.fallback || {}),
      buffer: require.resolve("buffer/"),
    };
    return config;
  },

  // Turbopack (dev avec --turbopack) — surtout PAS de chemins absolus
  turbopack: {
    resolveAlias: {
      buffer: "buffer", // <- use module specifier, not require.resolve(...)
    },
  },
};

module.exports = nextConfig;
