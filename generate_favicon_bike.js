#!/usr/bin/env node
/**
 * Fetch a bike silhouette SVG from the web, normalize & brand it into a favicon.
 * - Downloads any SVG (URL or local path), flattens it, recolors, and frames it.
 * - Produces /public/favicon.svg (and optional PNGs/ICO if --raster)
 *
 * Usage examples:
 *   node tools/fetch_bike_favicon.mjs \
 *     --src="https://assets-web.bonjour-ratp.fr/Velib_hero_83f6afda06.svg"
 *
 *   node tools/fetch_bike_favicon.mjs \
 *     --src=./assets/velib.svg --bg=#0b1220 --stroke=#eaf0fa --accent1=#ff7f11 --accent2=#00aaff --size=64 --radius=14 --raster
 */

import fs from "node:fs/promises";
import path from "node:path";
import { optimize } from "svgo";
import sharp from "sharp";

/* ---------- CLI args ---------- */
const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const [k, v] = a.replace(/^--/, "").split("=");
  return [k, v ?? true];
}));

if (!args.src) {
  console.error("Missing --src=<url-or-path-to-svg>");
  process.exit(1);
}

const OUT_DIR = args.outDir || "public";
const SIZE = Number(args.size || 64);
const RADIUS = Number(args.radius || 14);
const BG = String(args.bg || "#0b1220");
const STROKE = String(args.stroke || "#e7eef9");
const SW = Number(args.sw || 2.6);
const ACC1 = String(args.accent1 || "#ff7f11");
const ACC2 = String(args.accent2 || "#00aaff");
const TRACK = args.track !== "off";
const RASTER = !!args.raster;

/* ---------- helpers ---------- */
const isHttp = (s) => /^https?:\/\//i.test(s);
const readSrc = async (src) => {
  if (isHttp(src)) {
    const res = await fetch(src);
    if (!res.ok) throw new Error(`HTTP ${res.status} while fetching ${src}`);
    return await res.text();
  }
  return await fs.readFile(src, "utf8");
};

const shade = (hex, d) => {
  const { r, g, b } = hexToRgb(hex);
  const k = (v) => Math.min(255, Math.max(0, v + d));
  return `#${toHex(k(r))}${toHex(k(g))}${toHex(k(b))}`;
};
function hexToRgb(h) {
  let s = h.replace("#", "");
  if (s.length === 3) s = s.split("").map(c => c + c).join("");
  const n = parseInt(s, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}
const toHex = (n) => n.toString(16).padStart(2, "0");

/** brutal recolor: force all strokes/fills to STROKE (keeps opacity) */
function recolorSVG(svg, color) {
  // remove inline colors then set a base color on root <g>
  let s = svg
    .replace(/\sfill="[^"]*"/gi, "")
    .replace(/\sstroke="[^"]*"/gi, "")
    .replace(/\sstyle="[^"]*"/gi, "");
  // wrap content into group to apply our color
  const inner = extractInner(s);
  return `<g fill="${color}" stroke="${color}" stroke-width="1" vector-effect="non-scaling-stroke">${inner}</g>`;
}

/** extract inside of outermost <svg>…</svg>  */
function extractInner(svg) {
  const open = svg.indexOf(">");
  const close = svg.lastIndexOf("</svg>");
  if (open >= 0 && close > open) return svg.slice(open + 1, close).trim();
  return svg.trim();
}

/* ---------- main ---------- */
const rawSvg = await readSrc(args.src);

/* svgo optimize original (keeps shapes) */
const svgo = optimize(rawSvg, {
  multipass: true,
  plugins: [
    "removeXMLNS",
    "removeDoctype",
    "removeComments",
    "removeMetadata",
    "removeEditorsNSData",
    "cleanupAttrs",
    "removeUselessDefs",
    "convertStyleToAttrs",
    "convertTransform",
    "cleanupNumericValues",
    "collapseGroups",
    { name: "removeViewBox", active: false },
  ],
});
const cleaned = svgo.data;

/* recolor + embed into our 64×64 frame (rounded bg + optional data track) */
const bikeGroup = recolorSVG(cleaned, STROKE);
const sw = Math.max(2, Math.min(3.2, SW));

const svgOut = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <radialGradient id="bg" cx="50%" cy="45%" r="70%">
      <stop offset="0%" stop-color="${BG}"/>
      <stop offset="100%" stop-color="${shade(BG, -8)}"/>
    </radialGradient>
    <linearGradient id="track" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="${ACC1}"/>
      <stop offset="100%" stop-color="${ACC2}"/>
    </linearGradient>
    <clipPath id="pad"><rect x="8" y="8" width="48" height="48" rx="10"/></clipPath>
  </defs>

  <rect width="64" height="64" rx="${RADIUS}" fill="url(#bg)"/>

  ${TRACK ? `<path d="M10 48 C18 36, 28 50, 36 34 S52 28, 56 22"
           fill="none" stroke="url(#track)" stroke-width="${sw + 0.6}"
           stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>` : ""}

  <!-- Imported & recolored silhouette -->
  <g transform="translate(6,6) scale(0.78)" clip-path="url(#pad)">${bikeGroup}</g>

  ${TRACK ? `<circle cx="56" cy="22" r="${sw}" fill="url(#track)"/>` : ""}
</svg>
`;

/* write outputs */
await fs.mkdir(OUT_DIR, { recursive: true });
const outSvgPath = path.join(OUT_DIR, "favicon.svg");
await fs.writeFile(outSvgPath, svgOut, "utf8");
console.log(`✓ wrote ${outSvgPath}`);

/* optional rasters (PNG + ICO) */
if (RASTER) {
  const sizes = [16, 32, 64, 180, 192, 256, 512];
  await Promise.all(
    sizes.map(async (px) => {
      const buf = await sharp(Buffer.from(svgOut)).resize(px, px).png().toBuffer();
      const p = path.join(OUT_DIR, `icon-${px}.png`);
      await fs.writeFile(p, buf);
      console.log(`✓ wrote ${p}`);
    })
  );

  // favicon.ico (16,32,48)
  const icoBuf = await sharp(Buffer.from(svgOut))
    .resize(48, 48)
    .toFormat("ico", { sizes: [16, 32, 48] })
    .toBuffer();
  const icoPath = path.join(OUT_DIR, "favicon.ico");
  await fs.writeFile(icoPath, icoBuf);
  console.log(`✓ wrote ${icoPath}`);
}
