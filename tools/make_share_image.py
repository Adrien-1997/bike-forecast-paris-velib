# tools/make_share_image.py
# Construit une image unique 1200x627 à partir des visuels générés par make_report.py
import pathlib, io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT   = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DEXP   = ROOT / "docs" / "exports"
EXP    = ROOT / "exports"

# --- helpers -------------------------------------------------------------
def _find_export(stem):
    for p in [EXP/f"{stem}.parquet", EXP/f"{stem}.csv", DEXP/f"{stem}.parquet", DEXP/f"{stem}.csv"]:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    return pd.read_parquet(p)
                else:
                    return pd.read_csv(p)
            except Exception:
                pass
    return pd.DataFrame()

def _load_img(name: str):
    p = ASSETS / name
    return Image.open(p).convert("RGB") if p.exists() else None

def _draw_center(img, tile, margin=6):
    """Resize img to fit tile (keeping ratio) and paste centered with small margin."""
    if img is None:  # placeholder
        ph = Image.new("RGB", tile.size, (245,245,245))
        d = ImageDraw.Draw(ph)
        w,h = ph.size
        d.text((w//2-60,h//2-8), "missing", fill=(120,120,120))
        return ph
    w,h = img.size
    tw,th = tile.size[0]-2*margin, tile.size[1]-2*margin
    ratio = min(tw/w, th/h)
    new = img.resize((max(1,int(w*ratio)), max(1,int(h*ratio))), Image.LANCZOS)
    canvas = Image.new("RGB", tile.size, (255,255,255))
    x = (canvas.size[0]-new.size[0])//2
    y = (canvas.size[1]-new.size[1])//2
    canvas.paste(new, (x,y))
    return canvas

# --- last update ---------------------------------------------------------
hour = _find_export("velib_hourly")
last_str = "n/a"
if not hour.empty:
    s = pd.to_datetime(hour["hour_utc"], errors="coerce")
    if s.notna().any():
        last_str = str(s.max())

# --- charge les 4 images produits par make_report -----------------------
hero = _load_img("hero_occ.png")
risk = _load_img("top_risk.png")
vola = _load_img("top_vol.png")
corr = _load_img("corr_occ_temp.png")

# --- canevas LinkedIn 1200x627 (ratio ~1.91:1) --------------------------
W, H = 1200, 627
bg   = Image.new("RGB", (W,H), (255,255,255))
draw = ImageDraw.Draw(bg)

# header
title   = "Vélib' Paris — Risk & Forecast"
subtitle= f"Last update (UTC): {last_str} • Seuils: <20% / >80%"
try:
    # DejaVu est dispo sur GitHub runner ; local fallback sinon
    font_t = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    font_s = ImageFont.truetype("DejaVuSans.ttf",       20)
except Exception:
    font_t = ImageFont.load_default()
    font_s = ImageFont.load_default()

draw.text((24, 18), title,    fill=(33,33,33), font=font_t)
draw.text((24, 58), subtitle, fill=(90,90,90), font=font_s)

# grille 2x2 sous le header
top = 90
pad = 14
w  = (W - pad*3)//2
h  = (H - top - pad*3)//2

tiles_xy = [
    (pad,            top+pad,            hero),
    (pad*2 + w,      top+pad,            risk),
    (pad,            top*0 + pad*2 + top + h, vola),
    (pad*2 + w,      top*0 + pad*2 + top + h, corr),
]

for x,y,img in tiles_xy:
    tile = Image.new("RGB", (w,h), (255,255,255))
    tile = _draw_center(img, tile)
    bg.paste(tile, (x,y))

out = ASSETS / "share_linkedin.png"
bg.save(out, format="PNG", optimize=True)
print("OK —", out)
