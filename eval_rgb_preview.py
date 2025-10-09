#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_rgb_preview.py — Vista previa RGB de imágenes FITS reales (g,r,i)
Autor: Yamil + ChatGPT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import AsinhStretch, MinMaxInterval
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATASET = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip_test_dataset/test_dataset"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ["i", "r", "g"]  # i→rojo, r→verde, g→azul
N_IMAGES = 9

# ---------------- FUNCIONES ----------------
def load_band(path):
    """Carga la matriz de píxeles desde una banda FITS."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        # Buscar columna que contenga 'band' en su nombre
        colnames = [c.name for c in data.columns]
        key = next((c for c in colnames if "band" in c.lower()), None)
        if key is None:
            raise ValueError(f"No se encontró columna tipo 'band' en {path}")
        img = np.array(data[key][0], dtype=np.float32)
        return img

def make_rgb_image(obj_id):
    imgs = []
    for band in BANDS:
        path = os.path.join(DATASET, f"{obj_id}_{band}.fits")
        if not os.path.exists(path):
            return None
        try:
            img = load_band(path)
            imgs.append(img)
        except Exception as e:
            print(f"[WARN] Error leyendo {path}: {e}")
            return None

    if len(imgs) != 3:
        return None

    # Stretch asinh + normalización robusta
    stretch = AsinhStretch() + MinMaxInterval()
    normed = [stretch(img) for img in imgs]
    rgb = np.dstack(normed)
    rgb = np.clip(rgb / np.percentile(rgb, 99.5), 0, 1)
    return np.flipud(rgb)

# ---------------- PROCESAMIENTO ----------------
all_ids = sorted(set("_".join(f.split("_")[:2]) for f in os.listdir(DATASET) if f.endswith("_r.fits")))
print(f"📸 Mostrando {N_IMAGES} ejemplos de un total de {len(all_ids)} objetos.")

imgs = []
valid_ids = []

for obj_id in tqdm(all_ids[:N_IMAGES]):
    rgb = make_rgb_image(obj_id)
    if rgb is not None:
        imgs.append(rgb)
        valid_ids.append(obj_id)

# ---------------- MOSAICO ----------------
if not imgs:
    raise RuntimeError("❌ No se pudo generar ninguna imagen válida.")

cols = 3
rows = int(np.ceil(len(imgs) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
axes = axes.flatten()

for idx, (ax, img) in enumerate(zip(axes, imgs)):
    ax.imshow(img, origin="lower")
    ax.set_title(valid_ids[idx].replace("object_", ""), fontsize=8)
    ax.axis("off")

for ax in axes[len(imgs):]:
    ax.axis("off")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "preview_rgb_grid.png")
plt.savefig(out_path, dpi=200)
print(f"✅ Mosaico guardado en {out_path}")
