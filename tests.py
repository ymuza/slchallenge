#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_embeddings_v3_preview.py — Verificación visual de reconstrucción RGB
a partir de los FITS del dataset real (r,g,i).

Muestra un mosaico 3x3 de objetos reconstruidos antes del procesamiento completo.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings

DATASET_DIR = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip_test_dataset/test_dataset"
N_PREVIEW = 9  # cantidad de objetos a mostrar

def read_fits_band(path, key):
    """Lee un FITS y devuelve la matriz de la banda."""
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[1].data[key][0]
            if not isinstance(data, np.ndarray):
                raise ValueError("No contiene un array válido")
            return data.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error leyendo {os.path.basename(path)}: {e}")

# ---------------- Recolectar objetos ----------------
all_ids = sorted({f.split("_")[1] for f in os.listdir(DATASET_DIR) if f.endswith("_r.fits")})
subset_ids = all_ids[:N_PREVIEW]
print(f"📸 Mostrando {len(subset_ids)} ejemplos de un total de {len(all_ids)} objetos.")

imgs = []
valid_ids = []

for base_id in subset_ids:
    try:
        r_file = os.path.join(DATASET_DIR, f"object_{base_id}_r.fits")
        g_file = os.path.join(DATASET_DIR, f"object_{base_id}_g.fits")
        i_file = os.path.join(DATASET_DIR, f"object_{base_id}_i.fits")

        r = read_fits_band(r_file, "band_r")
        g = read_fits_band(g_file, "band_g")
        ib = read_fits_band(i_file, "band_i")

        # Normalizar y apilar como RGB
        img = np.stack([r, g, ib])
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img = np.transpose(img, (1, 2, 0))  # (H,W,C)
        imgs.append(img)
        valid_ids.append(base_id)
    except Exception as e:
        warnings.warn(f"[WARN] Error procesando {base_id}: {e}")

# ---------------- Mostrar mosaico ----------------
# ---------------- Mostrar mosaico ----------------
if not imgs:
    raise RuntimeError("❌ No se pudo cargar ninguna imagen válida.")

cols = 3
rows = int(np.ceil(len(imgs) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
axes = axes.flatten()

for idx, (ax, img) in enumerate(zip(axes, imgs)):
    ax.imshow(img, origin="lower")
    ax.set_title(f"object_{valid_ids[idx]}")
    ax.axis("off")

for ax in axes[len(imgs):]:
    ax.axis("off")

plt.tight_layout()

# 💾 Guardar en outputs/
os.makedirs("outputs", exist_ok=True)
out_path = "documentos_viejos/preview_rgb_grid.png"
plt.savefig(out_path, dpi=200)
print(f"✅ Mosaico guardado en {out_path}")

