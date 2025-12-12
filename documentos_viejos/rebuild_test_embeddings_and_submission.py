#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_test_embeddings_and_submission.py

- Recalcula embeddings del set REAL (FITS r,g,i)
- Registra fallos por ID (banda/razón)
- Clasifica con KNN (entrenado sobre embeddings de entrenamiento)
- Asigna is_lens = -99 a IDs que fallaron
- Predice photo-z (KNN regressor) para casos válidos, y -1.0 para fallidos
- Genera CSV final compatible con el challenge

Salidas:
- outputs/embeddings_test_fixed.npy
- outputs/embeddings_test_ids_fixed.npy
- outputs/failed_ids.txt
- outputs/failed_report.csv
- outputs/submission_final_1024_with_failures.csv
"""

import os
import sys
import gc
import csv
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from astropy.io import fits
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# ------------------- CONFIG -------------------
DATASET_DIR = "/home/yamil/doctorado/slchallenge/test_dataset_updated"
OUT_DIR = "outputs"
EMB_TRAIN = os.path.join(OUT_DIR, "embeddings.npy")          # (Ntr, 1024)
Y_TRAIN   = os.path.join(OUT_DIR, "y_train_lenses.npy")      # (Ntr,)
Z_TRAIN   = os.path.join(OUT_DIR, "z_train.npy")             # (≈98697,)  (si existe)

EMB_TEST_OUT = os.path.join(OUT_DIR, "embeddings_test_fixed.npy")
IDS_TEST_OUT = os.path.join(OUT_DIR, "embeddings_test_ids_fixed.npy")
FAILED_TXT   = os.path.join(OUT_DIR, "failed_ids.txt")
FAILED_CSV   = os.path.join(OUT_DIR, "failed_report.csv")
SUBMISSION   = os.path.join(OUT_DIR, "submission_final_1024_with_failures.csv")

# Modelo de features (DINOv2 vitl14) + batch/CPU fallback
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Limitar uso de VRAM y tamaño de lote por tu GPU (GTX 1070 u otra)
BATCH_SIZE = 16 if DEVICE == "cuda" else 8
NUM_WORKERS = 16  # threads para lectura FITS

# Clasificador / regressor
K_CLASS = 11          # mejor AUC validación previa
THRESH = 0.46         # threshold inclusivo (Youden de validación previa)
FILL_Z_ON_FAIL = -1.0 # photo-z para fallos

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- UTILIDADES FITS --------------------
def read_fits_band_any(path):
    """
    Lee 'la primera cosa útil' de un FITS:
    1) Si HDU[1] es tabla con columna band_*, la usa
    2) Si hay alguna HDU con .data 2D, la usa
    Devuelve np.ndarray (H,W) float32
    """
    with fits.open(path, memmap=False) as hdul:
        # Caso 1: tabla con columna band_*
        if len(hdul) > 1 and hasattr(hdul[1], "data") and hdul[1].data is not None:
            rec = hdul[1].data
            # buscar columna que tenga un array 2D
            for colname in rec.names:
                if colname.lower().startswith("band_"):
                    arr = rec[colname][0]
                    if isinstance(arr, np.ndarray) and arr.ndim == 2:
                        return arr.astype(np.float32)
        # Caso 2: imagen directa en alguna HDU
        for hdu in hdul:
            if hdu.data is not None and isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
                return hdu.data.astype(np.float32)
    raise ValueError(f"FITS vacío o sin 2D útil: {path}")

def load_rgb_object(base_id):
    """
    Intenta cargar r,g,i para base_id → tensor 3xHxW normalizado [0,1].
    Retorna (base_id, tensor, None) en éxito o (base_id, None, 'razón') en fallo.
    """
    bands = []
    missing = []
    for b in ["r", "g", "i"]:
        path = os.path.join(DATASET_DIR, f"{base_id}_{b}.fits")
        if not os.path.exists(path):
            missing.append(b)
            continue
        try:
            arr = read_fits_band_any(path)
            if arr.ndim != 2 or min(arr.shape) < 5:
                return base_id, None, f"banda {b} inválida shape={arr.shape}"
            # normaliza a [0,1] robusto
            mn, mx = np.min(arr), np.max(arr)
            scale = (mx - mn) if (mx - mn) > 0 else 1.0
            arr = (arr - mn) / scale
            bands.append(arr)
        except Exception as e:
            return base_id, None, f"error en banda {b}: {e}"
    if missing:
        return base_id, None, f"faltan bandas: {','.join(missing)}"
    if len(bands) != 3:
        return base_id, None, "no se pudieron formar 3 bandas"
    rgb = np.stack(bands, axis=0)  # 3xHxW
    return base_id, torch.tensor(rgb, dtype=torch.float32), None

# -------------------- MODELO DINOv2 --------------------
def get_dinov2_model():
    print(f"🧠 Cargando DINOv2 (vitl14) en {DEVICE}...")
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    except Exception:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval().to(DEVICE)
    print("✅ DINOv2 cargado.")
    return model

model = get_dinov2_model()

# DINOv2 espera 518x518, 3 canales, normalización similar a la oficial
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
])

# ------------------- LISTA DE IDS BASE -------------------
fits_ids = sorted({ "_".join(f.split("_")[:2]) for f in os.listdir(DATASET_DIR) if f.endswith("_r.fits") })
print(f"🔎 Detectados {len(fits_ids)} objetos (por _r.fits)")

# ------------------- EXTRACCIÓN DE EMBEDDINGS -------------------
embeddings = []
ids_ok = []
failed = []  # (id, reason)

pbar = tqdm(total=len(fits_ids), desc="Extrayendo embeddings")
for start in range(0, len(fits_ids), BATCH_SIZE):
    batch_ids = fits_ids[start:start+BATCH_SIZE]
    batch_imgs = []
    batch_idx_ok = []

    # carga paralela en CPU
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(load_rgb_object, bid): bid for bid in batch_ids}
        for fut in as_completed(futures):
            bid, img, err = fut.result()
            if err:
                failed.append((bid, err))
            else:
                batch_imgs.append(transform(img))
                batch_idx_ok.append(bid)
            pbar.update(1)

    if not batch_imgs:
        continue

    # a tensor y forward
    imgs_tensor = torch.stack(batch_imgs).to(DEVICE, non_blocking=True)
    try:
        with torch.no_grad():
            feats = model(imgs_tensor)  # [B, C] o [B, C, H, W] dependiendo del head
            if feats.ndim == 4:
                feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
            embs = feats.detach().cpu().numpy()
        embeddings.append(embs)
        ids_ok.extend(batch_idx_ok)
    except Exception as e:
        # si falla el batch completo, marcar todos como fallos
        for bid in batch_idx_ok:
            failed.append((bid, f"forward error: {e}"))
    finally:
        del imgs_tensor
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

pbar.close()

if len(embeddings) == 0:
    print("❌ No se generaron embeddings. Revisa rutas/VRAM/FITS.")
    sys.exit(1)

embeddings = np.vstack(embeddings)
ids_ok = np.array(ids_ok)
np.save(EMB_TEST_OUT, embeddings)
np.save(IDS_TEST_OUT, ids_ok)

print(f"✅ Embeddings guardados: {EMB_TEST_OUT} {embeddings.shape}")
print(f"✅ IDs guardados: {IDS_TEST_OUT} {ids_ok.shape}")
print(f"⚠️ Fallos: {len(failed)}")

# guardar fallos
if failed:
    with open(FAILED_TXT, "w") as f:
        for bid, why in failed:
            f.write(f"{bid}\t{why}\n")
    with open(FAILED_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "reason"])
        for bid, why in failed:
            w.writerow([bid, why])
    print(f"📝 Fallos → {FAILED_TXT} y {FAILED_CSV}")

# ------------------- CLASIFICACIÓN / REGRESIÓN -------------------
# Cargar entrenamiento
X_train = np.load(EMB_TRAIN)
y_train = np.load(Y_TRAIN)
if X_train.shape[0] != y_train.shape[0]:
    m = min(X_train.shape[0], y_train.shape[0])
    warnings.warn(f"[WARN] Mismatch train: truncando a {m}")
    X_train = X_train[:m]
    y_train = y_train[:m]

# Test OK = solo los que sí tienen embedding
X_test_ok = embeddings
ids_test_ok = ids_ok

# Escalado
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test_ok)

# KNN clasificador
clf = KNeighborsClassifier(n_neighbors=K_CLASS, weights="distance", n_jobs=-1)
clf.fit(X_train_sc, y_train)
probs_ok = clf.predict_proba(X_test_sc)[:, 1]
is_lens_ok = (probs_ok >= THRESH).astype(int)

# Regressor de photo-z (si existe z_train)
predz_ok = None
if os.path.exists(Z_TRAIN):
    z_train = np.load(Z_TRAIN)
    m = min(len(z_train), len(X_train_sc))
    if m < len(z_train):
        z_train = z_train[:m]
    if m < len(X_train_sc):
        X_train_sc_z = X_train_sc[:m]
    else:
        X_train_sc_z = X_train_sc
    reg = KNeighborsRegressor(n_neighbors=K_CLASS, weights="distance")
    reg.fit(X_train_sc_z, z_train)
    predz_ok = reg.predict(X_test_sc)
else:
    warnings.warn("[WARN] No se encontró z_train.npy. Se asignará -1.0 a predicted_z.")

# ------------------- ARMADO DEL CSV FINAL -------------------
# Para IDs fallados → is_lens = -99, prob_lens=0.0, predicted_z=-1.0
failed_ids = set(bid for bid, _ in failed)

all_ids_sorted = sorted(fits_ids)  # todos los IDs esperados por nombre
id_to_idx_ok = {bid: i for i, bid in enumerate(ids_test_ok)}

rows = []
for bid in all_ids_sorted:
    if bid in failed_ids:
        rows.append([bid, -99, -1.0])  # is_lens, predicted_z
    else:
        i = id_to_idx_ok.get(bid, None)
        if i is None:
            # si por alguna razón no está, trátalo como fallo
            rows.append([bid, -99, -1.0])
        else:
            il = int(is_lens_ok[i])
            z  = float(predz_ok[i]) if predz_ok is not None else FILL_Z_ON_FAIL
            rows.append([bid, il, z])

import pandas as pd
df = pd.DataFrame(rows, columns=["id", "is_lens", "predicted_z"])
df.to_csv(SUBMISSION, index=False)
print(f"💾 CSV final → {SUBMISSION}")

# Resumen
n_total   = len(all_ids_sorted)
n_failed  = sum(1 for r in rows if r[1] == -99)
n_lenses  = sum(1 for r in rows if r[1] == 1)
n_nonlens = sum(1 for r in rows if r[1] == 0)
print("\n📊 Resumen:")
print(f"  • Total objetos        : {n_total}")
print(f"  • Fallidos (is_lens=-99): {n_failed} ({100.0*n_failed/n_total:.2f}%)")
print(f"  • Predichos LENSES (1)  : {n_lenses} ({100.0*n_lenses/n_total:.2f}%)")
print(f"  • Predichos NON-LENS (0): {n_nonlens} ({100.0*n_nonlens/n_total:.2f}%)")