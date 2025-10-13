#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_embeddings_astroclip_real_parallel.py

Genera embeddings para el dataset real utilizando DINOv2 (vitl14)
cargado automáticamente desde torch.hub.
Procesa las imágenes FITS (bandas r, g, i) en paralelo.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip_test_dataset/test_dataset"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EMB_PATH = os.path.join(OUTPUT_DIR, "embeddings_real_v3.npy")
IDS_PATH = os.path.join(OUTPUT_DIR, "ids_real_v3.npy")

print(f"🧠 Usando dispositivo: {DEVICE}")

# ---------------------------------------------------------------
# Cargar modelo DINOv2 desde torch.hub
# ---------------------------------------------------------------
print("🔹 Cargando modelo DINOv2 (vitl14) desde torch.hub...")
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(DEVICE)
dinov2_model.eval()
print("✅ DINOv2 cargado exitosamente desde PyTorch Hub\n")

# ---------------------------------------------------------------
# Función para leer una imagen FITS combinando bandas r, g, i
# ---------------------------------------------------------------
def read_fits_rgb(base_id):
    bands = ["r", "g", "i"]
    img_stack = []
    for b in bands:
        path = os.path.join(DATASET_DIR, f"{base_id}_{b}.fits")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe la banda {b} para {base_id}")
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"FITS vacío: {path}")
            data = np.nan_to_num(data).astype(np.float32)
            img_stack.append(data)

    img = np.stack(img_stack, axis=0)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    img_tensor = F.interpolate(img_tensor, size=(518, 518), mode="bilinear", align_corners=False)
    return img_tensor

# ---------------------------------------------------------------
# Función para extraer embeddings de un archivo base_id
# ---------------------------------------------------------------
def process_object(base_id):
    try:
        img_tensor = read_fits_rgb(base_id)
        with torch.no_grad():
            feats = dinov2_model.forward_features(img_tensor)
            emb = feats["x_norm_clstoken"].squeeze().cpu().numpy()
        return base_id, emb
    except Exception as e:
        warnings.warn(f"[WARN] Error procesando {base_id}: {e}")
        return None

# ---------------------------------------------------------------
# Listar objetos base (sin banda)
# ---------------------------------------------------------------
all_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith("_r.fits")])
base_ids = [os.path.splitext(f)[0][:-2] for f in all_files]
print(f"🔎 Se encontraron {len(base_ids)} objetos con bandas r,g,i\n")

# ---------------------------------------------------------------
# Procesamiento paralelo
# ---------------------------------------------------------------
embeddings = []
ids = []
MAX_WORKERS = min(8, os.cpu_count() or 4)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_object, base_id): base_id for base_id in base_ids}
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Extrayendo embeddings")):
        result = future.result()
        if result is not None:
            obj_id, emb = result
            ids.append(obj_id)
            embeddings.append(emb)

# ---------------------------------------------------------------
# Guardar resultados
# ---------------------------------------------------------------
if len(embeddings) == 0:
    raise RuntimeError("❌ No se generaron embeddings. Verifica extensiones o FITS corruptos.")

embeddings = np.vstack(embeddings)
ids = np.array(ids)

np.save(EMB_PATH, embeddings)
np.save(IDS_PATH, ids)

print(f"✅ Embeddings guardados en: {EMB_PATH}")
print(f"✅ IDs guardados en: {IDS_PATH}")
print(f"📊 Total procesado: {len(ids)} objetos.")
