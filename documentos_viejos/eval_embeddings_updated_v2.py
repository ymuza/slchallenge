#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_embeddings_updated_v2.py — Genera embeddings para dataset de test usando DINOv2

Este script:
1️⃣ Carga el modelo DINOv2 (ViT-L/14) desde PyTorch Hub (sin pesos manuales).
2️⃣ Procesa imágenes FITS del dataset de test.
3️⃣ Extrae embeddings 1024-D de cada tripleta (r, g, i).
4️⃣ Guarda resultados en `outputs/embeddings_test.npy`, `outputs/ids_test.npy`, `outputs/labels_test.npy`.
5️⃣ Muestra estadísticas finales y ejemplos para validación rápida.

Requiere:
- torch >= 2.0
- dinov2
- astropy
- tqdm
- numpy

Autor: Yamil 🧠
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensor_dataset_indexmap import GalaxyBatchDataset

# ---------------- CONFIG ----------------
TEST_ROOT = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip_test_dataset"
OUTPUT_DIR = "../outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- 1️⃣ CARGAR MODELO ----------------
print("🔹 Cargando modelo DINOv2 (ViT-L/14) desde PyTorch Hub...")
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
dinov2_model.eval().to(DEVICE)
print("✅ Modelo cargado correctamente en", DEVICE)

# ---------------- 2️⃣ PROCESAR DATASETS ----------------
subfolders = [
    os.path.join(TEST_ROOT, d)
    for d in os.listdir(TEST_ROOT)
    if os.path.isdir(os.path.join(TEST_ROOT, d))
]

print(f"\n🔎 Carpetas encontradas: {subfolders}")

embeddings_all = []
ids_all = []
labels_all = []

for folder in subfolders:
    label = 1 if "lens" in folder.lower() else 0
    print(f"\n🔵 Procesando carpeta: {folder}  |  Label={label}")

    dataset = GalaxyBatchDataset(root=folder, meta_fits=None, size=224)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch in tqdm(loader, desc=os.path.basename(folder)):
            imgs = batch["image"].to(DEVICE)

            # 🔧 Redimensionar si no es múltiplo de 14 (ej: 224×224)
            if imgs.shape[-1] != 224:
                imgs = torch.nn.functional.interpolate(
                    imgs, size=(224, 224), mode="bilinear", align_corners=False
                )

            base_ids = batch["id"]
            feats = dinov2_model(imgs)

            if isinstance(feats, (tuple, list)):
                feats = feats[0]

            embeddings_all.append(feats.cpu().numpy())
            ids_all.extend(base_ids)
            labels_all.extend([label] * len(base_ids))

# ---------------- 3️⃣ GUARDAR SALIDAS ----------------
embeddings_all = np.concatenate(embeddings_all, axis=0)
ids_all = np.array(ids_all)
labels_all = np.array(labels_all)

np.save(os.path.join(OUTPUT_DIR, "embeddings_test.npy"), embeddings_all)
np.save(os.path.join(OUTPUT_DIR, "ids_test.npy"), ids_all)
np.save(os.path.join(OUTPUT_DIR, "labels_test.npy"), labels_all)

print(f"\n✅ Embeddings guardados: {embeddings_all.shape}")
print(f"✅ IDs guardados: {ids_all.shape}")
print(f"✅ Labels guardados: {labels_all.shape}")

# ---------------- 4️⃣ VALIDACIÓN FINAL ----------------
print("\n📊 --- VALIDACIÓN DE RESULTADOS ---")
print(f"Total de imágenes procesadas: {len(ids_all)}")
print(f"Dimensión del embedding: {embeddings_all.shape[1]}")
print(f"Lentes: {np.sum(labels_all == 1)}, No-lentes: {np.sum(labels_all == 0)}")

# Ejemplo de IDs
print("\n🧾 Ejemplos de IDs:")
for ex in ids_all[:10]:
    print("  ", ex)

# Estadísticas de embeddings
mean_vec = np.mean(embeddings_all, axis=0)
std_vec = np.std(embeddings_all, axis=0)
print(f"\n📈 Promedio de embeddings: {mean_vec.mean():.5f} ± {std_vec.mean():.5f}")

print(f"\n📁 Archivos guardados en: {os.path.abspath(OUTPUT_DIR)}")
print("✅ Proceso completado exitosamente.")

