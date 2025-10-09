#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
challenge_submission_v3_isLens.py — Genera submission.csv con campo is_lens

Este script toma los embeddings del dataset real (test),
predice el redshift fotométrico con un modelo KNN entrenado
sobre datos etiquetados (parameters_updated.fits + parameters.fits),
e incluye el campo `is_lens` (1 = lente, 0 = no lente).

Entradas:
- outputs/embeddings_test.npy
- outputs/ids_test.npy
- outputs/labels_test.npy
- outputs/embeddings.npy
- outputs/labels.npy
- parameters_updated.fits (para lenses)
- parameters.fits (para non-lenses)

Salida:
- submission_real.csv con columnas: id, is_lens, predicted_z
"""

import os
import numpy as np
import pandas as pd
from astropy.table import Table
from sklearn.neighbors import KNeighborsRegressor

# ---------------- CONFIG ----------------
EMB_TRAIN_PATH   = "outputs/embeddings.npy"
LABELS_TRAIN_PATH = "outputs/labels.npy"
EMB_TEST_PATH    = "outputs/embeddings_test.npy"
IDS_TEST_PATH    = "outputs/ids_test.npy"
LABELS_TEST_PATH = "outputs/labels_test.npy"

LENSES_META      = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/parameters_updated.fits"
NONLENSES_META   = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_nonlenses/parameters.fits"
OUTPUT_CSV       = "outputs/submission_real.csv"
K                = 5

# ---------------- FUNCIONES ----------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ---------------- LOAD TRAIN ----------------
print("🔹 Cargando embeddings y labels de entrenamiento...")
X_train = np.load(EMB_TRAIN_PATH)
labels_train = np.load(LABELS_TRAIN_PATH)

print("🔹 Cargando catálogos de entrenamiento (lenses y non-lenses)...")
cat_lens = Table.read(LENSES_META, format="fits")
z_lens = np.array(cat_lens["zlens"], dtype=np.float32)
cat_non = Table.read(NONLENSES_META, format="fits")
z_non = np.array(cat_non["z_central"], dtype=np.float32)

# Separar embeddings por tipo
X_lens = X_train[labels_train == 1]
X_non  = X_train[labels_train == 0]

# Emparejar tamaños con catálogos
n_lens = min(len(z_lens), len(X_lens))
n_non  = min(len(z_non), len(X_non))
X_lens, z_lens = X_lens[:n_lens], z_lens[:n_lens]
X_non,  z_non  = X_non[:n_non],  z_non[:n_non]

# Combinar todo
X_all = np.vstack([X_lens, X_non])
y_all = np.concatenate([z_lens, z_non])
print(f"✅ Dataset combinado de entrenamiento: {X_all.shape}, {y_all.shape}")

# ---------------- ENTRENAR KNN ----------------
print(f"🔹 Entrenando modelo KNN (k={K})...")
knn = KNeighborsRegressor(n_neighbors=K, weights="distance")
knn.fit(X_all, y_all)

# ---------------- LOAD TEST ----------------
print("🔹 Cargando embeddings de test reales...")
X_test = np.load(EMB_TEST_PATH)
ids_test = np.load(IDS_TEST_PATH)
labels_test = np.load(LABELS_TEST_PATH)

print(f"✅ Dataset de test: {X_test.shape}, {ids_test.shape}, labels={labels_test.shape}")

# ---------------- PREDICCIÓN ----------------
print("🔹 Prediciendo redshifts...")
y_pred = knn.predict(X_test)

# Evitar NaN
mask_valid = ~np.isnan(y_pred)
y_pred = y_pred[mask_valid]
ids_test = ids_test[mask_valid]
labels_test = labels_test[mask_valid]

# ---------------- EXPORTAR ----------------
df = pd.DataFrame({
    "id": ids_test,
    "is_lens": labels_test.astype(int),
    "predicted_z": y_pred
})

# Asegurar que los redshifts estén dentro del rango físico
df["predicted_z"] = np.clip(df["predicted_z"], 0, 2.0)

# Guardar CSV final
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Archivo generado: {OUTPUT_CSV} ({len(df)} filas)")
print(df.head(10))
