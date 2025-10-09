#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
challenge_submission_v2.py — Generar archivo submission.csv para el challenge

Este script usa los embeddings de AstroCLIP y los catálogos actualizados
para producir el archivo de submission final con el formato:

    ID,is_lens,z_phot

Donde:
- ID = identificador del objeto (Lens ID u Object ID).
- is_lens = 1 si es lente, 0 si es non-lens.
- z_phot = redshift fotométrico predicho con KNN.

Conceptos ML:
- Se entrena un KNN Regressor con embeddings.
- Se estima el redshift fotométrico tanto para lentes como para non-lentes.
- Se genera el archivo CSV con todas las predicciones.
"""

import os
import numpy as np
from astropy.table import Table
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# ---------------- Config ----------------
EMB_PATH   = "outputs/embeddings.npy"
LAB_PATH   = "outputs/labels.npy"

LENSES_META    = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/parameters_updated.fits"
NONLENSES_META = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_nonlenses/parameters.fits"

SUBMISSION_PATH = "outputs/submission.csv"
K = 5

# ---------------- Helpers ----------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ---------------- Load data ----------------
print("🔹 Cargando embeddings y labels...")
X = np.load(EMB_PATH)
labels = np.load(LAB_PATH)

print("🔹 Leyendo catálogo lenses...")
cat_lens = Table.read(LENSES_META, format="fits")
z_lens = np.array(cat_lens["zlens"], dtype=np.float32)
id_lens = np.array(cat_lens["Lens ID"]).astype(str)

print("🔹 Leyendo catálogo non-lenses...")
cat_non = Table.read(NONLENSES_META, format="fits")
z_non = np.array(cat_non["z_central"], dtype=np.float32)
id_non = np.array(cat_non["Object ID"]).astype(str)

# Separar embeddings por clase
X_lens, X_non = X[labels == 1], X[labels == 0]

# Alinear tamaños
n_lens = min(len(z_lens), len(X_lens))
n_non = min(len(z_non), len(X_non))
X_lens, z_lens, id_lens = X_lens[:n_lens], z_lens[:n_lens], id_lens[:n_lens]
X_non, z_non, id_non = X_non[:n_non], z_non[:n_non], id_non[:n_non]

# Dataset combinado
X_all = np.vstack([X_lens, X_non])
y_all = np.concatenate([z_lens, z_non])
id_all = np.concatenate([id_lens, id_non])
labels_all = np.concatenate([np.ones(n_lens), np.zeros(n_non)])

print(f"✅ Dataset combinado: {X_all.shape}, {y_all.shape}")

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# Entrenar KNN
knn = KNeighborsRegressor(n_neighbors=K, weights="distance")
knn.fit(X_train, y_train)

# Predicciones finales para TODO el dataset
z_pred = knn.predict(X_all)

# Métricas en test
y_pred_test = knn.predict(X_test)
print(f"📊 MAE={mae(y_test, y_pred_test):.4f}, RMSE={rmse(y_test, y_pred_test):.4f}")

# ---------------- Guardar Submission ----------------
# ---------------- Guardar Submission ----------------
os.makedirs("outputs", exist_ok=True)

with open(SUBMISSION_PATH, "w") as f:
    f.write("ID,is_lens,z_phot\n")
    for obj_id, is_lens, z in zip(id_all, labels_all.astype(int), z_pred):
        f.write(f"{obj_id},{is_lens},{z:.5f}\n")

print(f"📁 Submission guardado en {SUBMISSION_PATH}")

