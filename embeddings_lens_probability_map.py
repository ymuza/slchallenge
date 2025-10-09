#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embeddings_lens_probability_map.py
----------------------------------
Visualiza el espacio de embeddings (train + real) usando PCA
y colorea los puntos del dataset real según su probabilidad
de ser LENS (estimada por KNN en el espacio de entrenamiento).

Entradas:
  - outputs/embeddings_updated.npy      → embeddings del train
  - outputs/labels_updated.npy          → labels del train (0/1)
  - outputs/embeddings_test.npy         → embeddings del test real

Salida:
  - outputs/embeddings_lens_probability_map.png
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import os

# ---------------- Configuración ----------------
TRAIN_EMB = "outputs/embeddings_updated.npy"
TRAIN_LAB = "outputs/labels_updated.npy"
REAL_EMB  = "outputs/embeddings_test.npy"
OUT_PATH  = "documentos_viejos/embeddings_lens_probability_map.png"
K = 15  # número de vecinos para estimar probabilidad

# ---------------- Cargar datos ----------------
print("🔹 Cargando embeddings y labels...")
X_train = np.load(TRAIN_EMB)
y_train = np.load(TRAIN_LAB)
X_real  = np.load(REAL_EMB)

print(f"✅ Train: {X_train.shape}, Real: {X_real.shape}")
print(f"   Lenses: {np.sum(y_train==1)}, Non-lenses: {np.sum(y_train==0)}")

# ---------------- Calcular probabilidad lens ----------------
print("🔹 Entrenando KNN...")
knn = KNeighborsClassifier(n_neighbors=K, weights="distance")
knn.fit(X_train, y_train)

print("🔹 Prediciendo probabilidades en conjunto real...")
probs = knn.predict_proba(X_real)[:, 1]  # probabilidad de ser lens

# ---------------- Reducción PCA para visualización ----------------
print("🔹 Reducción PCA para visualizar embeddings...")
pca = PCA(n_components=2, random_state=42)
X_all = np.vstack([X_train, X_real])
X_proj = pca.fit_transform(X_all)

# Separar índices
n_train = len(X_train)
X_train_proj = X_proj[:n_train]
X_real_proj  = X_proj[n_train:]

# ---------------- Plot ----------------
print("📊 Generando gráfico...")

plt.figure(figsize=(8, 8))

# Fondo: datos del entrenamiento (gris claro)
plt.scatter(X_train_proj[y_train == 0, 0],
            X_train_proj[y_train == 0, 1],
            s=2, color="lightgray", alpha=0.3, label="Non-lens (train)")
plt.scatter(X_train_proj[y_train == 1, 0],
            X_train_proj[y_train == 1, 1],
            s=2, color="orange", alpha=0.3, label="Lens (train)")

# Dataset real, coloreado por probabilidad
sc = plt.scatter(X_real_proj[:, 0], X_real_proj[:, 1],
                 c=probs, cmap="viridis", s=3, alpha=0.7,
                 label="Real (test, prob. lens)")

cbar = plt.colorbar(sc)
cbar.set_label("Probabilidad de ser Lens")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Mapa de probabilidad de Lenses (espacio de embeddings)")
plt.legend(markerscale=5)
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig(OUT_PATH, dpi=200)
print(f"✅ Gráfico guardado en {OUT_PATH}")
