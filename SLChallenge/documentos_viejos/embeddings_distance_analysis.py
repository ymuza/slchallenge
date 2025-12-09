#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embeddings_distance_analysis.py
Analiza si los embeddings reales (test) se parecen más a los de lentes o no-lentes del conjunto de entrenamiento.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# ---------------- Config ----------------
EMB_TRAIN_PATH = "../SLChallenge/outputs/embeddings.npy"
LABELS_TRAIN_PATH = "../SLChallenge/outputs/labels.npy"
EMB_REAL_PATH = "../SLChallenge/outputs/embeddings_test.npy"
K = 10  # vecinos

# ---------------- Load ----------------
print("🔹 Cargando embeddings...")
X_train = np.load(EMB_TRAIN_PATH)
y_train = np.load(LABELS_TRAIN_PATH)
X_real = np.load(EMB_REAL_PATH)

print(f"✅ Train: {X_train.shape}, Real: {X_real.shape}")
print(f"   Lenses: {np.sum(y_train==1)}, Non-lenses: {np.sum(y_train==0)}")

# ---------------- PCA Visualization ----------------
print("🔹 Reducción PCA para visualización...")
pca = PCA(n_components=2)
X_proj = pca.fit_transform(np.vstack([X_train, X_real]))

X_proj_train = X_proj[:len(X_train)]
X_proj_real = X_proj[len(X_train):]

plt.figure(figsize=(7, 7))
plt.scatter(X_proj_train[y_train==0, 0], X_proj_train[y_train==0, 1],
            s=3, alpha=0.2, label="Non-lens train", color="steelblue")
plt.scatter(X_proj_train[y_train==1, 0], X_proj_train[y_train==1, 1],
            s=3, alpha=0.3, label="Lens train", color="orange")
plt.scatter(X_proj_real[:, 0], X_proj_real[:, 1],
            s=3, alpha=0.3, label="Real test", color="green")
plt.legend()
plt.title("PCA de embeddings (train vs real)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig("outputs/pca_train_vs_real.png", dpi=150)
plt.close()

# ---------------- Distancia a vecinos ----------------
print(f"🔹 Buscando {K} vecinos más cercanos en train...")
nn = NearestNeighbors(n_neighbors=K, algorithm="auto", metric="euclidean")
nn.fit(X_train)

dists, idxs = nn.kneighbors(X_real, return_distance=True)
mean_dists = np.mean(dists, axis=1)
neighbor_labels = y_train[idxs]

# Mayoría de labels entre los vecinos
majority_vote = (np.mean(neighbor_labels, axis=1) >= 0.5).astype(int)

# ---------------- Estadísticas ----------------
lens_like = np.sum(majority_vote == 1)
nonlens_like = np.sum(majority_vote == 0)

print("📊 Resultados:")
print(f"  • Real test total: {len(X_real)}")
print(f"  • Más parecidos a LENSES: {lens_like} ({lens_like/len(X_real)*100:.2f}%)")
print(f"  • Más parecidos a NON-LENSES: {nonlens_like} ({nonlens_like/len(X_real)*100:.2f}%)")
print(f"  • Distancia media global: {np.mean(mean_dists):.4f}")

# ---------------- Histogramas ----------------
plt.figure(figsize=(8, 4))
plt.hist(mean_dists[majority_vote==1], bins=50, alpha=0.6, label="Cercanos a Lenses", color="orange")
plt.hist(mean_dists[majority_vote==0], bins=50, alpha=0.6, label="Cercanos a Non-lenses", color="steelblue")
plt.xlabel("Distancia promedio a vecinos")
plt.ylabel("Frecuencia")
plt.title(f"Distribución de distancias a {K} vecinos más cercanos")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/distance_distribution.png", dpi=150)
plt.close()

print("✅ Gráficos guardados en 'outputs/'")
