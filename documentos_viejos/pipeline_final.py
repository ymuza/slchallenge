#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline final para el challenge (embeddings 1024 reales ~96k):
- Selección de K (KNN) por AUC-ROC en validación estratificada.
- Umbral óptimo por Youden en ROC.
- KNN regressor para predicted_z en real.
- CSV final + gráficas y métricas.

Entradas esperadas (existentes):
  outputs/embeddings.npy              -> train embeddings (N_train x 1024)
  outputs/y_train_lenses.npy          -> labels 0/1 (N_train,)
  outputs/z_train.npy                 -> redshift real de train (N_train,)
  outputs/embeddings_1024.npy         -> embeddings reales (N_real x 1024)
  outputs/embeddings_1024_ids.npy     -> ids reales (N_real,)

Salidas:
  outputs/submission_final_1024.csv
  outputs/performance_1024_final/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)

# ----------------- Config -----------------
os.makedirs("outputs/performance_1024_final", exist_ok=True)

TRAIN_X_PATH = "outputs/embeddings.npy"
TRAIN_Y_PATH = "outputs/y_train_lenses.npy"
TRAIN_Z_PATH = "outputs/z_train.npy"

REAL_X_PATH  = "outputs/embeddings_test_fixed.npy"
REAL_ID_PATH = "outputs/embeddings_test_ids_fixed.npy"

SUBMISSION_OUT = "outputs/final.csv"
FIG_DIR = "outputs/performance_1024_final"

K_GRID = [3, 5, 7, 9, 11]
VAL_SIZE = 0.2
RANDOM_STATE = 42

# ----------------- Carga -----------------
print("🔹 Cargando datos...")
X_train = np.load(TRAIN_X_PATH)
y_train = np.load(TRAIN_Y_PATH)
z_train = np.load(TRAIN_Z_PATH)
X_real  = np.load(REAL_X_PATH)
ids_real = np.load(REAL_ID_PATH, allow_pickle=True)

# Alinear longitudes por seguridad
n_common = min(len(X_train), len(y_train), len(z_train))
if (len(X_train), len(y_train), len(z_train)) != (n_common, n_common, n_common):
    print(f"⚠️ Mismatch detectado. Truncando a {n_common}.")
X_train = X_train[:n_common]
y_train = y_train[:n_common]
z_train = z_train[:n_common]

# Tipos y escalado (float32 ahorra RAM; sklearn convierte interno a float64, pero reducimos copias)
X_train = X_train.astype(np.float32)
X_real  = X_real.astype(np.float32)

print(f"✅ Train: {X_train.shape}, Real: {X_real.shape}, Labels: {y_train.shape}, z_train: {z_train.shape}")

# ----------------- Split de validación para métricas y ajuste de threshold -----------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
)

# Escalado (fit con TRAIN completo es más estable para producción)
scaler = StandardScaler()
scaler.fit(X_train)
X_tr_s  = scaler.transform(X_tr)
X_val_s = scaler.transform(X_val)
X_real_s = scaler.transform(X_real)

# ----------------- Selección de K por AUC-ROC en CV (rápida) -----------------
print("🧠 Buscando mejor K por AUC-ROC...")
best_k = None
best_auc = -np.inf

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
for k in K_GRID:
    aucs = []
    for tr_idx, te_idx in cv.split(X_tr_s, y_tr):
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        clf.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        proba = clf.predict_proba(X_tr_s[te_idx])[:, 1]
        aucs.append(roc_auc_score(y_tr[te_idx], proba))
    mean_auc = float(np.mean(aucs))
    print(f"  • k={k:<2d} → AUC={mean_auc:.4f}")
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_k = k

print(f"✅ Mejor k={best_k} (AUC medio={best_auc:.4f})")

# ----------------- Entrenar con k óptimo y ajustar threshold por Youden en ROC (validación) -----------------
clf = KNeighborsClassifier(n_neighbors=best_k, weights="distance", n_jobs=-1)
clf.fit(X_tr_s, y_tr)
val_proba = clf.predict_proba(X_val_s)[:, 1]

# ROC & PR (validación)
fpr, tpr, thr = roc_curve(y_val, val_proba)
auc_val = roc_auc_score(y_val, val_proba)
precision, recall, pr_thr = precision_recall_curve(y_val, val_proba)
ap_val = average_precision_score(y_val, val_proba)

# Threshold óptimo por Youden
youden = tpr - fpr
thr_opt = thr[np.argmax(youden)]
print(f"🎯 Threshold óptimo (Youden) en validación: {thr_opt:.4f} | AUC={auc_val:.4f} | AP={ap_val:.4f}")

# Matriz de confusión en validación (con thr_opt)
y_val_pred = (val_proba >= thr_opt).astype(int)
cm = confusion_matrix(y_val, y_val_pred)

# --- Figuras (validación) ---
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
plt.plot([0,1],[0,1],'k--', lw=1)
# marcar threshold óptimo en la curva
idx_opt = np.argmax(youden)
plt.scatter([fpr[idx_opt]], [tpr[idx_opt]], s=60, marker='o', edgecolor='k', facecolor='none', label=f"thr*={thr_opt:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=150)

plt.figure(figsize=(5,5))
plt.plot(recall, precision, lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall (AP={ap_val:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "precision_recall_curve.png"), dpi=150)

plt.figure(figsize=(5,5))
Disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
Disp.plot(values_format='d', cmap="Blues")
plt.title(f"Confusion Matrix (thr*={thr_opt:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)

# ----------------- Entrenar regressor para predicted_z (validación y real) -----------------
print("📐 Entrenando KNN regressor para photo-z...")
knn_z = KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)
knn_z.fit(scaler.transform(X_train), z_train)

# Validación (para scatter true vs pred z)
z_val_pred = knn_z.predict(X_val_s)

plt.figure(figsize=(5,5))
plt.scatter(z_train[:20000], knn_z.predict(scaler.transform(X_train[:20000])), s=5, alpha=0.2)
plt.plot([z_train.min(), z_train.max()], [z_train.min(), z_train.max()], 'r--', lw=1)
plt.xlabel("True z (train subset)")
plt.ylabel("Predicted z (train subset)")
plt.title(f"Photo-z KNN (k={best_k})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "true_vs_pred_z_train_subset.png"), dpi=150)

# Validación pura scatter (solo val split)
plt.figure(figsize=(5,5))
plt.scatter(z_train[y_train.shape[0]-len(y_val):y_train.shape[0]][:len(z_val_pred)], z_val_pred, s=8, alpha=0.3)
zmin, zmax = float(np.min(z_train)), float(np.max(z_train))
plt.plot([zmin, zmax], [zmin, zmax], 'r--', lw=1)
plt.xlabel("True z (val)")
plt.ylabel("Predicted z (val)")
plt.title(f"Photo-z KNN (val split, k={best_k})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "true_vs_pred_z_val.png"), dpi=150)

# ----------------- Inferencia en REAL -----------------
print("🚀 Inferencia en datos reales...")
proba_real = clf.predict_proba(X_real_s)[:, 1]
z_real_pred = knn_z.predict(X_real_s)
is_lens_real = (proba_real >= thr_opt).astype(int)

# Histograma de prob_lens (real)
plt.figure(figsize=(6,4))
plt.hist(proba_real, bins=50, alpha=0.8)
plt.xlabel("prob_lens")
plt.ylabel("count")
plt.title("Distribución de prob_lens (REAL)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "prob_histogram_real.png"), dpi=150)

# ----------------- CSV final -----------------
submission = pd.DataFrame({
    "id": ids_real.astype(str),
    "is_lens": is_lens_real.astype(int),
    "predicted_z": z_real_pred.astype(float),
    "prob_lens": proba_real.astype(float),
})

submission.to_csv(SUBMISSION_OUT, index=False)
lens_pct = 100.0 * submission["is_lens"].mean()
print(f"\n💾 CSV guardado: {SUBMISSION_OUT}")
print(f"   → Lentes estimadas: {lens_pct:.2f}%  (thr*={thr_opt:.2f}, k={best_k})")
print(f"📊 Figuras en: {FIG_DIR}/")