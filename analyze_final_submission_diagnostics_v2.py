# analyze_final_submission_diagnostics.py
# -------------------------------------------------------------
# Diagnóstico final del modelo KNN calibrado:
# - ROC curve + AUC
# - Threshold óptimo (Youden J)
# - Comparación z_train vs z_pred
# - Calibración de probabilidades
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.calibration import calibration_curve

# --- Configuración ---
os.makedirs("outputs/final_diagnostics", exist_ok=True)

CSV_PATH = "outputs/submission_final_knn_calibratedcsv"
ZTRAIN_PATH = "outputs/z_train.npy"

print("🔹 Cargando datos...")

# --- Carga ---
df = pd.read_csv(CSV_PATH)
z_train = np.load(ZTRAIN_PATH)

print(f"✅ CSV cargado: {df.shape[0]} filas")
print(f"✅ z_train.npy cargado: {len(z_train)} valores")

# --- Verificación de columnas ---
expected_cols = {"id", "is_lens", "predicted_z", "prob_lens"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"❌ Faltan columnas en el CSV: {missing}")

# --- Extracción de valores ---
y_true = df["is_lens"].values
y_score = df["prob_lens"].values
z_pred = df["predicted_z"].values

# --- Métricas ROC ---
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# --- Mejor threshold (Youden J) ---
J = tpr - fpr
best_idx = np.argmax(J)
best_thresh = thresholds[best_idx]
best_tpr, best_fpr = tpr[best_idx], fpr[best_idx]

print(f"\n📈 AUC = {roc_auc:.4f}")
print(f"🏁 Mejor threshold (Youden J): {best_thresh:.4f}")
print(f"   → TPR={best_tpr:.3f}, FPR={best_fpr:.3f}")

# --- Gráfico ROC ---
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle="--")
plt.scatter(best_fpr, best_tpr, color='red', s=60, label=f"Threshold óptimo = {best_thresh:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC – KNN Calibrado")
plt.legend()
plt.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/final_diagnostics/roc_curve_knn_final.png", dpi=250)
plt.close()

# --- Curva de calibración ---
prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Modelo calibrado')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("Probabilidad predicha")
plt.ylabel("Fracción real de lentes")
plt.title("Curva de calibración – KNN calibrado")
plt.legend()
plt.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/final_diagnostics/calibration_curve_knn_final.png", dpi=250)
plt.close()

# --- Histograma de scores ---
plt.figure(figsize=(8, 5))
sns.histplot(y_score[y_true == 0], bins=40, color='blue', alpha=0.6, label="No-lentes")
sns.histplot(y_score[y_true == 1], bins=40, color='orange', alpha=0.6, label="Lentes")
plt.axvline(best_thresh, color='red', ls='--', lw=2, label=f"Threshold óptimo = {best_thresh:.3f}")
plt.xlabel("Probabilidad de ser lente (KNN)")
plt.ylabel("Frecuencia")
plt.title("Distribución de prob_lens por clase")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/final_diagnostics/hist_prob_lens_knn_final.png", dpi=250)
plt.close()

# --- Comparación de redshifts ---
plt.figure(figsize=(8, 5))
sns.kdeplot(z_train, label="z_train (simulado)", fill=True, alpha=0.4)
sns.kdeplot(z_pred, label="z_pred (submission)", fill=True, alpha=0.4)
plt.xlabel("Redshift")
plt.ylabel("Densidad")
plt.title("Comparación de distribuciones de redshift")
plt.legend()
plt.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/final_diagnostics/redshift_comparison_knn_final.png", dpi=250)
plt.close()

print("\n✅ Gráficos generados en: outputs/final_diagnostics/")
print("📂 Archivos creados:")
print("  • roc_curve_knn_final.png")
print("  • calibration_curve_knn_final.png")
print("  • hist_prob_lens_knn_final.png")
print("  • redshift_comparison_knn_final.png")
