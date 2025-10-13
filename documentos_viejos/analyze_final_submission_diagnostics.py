"""
Este archivo genera el csv para el challenge. El% de lentes maso ajustado está en entre
1.5% y 2% del total

"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# --- Paths ---
OUTPUT_DIR = "../outputs/final_diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUB_PATH = "../outputs/submission_final_knn_calibrated.csv"
Z_TRAIN_PATH = "../outputs/z_train.npy"

print("🔹 Cargando datos...")
df = pd.read_csv(SUB_PATH)
z_train = np.load(Z_TRAIN_PATH)

# --- Validaciones ---
print(f"✅ submission_final_knn_calibrated.csv → {len(df)} filas")
print(f"✅ z_train.npy → {len(z_train)} valores")

# --- 1️⃣ Curva ROC ---
if "prob_lens" in df.columns:
    y_true = (df["is_lens"] == 1).astype(int)
    y_score = df["prob_lens"].values
else:
    raise ValueError("❌ El CSV no contiene la columna 'prob_lens' con las probabilidades del KNN calibrado.")

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Threshold óptimo (máx. diferencia TPR-FPR)
opt_idx = np.argmax(tpr - fpr)
opt_thresh = thresholds[opt_idx]
opt_tpr, opt_fpr = tpr[opt_idx], fpr[opt_idx]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.scatter(opt_fpr, opt_tpr, color="red", s=60, label=f"Mejor threshold = {opt_thresh:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - KNN Calibrado")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
roc_path = os.path.join(OUTPUT_DIR, "roc_curve_final.png")
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"📈 Curva ROC guardada: {roc_path}")

# --- 2️⃣ Distribución de redshifts ---
plt.figure(figsize=(8, 6))
plt.hist(z_train, bins=80, alpha=0.5, color="blue", label="Train (simulado)")
plt.hist(df["predicted_z"], bins=80, alpha=0.5, color="orange", label="Predicho (real calibrado)")
plt.xlabel("Redshift (z)")
plt.ylabel("Frecuencia")
plt.title("Distribución de redshifts - Train vs Predicho")
plt.legend()
zdist_path = os.path.join(OUTPUT_DIR, "redshift_distribution_comparison.png")
plt.savefig(zdist_path, dpi=200)
plt.close()
print(f"📊 Distribución de redshifts guardada: {zdist_path}")

# --- 3️⃣ Proporción de lentes detectadas ---
counts = df["is_lens"].value_counts(normalize=True) * 100
plt.figure(figsize=(5, 5))
plt.pie(
    counts,
    labels=[f"Non-Lens ({counts.get(0, 0):.1f}%)", f"Lens ({counts.get(1, 0):.1f}%)"],
    colors=["#66b3ff", "#ff6666"],
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Proporción de lentes detectadas en submission final")
pie_path = os.path.join(OUTPUT_DIR, "lens_ratio_final.png")
plt.savefig(pie_path, dpi=200)
plt.close()
print(f"📊 Gráfico de proporción guardado: {pie_path}")

# --- 4️⃣ Métricas finales ---
precision, recall, _ = precision_recall_curve(y_true, y_score)
best_f1_idx = np.argmax(2 * (precision * recall) / (precision + recall + 1e-6))
best_prec, best_rec = precision[best_f1_idx], recall[best_f1_idx]

print("\n✅ MÉTRICAS FINALES:")
print(f"  • AUC ROC: {roc_auc:.4f}")
print(f"  • Threshold óptimo: {opt_thresh:.4f}")
print(f"  • TPR (Recall) @ óptimo: {opt_tpr:.3f}")
print(f"  • FPR @ óptimo: {opt_fpr:.3f}")
print(f"  • Precisión @ mejor F1: {best_prec:.3f}")
print(f"  • Recall @ mejor F1: {best_rec:.3f}")

print("\n📁 Resultados guardados en:", OUTPUT_DIR)
