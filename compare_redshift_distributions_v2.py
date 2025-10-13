# compare_redshift_distributions_v2.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import os

# --- Configuración ---
os.makedirs("outputs", exist_ok=True)
Z_TRAIN_PATH = "outputs/z_train.npy"
SUBMISSION_PATH = "documentos_viejos/submission_real.csv"

print("🔹 Cargando datos...")

# --- Cargar redshifts de entrenamiento ---
z_train = np.load(Z_TRAIN_PATH)
print(f"✅ z_train cargado: {len(z_train)} valores")

# --- Cargar predicciones reales ---
submission = pd.read_csv(SUBMISSION_PATH)
print(f"✅ submission_real.csv cargado: {len(submission)} filas")

if "predicted_z" not in submission.columns:
    raise ValueError("❌ No se encontró columna 'predicted_z' en submission_real.csv")

z_real = submission["predicted_z"].values

# --- Estadísticas básicas ---
print("\n📊 Estadísticas básicas:")
print(f"  • z_train → min={z_train.min():.3f}, max={z_train.max():.3f}, mean={z_train.mean():.3f}, std={z_train.std():.3f}")
print(f"  • z_real  → min={z_real.min():.3f}, max={z_real.max():.3f}, mean={z_real.mean():.3f}, std={z_real.std():.3f}")

# --- Prueba KS (Kolmogorov–Smirnov) ---
ks_stat, ks_pvalue = ks_2samp(z_train, z_real)
print(f"\n🧪 Test KS: estadístico={ks_stat:.4f}, p-valor={ks_pvalue:.4e}")
if ks_pvalue < 0.05:
    print("⚠️ Diferencia estadísticamente significativa entre distribuciones.")
else:
    print("✅ No se detectan diferencias significativas (distribuciones similares).")

# --- Gráfico comparativo ---
plt.figure(figsize=(10, 6))
sns.kdeplot(z_train, label="Entrenamiento (simulado)", fill=True, color="steelblue", alpha=0.4)
sns.kdeplot(z_real, label="Conjunto real (predicho)", fill=True, color="orange", alpha=0.4)
plt.title("Comparación de distribución de redshifts")
plt.xlabel("Redshift (z)")
plt.ylabel("Densidad")
plt.legend()
plt.grid(True, alpha=0.3)

output_path = "documentos_viejos/redshift_distribution_comparison_v2.png"
plt.savefig(output_path, dpi=200)
print(f"\n📈 Gráfico guardado en: {output_path}")

# --- Métrica resumen ---
mean_diff = abs(z_train.mean() - z_real.mean())
print(f"\n📏 Diferencia de medias: {mean_diff:.4f}")
print("✅ Análisis completado correctamente.")
