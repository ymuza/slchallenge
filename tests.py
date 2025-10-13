# analyze_prob_vs_redshift.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Paths ---
os.makedirs("outputs/final_diagnostics", exist_ok=True)
CSV_PATH = "outputs/submission_final_knn_calibrated.csv"

# --- Load data ---
print("🔹 Cargando submission_final_knn_calibrated.csv...")
df = pd.read_csv(CSV_PATH)

if not {"predicted_z", "prob_lens"}.issubset(df.columns):
    raise ValueError("❌ El CSV debe contener las columnas 'predicted_z' y 'prob_lens'.")

print(f"✅ Datos cargados: {len(df)} filas")
print(df[["predicted_z", "prob_lens"]].describe())

# --- Scatter plot: prob_lens vs predicted_z ---
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df, x="predicted_z", y="prob_lens",
    fill=True, cmap="mako", thresh=0.05, levels=100
)
plt.title("Mapa de densidad: prob_lens vs predicted_z")
plt.xlabel("Redshift predicho (z_pred)")
plt.ylabel("Probabilidad de lente (prob_lens)")
plt.tight_layout()
plt.savefig("outputs/final_diagnostics/prob_vs_redshift_density.png", dpi=200)
print("📈 Gráfico guardado en outputs/final_diagnostics/prob_vs_redshift_density.png")

# --- También, versión binned promedio ---
bins = np.linspace(df["predicted_z"].min(), df["predicted_z"].max(), 50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
mean_prob = df.groupby(pd.cut(df["predicted_z"], bins))["prob_lens"].mean()

plt.figure(figsize=(10, 5))
plt.plot(bin_centers, mean_prob, "-o", color="darkred", alpha=0.8)
plt.xlabel("Redshift predicho (z_pred)")
plt.ylabel("Probabilidad media de lente")
plt.title("Probabilidad promedio de lente por rango de redshift")
plt.tight_layout()
plt.savefig("outputs/final_diagnostics/prob_vs_redshift_mean.png", dpi=200)
print("📈 Gráfico guardado en outputs/final_diagnostics/prob_vs_redshift_mean.png")

print("\n✅ Análisis completado correctamente.")
