#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
domain_shift_analysis_v2.py — Comparación entre dominio de entrenamiento y test real
Evalúa si hay desalineación (domain shift) entre las distribuciones de redshift (photo-z)
en el set de entrenamiento y las predicciones del set real.

Genera:
- Histograma comparativo entre z_true (train) y predicted_z (real)
- Distribución de z según la clase is_lens
- Métricas estadísticas de diferencia de distribución (media, std, KS test)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# -------------------- Paths --------------------
Z_TRAIN_PATH = "outputs/z_true.npy"
SUBMISSION_PATH = "documentos_viejos/submission_real.csv"

print("🔹 Cargando datos...")

# -------------------- Load data --------------------
z_train = np.load(Z_TRAIN_PATH)
submission = pd.read_csv(SUBMISSION_PATH)

z_real = submission["predicted_z"].values
is_lens = submission["is_lens"].values

print(f"✅ Entrenamiento: {len(z_train)} muestras, Real: {len(z_real)} muestras")
print(f"   is_lens únicos en test: {np.unique(is_lens, return_counts=True)}")

# -------------------- Métricas globales --------------------
mean_train, std_train = np.mean(z_train), np.std(z_train)
mean_real, std_real = np.mean(z_real), np.std(z_real)
ks_stat, ks_pval = ks_2samp(z_train, z_real)

print("\n📊 Estadísticas globales:")
print(f"• Train  → media={mean_train:.3f}, std={std_train:.3f}")
print(f"• Real   → media={mean_real:.3f}, std={std_real:.3f}")
print(f"• KS test statistic={ks_stat:.3f}, p-value={ks_pval:.2e}")

# -------------------- Plot 1: Distribución comparativa --------------------
plt.figure(figsize=(10,6))
sns.kdeplot(z_train, fill=True, color="royalblue", alpha=0.4, label="Entrenamiento (z_true)")
sns.kdeplot(z_real, fill=True, color="orange", alpha=0.4, label="Conjunto real (predicted_z)")
plt.title("Comparación de distribución de redshift — Entrenamiento vs Real", fontsize=14)
plt.xlabel("Redshift (z)")
plt.ylabel("Densidad")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/redshift_domain_comparison.png", dpi=150)
plt.close()

# -------------------- Plot 2: Distribución por clase --------------------
plt.figure(figsize=(10,6))
for val in np.unique(is_lens):
    sns.kdeplot(
        z_real[is_lens == val],
        fill=True,
        alpha=0.4,
        label=f"is_lens = {val}"
    )

plt.title("Distribución de redshift por clase (is_lens)", fontsize=14)
plt.xlabel("Redshift (z)")
plt.ylabel("Densidad")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/redshift_by_class_real.png", dpi=150)
plt.close()

print("✅ Gráficos guardados en outputs/:")
print("   • redshift_domain_comparison.png")
print("   • redshift_by_class_real.png")
