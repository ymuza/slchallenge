"""


1 - Carga los datosreales (submission_real.csv) y de entrenamiento (z_train.npy).

2 - Calcula el rango de redshifts en ambos dominios.

3 - Normaliza los redshifts reales a [0,1].

4 - Reescala al rango del entrenamiento, generando una nueva columna predicted_z_aligned.

5 - Guarda el nuevo CSV como submission_real_aligned.csv.


"""



# align_redshift_domains.py
import numpy as np
import pandas as pd
import os

# --- Configuración de paths ---
Z_TRAIN_PATH = "outputs/z_train.npy"
SUBMISSION_PATH = "outputs/submission_real.csv"
OUTPUT_PATH = "outputs/submission_real_aligned.csv"

os.makedirs("outputs", exist_ok=True)

print("🔹 Cargando datos...")
z_train = np.load(Z_TRAIN_PATH)
submission = pd.read_csv(SUBMISSION_PATH)

print(f"✅ z_train cargado: {len(z_train)} valores")
print(f"✅ submission cargado: {len(submission)} filas")

# --- Estadísticas ---
z_train_min, z_train_max = z_train.min(), z_train.max()
z_real_min, z_real_max = submission["predicted_z"].min(), submission["predicted_z"].max()

print("\n📊 Rango de redshifts:")
print(f"  • Train → min={z_train_min:.3f}, max={z_train_max:.3f}")
print(f"  • Real  → min={z_real_min:.3f}, max={z_real_max:.3f}")

# --- Normalización y mapeo ---
z_real_scaled = (submission["predicted_z"] - z_real_min) / (z_real_max - z_real_min)
submission["predicted_z_aligned"] = z_real_scaled * (z_train_max - z_train_min) + z_train_min

# --- Estadísticas del nuevo conjunto ---
print("\n✅ Redshifts alineados:")
print(f"  • min={submission['predicted_z_aligned'].min():.3f}")
print(f"  • max={submission['predicted_z_aligned'].max():.3f}")
print(f"  • mean={submission['predicted_z_aligned'].mean():.3f}")
print(f"  • std={submission['predicted_z_aligned'].std():.3f}")

# --- Guardado ---
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Archivo guardado en: {OUTPUT_PATH}")
