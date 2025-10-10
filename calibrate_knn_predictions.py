import pandas as pd
import numpy as np
import os

# --- Configuración ---
INPUT_PATH = "outputs/submission_final_knn.csv"
OUTPUT_PATH = "outputs/submission_final_knn_calibrated.csv"
TARGET_RATIO = 0.02  # 2% de lentes realistas

print("🔹 Cargando predicciones...")
df = pd.read_csv(INPUT_PATH)
print(f"✅ Archivo cargado: {len(df)} filas")

# Verificamos columna is_lens
if "is_lens" not in df.columns:
    raise ValueError("❌ No se encontró columna 'is_lens' en el CSV.")

# --- Calibración ---
lens_candidates = df[df["is_lens"] == 1]
nonlens_candidates = df[df["is_lens"] == 0]

n_target = int(len(df) * TARGET_RATIO)
print(f"🎯 Manteniendo solo el {TARGET_RATIO*100:.1f}% superior de lentes → {n_target} objetos")

# Si hay más de lo necesario, seleccionamos aleatoriamente para mantener diversidad
if len(lens_candidates) > n_target:
    selected_lenses = lens_candidates.sample(n=n_target, random_state=42)
else:
    selected_lenses = lens_candidates

# Reasignar todos como no-lentes y luego marcar los seleccionados como lentes
df["is_lens"] = 0
df.loc[selected_lenses.index, "is_lens"] = 1

# --- Estadísticas ---
counts = df["is_lens"].value_counts(normalize=True) * 100
print("\n📊 Distribución calibrada:")
print(counts)
print(f"→ {counts.get(1, 0):.2f}% de las imágenes marcadas como lentes.")

# --- Guardado ---
os.makedirs("outputs", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Archivo guardado en: {OUTPUT_PATH}")
