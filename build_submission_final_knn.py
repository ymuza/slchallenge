import pandas as pd
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

KNN_PATH = "outputs/knn_predictions.csv"
ALIGNED_PATH = "documentos_viejos/submission_real_aligned.csv"
OUTPUT_PATH = "documentos_viejos/submission_final_knn.csv"

print("🔹 Cargando predicciones del KNN y redshifts alineados...")

# --- Carga de archivos ---
df_knn = pd.read_csv(KNN_PATH)
df_z = pd.read_csv(ALIGNED_PATH)

print(f"✅ KNN cargado: {df_knn.shape[0]} filas")
print(f"✅ Redshifts cargados: {df_z.shape[0]} filas")

print("\n🧩 Columnas KNN:", df_knn.columns.tolist())
print("🧩 Columnas redshift:", df_z.columns.tolist())

# --- Limpieza de columnas ---
# Renombrar la columna 'is_lens' en redshifts para evitar duplicados
if "is_lens" in df_z.columns:
    df_z = df_z.rename(columns={"is_lens": "is_lens_orig"})

# --- Merge ---
df_final = pd.merge(df_z, df_knn, on="id", how="inner")

# --- Verificación de columnas ---
print("\n🔍 Columnas resultantes:", df_final.columns.tolist())

# Si por alguna razón la columna de KNN no es 'is_lens', la localizamos
lens_col_candidates = [c for c in df_final.columns if "is_lens" in c.lower()]
if len(lens_col_candidates) == 0:
    raise ValueError("❌ No se encontró ninguna columna con 'is_lens' tras el merge.")
else:
    print(f"🧠 Columna de clasificación usada: {lens_col_candidates[-1]}")
    df_final["is_lens"] = df_final[lens_col_candidates[-1]]

# --- Aseguramos formato correcto ---
df_final["is_lens"] = df_final["is_lens"].astype(int)
if "predicted_z_aligned" in df_final.columns:
    df_final["predicted_z"] = df_final["predicted_z_aligned"]

# --- Columnas finales limpias ---
df_final = df_final[["id", "is_lens", "predicted_z"]]

# --- Estadísticas ---
print(f"\n✅ Merge completado: {df_final.shape[0]} filas")
counts = df_final["is_lens"].value_counts(normalize=True) * 100
print("\n📈 Distribución de clases:")
for cls, pct in counts.items():
    label = "Lente" if cls == 1 else "No Lente"
    print(f"  {label}: {pct:.2f}%")

print("\n📊 Estadísticas de redshift:")
print(df_final["predicted_z"].describe())

# --- Guardar resultado ---
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Archivo final guardado en: {OUTPUT_PATH}")
print("✅ ¡Listo para evaluación o submission!")
