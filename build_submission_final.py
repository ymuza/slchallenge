# build_submission_final.py
import pandas as pd
import os

# --- Configuración ---
REAL_PATH = "outputs/submission_real.csv"
ALIGNED_PATH = "outputs/submission_real_aligned.csv"
OUT_PATH = "outputs/submission_final.csv"

print("🔹 Cargando predicciones originales y redshifts alineados...")

# --- Cargar ambos archivos ---
real_df = pd.read_csv(REAL_PATH)
aligned_df = pd.read_csv(ALIGNED_PATH)

print(f"✅ submission_real.csv → {real_df.shape}")
print(f"✅ submission_real_aligned.csv → {aligned_df.shape}")

# --- Validar columnas ---
if "predicted_z" not in real_df.columns:
    raise ValueError("❌ Falta la columna 'predicted_z' en submission_real.csv.")
if "predicted_z_aligned" not in aligned_df.columns:
    # Acepta variantes si el archivo no tiene nombre exacto
    aligned_col = [c for c in aligned_df.columns if "aligned" in c or "pred" in c][-1]
    aligned_df = aligned_df.rename(columns={aligned_col: "predicted_z_aligned"})

# --- Combinar ---
merged = real_df.copy()
merged["predicted_z_aligned"] = aligned_df["predicted_z_aligned"]

# --- Asegurar estructura final ---
final_df = merged[["id", "is_lens", "predicted_z_aligned"]]
final_df = final_df.rename(columns={"predicted_z_aligned": "predicted_z"})

# --- Guardar ---
os.makedirs("outputs", exist_ok=True)
final_df.to_csv(OUT_PATH, index=False)
print(f"💾 Archivo final guardado en: {OUT_PATH}")

# --- Resumen ---
print("\n📊 Estadísticas del submission_final.csv:")
print(final_df.describe(include='all'))
print("\n✅ Submission listo para subir al leaderboard.")
