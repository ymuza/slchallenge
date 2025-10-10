import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuración ---
os.makedirs("outputs", exist_ok=True)
SUB_PATH = "outputs/submission_final.csv"
Z_TRAIN_PATH = "outputs/z_train.npy"  # para referencia del dominio de entrenamiento

print("🔹 Cargando resultados del KNN...")
df = pd.read_csv(SUB_PATH)

# Verificar columnas
print(f"✅ Cargado: {len(df)} filas")
print(df.head())

# Convertir columnas si es necesario
if df.columns.tolist() == ["object_00000", "0"]:  # caso mal separado
    df.columns = ["id", "is_lens"]
if "is_lens" not in df.columns:
    raise ValueError("❌ No se encontró la columna 'is_lens' en submission_final.csv")

# --- Distribución de clases ---
counts = df["is_lens"].value_counts().sort_index()
lens_ratio = counts.get(1, 0) / len(df) * 100
print("\n📈 Distribución de clases:")
print(counts)
print(f"→ {lens_ratio:.2f}% de las imágenes fueron clasificadas como lentes.")

# --- Gráfico: proporción de clases ---
plt.figure(figsize=(5, 5))
plt.pie(counts, labels=[f"Non-lens ({counts.get(0,0)})", f"Lens ({counts.get(1,0)})"],
        autopct="%1.1f%%", colors=["skyblue", "salmon"], startangle=90)
plt.title("Distribución de clases (KNN)")
plt.tight_layout()
plt.savefig("outputs/knn_lens_ratio_pie.png", dpi=200)
plt.close()
print("📊 Gráfico guardado: outputs/knn_lens_ratio_pie.png")

# --- Comparar con z_train (si existe) ---
if os.path.exists(Z_TRAIN_PATH):
    z_train = np.load(Z_TRAIN_PATH)
    plt.figure(figsize=(8, 4))
    sns.histplot(z_train, bins=60, color="blue", label="z_train", kde=True, stat="density", alpha=0.4)
    plt.title("Distribución de redshifts del conjunto de entrenamiento")
    plt.xlabel("z")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/knn_ztrain_distribution.png", dpi=200)
    plt.close()
    print("📊 Gráfico guardado: outputs/knn_ztrain_distribution.png")

# --- Histograma por clase ---
if "predicted_z" in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[df["is_lens"] == 1]["predicted_z"], bins=40, color="red", label="Lenses", kde=True, stat="density", alpha=0.5)
    sns.histplot(df[df["is_lens"] == 0]["predicted_z"], bins=40, color="blue", label="Non-lenses", kde=True, stat="density", alpha=0.5)
    plt.title("Distribución de redshifts por clase (KNN)")
    plt.xlabel("predicted_z")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/knn_redshift_by_class.png", dpi=200)
    plt.close()
    print("📊 Gráfico guardado: outputs/knn_redshift_by_class.png")

print("\n✅ Análisis finalizado. Gráfico en 'outputs/'.")