# analyze_submission_final.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración ---
SUBMISSION_PATH = "outputs/submission_final.csv"
os.makedirs("outputs", exist_ok=True)

print("🔹 Cargando submission_final.csv...")
df = pd.read_csv(SUBMISSION_PATH)

print(f"✅ Cargado: {df.shape[0]} filas")
print(df.head())

# --- Verificar columnas ---
expected_cols = {"id", "is_lens", "predicted_z"}
if not expected_cols.issubset(df.columns):
    raise ValueError(f"❌ Faltan columnas esperadas. Se esperaban: {expected_cols}")

# --- Conteo de clases ---
counts = df["is_lens"].value_counts()
lens_ratio = counts.get(1, 0) / df.shape[0] * 100

print("\n📈 Distribución de clases:")
print(counts)
print(f"→ {lens_ratio:.2f}% de las imágenes fueron clasificadas como lentes.")

# --- Estadísticas de redshift ---
print("\n📊 Estadísticas de redshift:")
print(df["predicted_z"].describe())

# --- Gráfico 1: Histograma general ---
plt.figure(figsize=(10,5))
sns.histplot(df["predicted_z"], bins=50, kde=True, color="skyblue")
plt.title("Distribución general de redshift (predicted_z)")
plt.xlabel("z")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("outputs/redshift_hist_all.png", dpi=200)
print("📊 Gráfico guardado: outputs/redshift_hist_all.png")

# --- Gráfico 2: Redshift por clase (si hay más de una clase) ---
if len(counts) > 1:
    plt.figure(figsize=(10,5))
    sns.kdeplot(data=df, x="predicted_z", hue="is_lens", fill=True, common_norm=False, alpha=0.4)
    plt.title("Distribución de redshift separada por tipo de objeto")
    plt.xlabel("z (predicho)")
    plt.tight_layout()
    plt.savefig("outputs/redshift_hist_by_class.png", dpi=200)
    print("📊 Gráfico guardado: outputs/redshift_hist_by_class.png")
else:
    print("⚠️ Solo hay una clase — se omite el gráfico por clase.")

# --- Gráfico 3: Proporción de lentes ---
plt.figure(figsize=(5,5))
if len(counts) == 1:
    label = "Lentes" if 1 in counts.index else "No-lentes"
    plt.pie(counts, labels=[f"{label} ({counts.iloc[0]})"], autopct="%1.2f%%", colors=["lightgray" if label == "No-lentes" else "gold"])
else:
    plt.pie(
        counts,
        labels=[f"Non-lens ({counts.get(0,0)})", f"Lens ({counts.get(1,0)})"],
        autopct="%1.2f%%",
        startangle=90,
        colors=["lightgray", "gold"],
    )
plt.title("Proporción de lentes vs no-lentes")
plt.savefig("outputs/lens_ratio_pie.png", dpi=200)
print("📊 Gráfico guardado: outputs/lens_ratio_pie.png")

print("\n✅ Análisis completado. Revisa la carpeta outputs/")
