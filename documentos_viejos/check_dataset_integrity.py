# check_dataset_integrity.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

# --- Configuración ---
EMB_PATH = "../SLChallenge/outputs/embeddings.npy"
LABELS_PATH = "../SLChallenge/outputs/y_train_lenses.npy"

print("🔹 Cargando datos...")
X = np.load(EMB_PATH)
y = np.load(LABELS_PATH)

print(f"✅ Embeddings: {X.shape}, Labels: {y.shape}")

# --- 1️⃣ Chequeo de longitudes ---
if len(X) != len(y):
    print(f"⚠️ Mismatch: len(X)={len(X)}, len(y)={len(y)}")
    print("   -> Truncando al mínimo común.")
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]
else:
    print("✅ Longitudes coinciden exactamente.")

# --- 2️⃣ Distribución de clases ---
counts = Counter(y)
total = sum(counts.values())
print("\n📊 Distribución de clases:")
for k, v in counts.items():
    print(f"  Clase {k}: {v} ({v/total*100:.2f}%)")

# --- 3️⃣ PCA para ver separación visual ---
print("\n🧠 Aplicando PCA a 2D para visualización...")
pca = PCA(n_components=2, random_state=42)
X2D = pca.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(X2D[:,0], X2D[:,1], c=y, cmap="coolwarm", s=2, alpha=0.6)
plt.title("Distribución PCA (2D) de embeddings coloreada por clase (is_lens)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="is_lens")
plt.tight_layout()
plt.savefig("outputs/pca_lens_distribution.png", dpi=200)
print("📈 Gráfico guardado: outputs/pca_lens_distribution.png")

# --- 4️⃣ Variancia explicada ---
explained = np.cumsum(pca.explained_variance_ratio_)
print(f"\n📈 Variancia explicada por las 2 primeras componentes: {explained[1]*100:.2f}%")

# --- 5️⃣ Correlación simple ---
mean_lens = X[y == 1].mean(axis=0)
mean_nonlens = X[y == 0].mean(axis=0)
diff = np.linalg.norm(mean_lens - mean_nonlens)
print(f"\n📏 Distancia media entre centroides de clases: {diff:.4f}")

if diff < 0.05:
    print("⚠️ Posible falta de señal discriminativa entre embeddings de lentes y no-lentes.")
else:
    print("✅ Hay señal detectable entre clases (los embeddings parecen distinguir lentes).")
