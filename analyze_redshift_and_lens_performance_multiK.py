import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os

# --- Paths ---
TRAIN_PATH = "outputs/embeddings.npy"
LABELS_PATH = "outputs/y_train_lenses.npy"

os.makedirs("outputs", exist_ok=True)

# --- Load data ---
print("🔹 Cargando datos...")
X = np.load(TRAIN_PATH)
y = np.load(LABELS_PATH)
min_len = min(len(X), len(y))
X, y = X[:min_len], y[:min_len]
print(f"✅ Embeddings: {X.shape}, Labels: {y.shape}")

# --- Normalize and PCA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if X_scaled.shape[1] > 256:
    print("⚙️ Aplicando PCA a 256 componentes...")
    pca = PCA(n_components=256)
    X_reduced = pca.fit_transform(X_scaled)
else:
    X_reduced = X_scaled

# --- Curvas ROC para varios K ---
k_values = [1, 3, 5, 7, 9, 15]
plt.figure(figsize=(8, 8))
best_auc, best_k, best_thr, best_point = 0, None, None, None

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_reduced, y)
    probs = knn.predict_proba(X_reduced)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"K={k} (AUC={roc_auc:.3f})")

    # Buscar mejor threshold local
    idx = np.argmax(tpr - fpr)
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_k = k
        best_thr = thresholds[idx]
        best_point = (fpr[idx], tpr[idx])

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.scatter(*best_point, color="red", s=80,
            label=f"Mejor threshold (K={best_k}, thr={best_thr:.3f})")
plt.title("Comparación de curvas ROC para distintos K (KNN Lens Classifier)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_curve_multiK.png", dpi=200)
plt.close()

print(f"✅ Mejor K={best_k}, AUC={best_auc:.3f}, threshold óptimo={best_thr:.3f}")
print("📈 Curva comparativa guardada en outputs/roc_curve_multiK.png")
