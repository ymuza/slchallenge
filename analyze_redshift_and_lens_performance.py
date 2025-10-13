import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr
import os

# --- Paths ---
TRAIN_PATH = "outputs/embeddings.npy"
LABELS_PATH = "outputs/y_train_lenses.npy"
ZTRAIN_PATH = "outputs/z_train.npy"
ZPRED_PATH = "documentos_viejos/submission_real_aligned.csv"

os.makedirs("outputs", exist_ok=True)

# --- Load data ---
print("🔹 Cargando datos...")
X = np.load(TRAIN_PATH)
y = np.load(LABELS_PATH)

# --- Truncate to common length ---
min_len = min(len(X), len(y))
X, y = X[:min_len], y[:min_len]
print(f"✅ Embeddings: {X.shape}, Labels: {y.shape}")

# --- Scale and reduce dimensions ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if X_scaled.shape[1] > 256:
    print("⚙️ Aplicando PCA a 256 componentes para reducir dimensionalidad...")
    pca = PCA(n_components=256)
    X_reduced = pca.fit_transform(X_scaled)
else:
    X_reduced = X_scaled

# --- Train KNN ---
K = 5
print(f"🧠 Entrenando KNN (K={K})...")
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_reduced, y)
probs = knn.predict_proba(X_reduced)[:, 1]

# --- ROC curve ---
fpr, tpr, thresholds = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)
best_idx = np.argmax(tpr - fpr)
best_thr = thresholds[best_idx]

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"K={K} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.scatter(fpr[best_idx], tpr[best_idx], color="red", s=80,
            label=f"Best thr={best_thr:.3f}")
plt.title("Curva ROC – Clasificación de lentes (KNN)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/roc_curve_lenses.png", dpi=200)
plt.close()
print("📈 Curva ROC guardada en outputs/roc_curve_lenses.png")

# --- Redshift correlation ---
import pandas as pd
df_pred = pd.read_csv(ZPRED_PATH)
if "predicted_z_aligned" in df_pred.columns:
    z_pred = df_pred["predicted_z_aligned"].values[:min_len]
else:
    z_pred = df_pred["predicted_z"].values[:min_len]

z_true = np.load(ZTRAIN_PATH)[:min_len]
corr, _ = pearsonr(z_true, z_pred)

plt.figure(figsize=(8, 6))
plt.scatter(z_true, z_pred, s=6, alpha=0.4, color="teal")
plt.plot([z_true.min(), z_true.max()],
         [z_true.min(), z_true.max()],
         "r--", label="y=x (perfecta)")
plt.xlabel("True Redshift (z)")
plt.ylabel("Predicted Redshift (ẑ)")
plt.title(f"Correlación Photo-z (Pearson r = {corr:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/redshift_correlation.png", dpi=200)
plt.close()

print("📈 Gráfico de correlación guardado en outputs/redshift_correlation.png")
print(f"✅ Correlación Pearson = {corr:.3f}")
