import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import os

# --- Configuración ---
TRAIN_PATH = "../outputs/embeddings.npy"
TEST_PATH = "../outputs/embeddings_test.npy"
LABELS_PATH = "../outputs/y_train_lenses.npy"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔹 Cargando datos...")
X_train = np.load(TRAIN_PATH)
X_test = np.load(TEST_PATH)
y_train = np.load(LABELS_PATH)

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}, Labels: {y_train.shape}")

# --- Alinear dimensiones ---
min_dim = min(X_train.shape[1], X_test.shape[1])
print(f"⚙️ Aplicando PCA a {min_dim} componentes comunes...")

# Ajustar PCA con el conjunto de menor dimensión
if X_train.shape[1] > X_test.shape[1]:
    pca = PCA(n_components=min_dim)
    X_train = pca.fit_transform(X_train[:, :min_dim])
    X_test = pca.transform(X_test)
else:
    pca = PCA(n_components=min_dim)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test[:, :min_dim])

# --- Sincronizar longitud de etiquetas ---
if len(y_train) != len(X_train):
    min_len = min(len(y_train), len(X_train))
    print(f"⚠️ Ajustando longitudes: usando primeros {min_len} ejemplos comunes")
    X_train = X_train[:min_len]
    y_train = y_train[:min_len]

print(f"✅ Nuevas dimensiones → Train: {X_train.shape}, Test: {X_test.shape}, Labels: {y_train.shape}")

# --- Normalización ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Evaluar varios K ---
K_VALUES = [1, 3, 5, 7, 9, 15]
roc_data = []

plt.figure(figsize=(8, 8))
plt.title("Comparación de curvas ROC para distintos K (KNN Lens Classifier)")

for k in K_VALUES:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_prob = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train[:len(y_prob)], y_prob[:len(y_train)])
    roc_auc = auc(fpr, tpr)
    roc_data.append((k, roc_auc))

    plt.plot(fpr, tpr, lw=2, label=f"K={k} (AUC={roc_auc:.3f})")

# --- Threshold óptimo para el mejor modelo ---
best_k, best_auc = max(roc_data, key=lambda x: x[1])
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_prob_best = knn_best.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_train[:len(y_prob_best)], y_prob_best[:len(y_train)])
roc_auc = auc(fpr, tpr)
youden_index = np.argmax(tpr - fpr)
best_thr = thresholds[youden_index]

plt.plot(fpr, tpr, color="blue", lw=2)
plt.scatter(fpr[youden_index], tpr[youden_index], color="red", s=80, label=f"Mejor threshold (K={best_k}, thr={best_thr:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "roc_curve_knn_multi_v3.png")
plt.savefig(output_path, dpi=200)
print(f"📈 Curva ROC guardada en: {output_path}")

# --- Guardar métricas ---
metrics_path = os.path.join(OUTPUT_DIR, "roc_metrics_knn.csv")
pd.DataFrame(roc_data, columns=["K", "AUC"]).to_csv(metrics_path, index=False)
print(f"💾 Métricas guardadas en: {metrics_path}")

print(f"\n🏆 Mejor modelo: K={best_k}, AUC={best_auc:.4f}, threshold óptimo={best_thr:.3f}")
