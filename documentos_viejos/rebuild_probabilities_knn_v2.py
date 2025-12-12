# rebuild_probabilities_knn_v2.py
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# === Paths ===
TRAIN_PATH = "outputs/embeddings.npy"
TEST_PATH = "outputs/embeddings_test.npy"
LABELS_PATH = "outputs/y_train_lenses.npy"
SUBMISSION_PATH = "outputs/submission_final_knn_calibrated.csv"
OUTPUT_PATH = "outputs/submission_final_knn_calibrated_with_prob.csv"

# === Config ===
K = 5
N_COMPONENTS = 768  # igualamos dimensionalidad

print("🔹 Cargando datos...")
X_train = np.load(TRAIN_PATH)
X_test = np.load(TEST_PATH)
y_train = np.load(LABELS_PATH)

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}, Labels: {y_train.shape}")

# Alinear tamaños
min_len = min(len(X_train), len(y_train))
if len(X_train) != len(y_train):
    print(f"⚠️ Truncando a {min_len} por mismatch de tamaños")
    X_train = X_train[:min_len]
    y_train = y_train[:min_len]

# === Reducción de dimensionalidad coherente ===
print(f"⚙️ Aplicando PCA para alinear a {N_COMPONENTS} dimensiones comunes...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(np.pad(X_test, ((0, 0), (0, X_train.shape[1] - X_test.shape[1]))))

pca = PCA(n_components=N_COMPONENTS)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# === KNN probabilístico ===
print(f"🧠 Entrenando KNN (k={K}) y generando probabilidades...")
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train_pca, y_train)
probabilities = knn.predict_proba(X_test_pca)[:, 1]

# === Cargar CSV original ===
print("📄 Cargando CSV de submission base...")
submission = pd.read_csv(SUBMISSION_PATH)

# Alinear tamaños si fuera necesario
min_len = min(len(submission), len(probabilities))
submission = submission.iloc[:min_len].copy()
probabilities = probabilities[:min_len]

# Agregar columna
submission["prob_lens"] = probabilities
print(f"✅ Columna 'prob_lens' añadida (shape={probabilities.shape})")

# Guardar archivo final
os.makedirs("outputs", exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Guardado: {OUTPUT_PATH}")
print(submission.head())