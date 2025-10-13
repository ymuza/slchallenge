# rebuild_probabilities_knn.py
# ---------------------------------------------------------
# Reconstruye la columna "prob_lens" en el CSV calibrado final,
# usando el clasificador KNN entrenado sobre embeddings alineados.
# Corrige automáticamente diferencias dimensionales entre train/test.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- Configuración ---
TRAIN_PATH = "outputs/embeddings.npy"
TEST_PATH = "outputs/embeddings_test.npy"
LABELS_PATH = "outputs/y_train_lenses.npy"
SUBMISSION_PATH = "outputs/submission_final_knn_calibrated.csv"

print("🔹 Cargando datos...")

# --- Carga de datos ---
X_train = np.load(TRAIN_PATH)
X_test = np.load(TEST_PATH)
y_train = np.load(LABELS_PATH)

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}, Labels: {y_train.shape}")

# --- Ajuste de tamaños ---
n = min(len(X_train), len(y_train))
if len(X_train) != len(y_train):
    print(f"⚠️ Mismatch detectado ({len(X_train)} vs {len(y_train)}). Truncando a {n}.")
    X_train = X_train[:n]
    y_train = y_train[:n]

# --- Alineación de dimensiones ---
if X_train.shape[1] != X_test.shape[1]:
    high_dim = max(X_train.shape[1], X_test.shape[1])
    low_dim = min(X_train.shape[1], X_test.shape[1])
    n_components = low_dim
    print(f"⚙️ Aplicando PCA simétrico → reducción a {n_components} componentes comunes...")

    # Entrenar PCA en el conjunto de mayor dimensión
    if X_train.shape[1] == high_dim:
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(
            np.pad(X_test, ((0, 0), (0, high_dim - X_test.shape[1])), mode='constant')
        )
    else:
        pca = PCA(n_components=n_components)
        X_test_reduced = pca.fit_transform(X_test)
        X_train_reduced = pca.transform(
            np.pad(X_train, ((0, 0), (0, high_dim - X_train.shape[1])), mode='constant')
        )
else:
    print("✅ Dimensiones consistentes, no se aplica PCA.")
    X_train_reduced, X_test_reduced = X_train, X_test

# --- Normalización ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# --- Entrenamiento del KNN ---
k = 5
print(f"🧠 Entrenando KNN con k={k}...")
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# --- Predicción de probabilidades ---
probs = knn.predict_proba(X_test_scaled)[:, 1]
print("✅ Probabilidades generadas correctamente.")

# --- Actualización del CSV ---
if not os.path.exists(SUBMISSION_PATH):
    raise FileNotFoundError(f"❌ No se encontró {SUBMISSION_PATH}")

df = pd.read_csv(SUBMISSION_PATH)
df["prob_lens"] = probs

# --- Guardar actualizado ---
df.to_csv(SUBMISSION_PATH, index=False)
print(f"💾 CSV actualizado con 'prob_lens' → {SUBMISSION_PATH}")

# --- Verificación rápida ---
print("\n📊 Ejemplo de columnas finales:")
print(df.head())
