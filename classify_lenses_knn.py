import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

TRAIN_EMB_PATH = "outputs/embeddings.npy"
TEST_EMB_PATH = "outputs/embeddings_test.npy"
TRAIN_LABELS_PATH = "outputs/z_train.npy"
OUTPUT_PATH = "outputs/knn_predictions.csv"

print("🔹 Cargando embeddings...")
X_train = np.load(TRAIN_EMB_PATH)
X_test = np.load(TEST_EMB_PATH)
z_train = np.load(TRAIN_LABELS_PATH)

# --- Alinear tamaño de entrenamiento ---
if len(X_train) != len(z_train):
    print(f"⚠️ Ajustando tamaño: X_train={len(X_train)} → {len(z_train)}")
    X_train = X_train[:len(z_train)]

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# --- Igualar dimensiones con PCA ---
if X_train.shape[1] != X_test.shape[1]:
    print(f"⚙️ Aplicando PCA: {X_train.shape[1]} → {X_test.shape[1]} dimensiones")
    pca = PCA(n_components=X_test.shape[1], random_state=42)
    X_train = pca.fit_transform(X_train)
    joblib.dump(pca, "outputs/pca_alignment.pkl")

# --- Estandarizar ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "outputs/scaler.pkl")

# --- Etiquetas binarias (ejemplo simple: lenses si z > 0.5) ---
y_train = (z_train > 0.5).astype(int)
print(f"📊 Proporción de lentes simuladas en entrenamiento: {y_train.mean()*100:.2f}%")

# --- Entrenar KNN ---
print("🧠 Entrenando clasificador KNN...")
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn.fit(X_train, y_train)

# --- Predecir ---
y_pred = knn.predict(X_test)

# --- Guardar submission ---
ids = [f"object_{i:05d}" for i in range(len(y_pred))]
submission = np.column_stack([ids, y_pred])
np.savetxt(OUTPUT_PATH, submission, fmt="%s", delimiter=",", header="id,is_lens", comments='')

print(f"\n✅ Submission guardado en: {OUTPUT_PATH}")
