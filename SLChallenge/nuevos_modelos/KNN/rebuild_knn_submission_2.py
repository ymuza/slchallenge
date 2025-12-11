import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# === RUTAS (ajusta si hace falta) ===
EMB_TRAIN_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_1024.npy"
Y_TRAIN_PATH   = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/y_train_lenses_aligned.npy"

EMB_TEST_PATH  = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"
IDS_TEST_PATH  = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"   # array de strings tipo object_00000

OUTPUT_CSV     = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs/submission_knn_rebuilt.csv"

# === HIPERPARÁMETROS DEL KNN ORIGINAL ===
K_CLASS = 11
THRESH  = 0.46   # igual que en generador_de_embeddings_reales_y_CSV_para_challenge.py

def main():
    print("📥 Cargando embeddings de entrenamiento...")
    X_train = np.load(EMB_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)

    # Si hay mismatch de tamaño, truncar al mínimo (igual que hacía tu script original)
    if X_train.shape[0] != y_train.shape[0]:
        m = min(X_train.shape[0], y_train.shape[0])
        print(f"⚠️ Mismatch en N_train, truncando a {m}")
        X_train = X_train[:m]
        y_train = y_train[:m]

    print("📥 Cargando embeddings de test...")
    X_test = np.load(EMB_TEST_PATH)
    ids_test = np.load(IDS_TEST_PATH)
    print("  X_test:", X_test.shape)
    print("  ids_test:", ids_test.shape)

    # === ESCALADO IGUAL QUE ANTES ===
    print("📏 Estandarizando embeddings...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # === KNN CLASIFICADOR ===
    print(f"🧠 Entrenando KNN (k={K_CLASS}, weights='distance')...")
    clf = KNeighborsClassifier(
        n_neighbors=K_CLASS,
        weights="distance",
        n_jobs=-1
    )
    clf.fit(X_train_sc, y_train)

    print("🔮 Prediciendo probabilidades para test...")
    probs = clf.predict_proba(X_test_sc)[:, 1]         # probabilidad de ser lente
    preds = (probs >= THRESH).astype(int)              # igual que tu THRESH original

    print("📊 Resumen rápido:")
    n_total = len(preds)
    n_ones  = int((preds == 1).sum())
    n_zeros = n_total - n_ones
    print(f"  Total: {n_total}")
    print(f"  Lenses (1): {n_ones} ({100*n_ones/n_total:.2f}%)")
    print(f"  Non-lenses (0): {n_zeros} ({100*n_zeros/n_total:.2f}%)")

    # === CSV FINAL: id, preds, prob ===
    df = pd.DataFrame({
        "id": ids_test,
        "preds": preds,
        "prob": probs
    })

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Guardado CSV: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
