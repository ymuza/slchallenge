import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ============================
# RUTAS - AJUSTA SI ES NECESARIO
# ============================

EMB_TRAIN = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_aligned.npy"
Y_TRAIN   = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/y_train_lenses_aligned.npy"

EMB_TEST  = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"
IDS_TEST  = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"  # array de strings tipo object_00000

OUTPUT_CSV     = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs/submission_knn_rebuilt2.csv"

CSV_ORIGINAL = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs/submission_final_enviado.csv"



def main():

    # --------- Cargar train ----------
    print("📥 Cargando embeddings de entrenamiento...")
    X_train = np.load(EMB_TRAIN)
    y_train = np.load(Y_TRAIN)
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)

    if X_train.shape[0] != y_train.shape[0]:
        m = min(X_train.shape[0], y_train.shape[0])
        print(f"⚠️ Mismatch en N_train, truncando a {m}")
        X_train = X_train[:m]
        y_train = y_train[:m]

    # --------- Cargar test ----------
    print("📥 Cargando embeddings de test...")
    X_test = np.load(EMB_TEST)
    ids_test = np.load(IDS_TEST)
    print("  X_test:", X_test.shape)
    print("  ids_test:", ids_test.shape)

    # --------- Cargar CSV original ----------
    print("📥 Cargando CSV original...")
    df_orig = pd.read_csv(CSV_ORIGINAL)

    # Aseguramos mismo orden de IDs entre embeddings y CSV
    df_orig = df_orig.set_index("id").loc[ids_test]
    y_orig = df_orig["preds"].values.astype(int)
    print("  y_orig (del CSV):", y_orig.shape)

    # --------- Escalado ----------
    print("📏 Escalando embeddings...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --------- Entrenar KNN ----------
    K_CLASS = 20
    print(f"🧠 Entrenando KNN k={K_CLASS}, weights='distance'...")
    clf = KNeighborsClassifier(
        n_neighbors=K_CLASS,
        weights="distance",
        n_jobs=-1
    )
    clf.fit(X_train_sc, y_train)

    # --------- Probabilidades ----------
    print("🔮 Calculando probabilidades...")
    probs = clf.predict_proba(X_test_sc)[:, 1]   # prob de ser lente

    # --------- Buscar threshold que maximiza matches ----------
    print("🎯 Buscando threshold que maximice matches con el CSV original...")

    # Ordenamos por prob decreases
    idx_sorted = np.argsort(probs)[::-1]
    probs_sorted = probs[idx_sorted]
    y_orig_sorted = y_orig[idx_sorted]

    n = len(probs_sorted)

    # Para k = número de 1s asumidos (top-k como lentes)
    # Calculamos matches(k) = TP(k) + TN(k)

    is_one = (y_orig_sorted == 1).astype(int)
    cum_ones = np.cumsum(is_one)         # TP si top-k se predicen como 1
    total_ones = cum_ones[-1]

    # Para cada k, TP = cum_ones[k-1]
    # ones_suffix = total_ones - TP
    # zeros_suffix = (n - k) - ones_suffix
    # matches(k) = TP + zeros_suffix

    Ks = np.arange(1, n+1)
    TP = cum_ones  # TP[k-1]
    ones_suffix = total_ones - TP
    zeros_suffix = (n - Ks) - ones_suffix
    matches = TP + zeros_suffix

    best_idx = np.argmax(matches)
    #best_k = Ks[best_idx]
    best_k = 20
    best_matches = matches[best_idx]

    #best_threshold = probs_sorted[best_k - 1]
    best_threshold = 0.6

    print(f"✅ Mejor k: {best_k}")
    print(f"✅ Mejor threshold: {best_threshold:.6f}")
    print(f"✅ Matches (max): {best_matches} / {n} = {best_matches/n*100:.2f}%")

    # --------- Aplicar threshold global ----------
    preds = (probs >= best_threshold).astype(int)

    # Stats de distribución final
    total = len(preds)
    ones  = int(preds.sum())
    zeros = total - ones

    print("\n📊 Distribución final (nuevo CSV):")
    print(f"  Total: {total}")
    print(f"  Lenses (1): {ones} ({100*ones/total:.2f}%)")
    print(f"  Non-lenses (0): {zeros} ({100*zeros/total:.2f}%)")

    # --------- Guardar CSV ----------
    df_out = pd.DataFrame({
        "id": ids_test,
        "preds": preds,
        "prob": probs
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 CSV guardado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
