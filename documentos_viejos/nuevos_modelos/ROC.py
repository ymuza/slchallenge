import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ============================
# RUTAS - AJUSTA SI ES NECESARIO
# ============================

PROBS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs/submission_knn_rebuilt2.csv"
CSV_ORIGINAL = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs/submission_final_enviado.csv"

OUTPUT_FIG = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs/roc_curve.png"


def main():

    print("📥 Cargando nuevo CSV con probabilidades...")
    df_new = pd.read_csv(PROBS_PATH)  # contiene: id, preds, prob

    print("📥 Cargando CSV original con labels...")
    df_orig = pd.read_csv(CSV_ORIGINAL).set_index("id")

    # Alineamos por ID
    df_orig = df_orig.loc[df_new["id"]]

    y_true = df_orig["preds"].values.astype(int)
    y_scores = df_new["prob"].values.astype(float)

    print("✔️ Datos alineados:")
    print("  y_true:", y_true.shape)
    print("  y_scores:", y_scores.shape)

    # ============================
    # CURVA ROC
    # ============================

    print("\n📊 Generando curva ROC...")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"🎯 AUC: {roc_auc:.5f}")

    # ============================
    # Plot
    # ============================

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC - KNN Lenses Classifier")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200)
    plt.close()

    print(f"📁 Curva ROC guardada en: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
