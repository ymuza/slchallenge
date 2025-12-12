import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ================================
# CONFIG
# ================================
CSV_OLD = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/submission_final_enviado.csv"
CSV_KNN = "outputs/submission_knn_rebuilt.csv"   # probas nuevas del KNN
OUT_PNG = "outputs/roc_knn.png"

# ================================
# LOAD DATA
# ================================
df_old = pd.read_csv(CSV_OLD).sort_values("id")
df_knn = pd.read_csv(CSV_KNN).sort_values("id")

y_true = df_old["preds"].values.astype(int)
y_score = df_knn["pred_prob"].values.astype(float)

# ================================
# ROC + AUC
# ================================
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

print("AUC KNN =", roc_auc)

# ================================
# PLOT
# ================================
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"KNN ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], 'k--')   # línea diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN Reconstruction")
plt.legend()
plt.grid(True)

plt.savefig(OUT_PNG, dpi=200)
plt.close()

print("✔ ROC guardado en:", OUT_PNG)
