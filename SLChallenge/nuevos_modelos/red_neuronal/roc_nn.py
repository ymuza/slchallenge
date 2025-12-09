import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ================================
# CONFIG
# ================================
CSV_OLD = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/submission_final_enviado.csv"            # labels originales
CSV_NN = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/outputs/submission_nn_rebuilt.csv"      # probas de la red neuronal
OUT_PNG = "outputs/roc_nn.png"

# ================================
# LOAD DATA (CSV bien formateado)
# ================================
df_old = pd.read_csv(CSV_OLD).sort_values("id")
df_nn  = pd.read_csv(CSV_NN).sort_values("id")

# Extraer labels y probas
y_true = df_old["preds"].values.astype(int)
y_score = df_nn["prob"].values.astype(float)

# ================================
# ROC + AUC
# ================================
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

print("AUC NN =", roc_auc)

# ================================
# PLOT
# ================================
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"NN ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network Reconstruction")
plt.legend()
plt.grid(True)

plt.savefig(OUT_PNG, dpi=200)
plt.close()

print("✔ ROC guardado en:", OUT_PNG)
