import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# ============================
# CONFIG
# ============================

TRAIN_PATH  = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_aligned.npy"
LABELS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/y_train_lenses_aligned.npy"

K = 5                         # número de folds
N_NEIGHBORS = 20            # KNN
THRESHOLDS = [0.20, 0.25, 0.30, 0.50, 0.60, 0.70, 0.80]


# ============================
# CARGA DATOS
# ============================

X = np.load(TRAIN_PATH)
y = np.load(LABELS_PATH)

print(f"Embeddings: {X.shape}")
print(f"Labels:     {y.shape}")


# ============================
# K-FOLD CROSS VALIDATION
# ============================

kf = KFold(n_splits=K, shuffle=True, random_state=42)

all_probs = []
all_labels = []

fold_idx = 1

for train_idx, val_idx in kf.split(X):

    print(f"\n===== FOLD {fold_idx}/{K} =====")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Modelo
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance")
    knn.fit(X_train, y_train)

    # Probabilidades del fold
    probs = knn.predict_proba(X_val)[:, 1]

    all_probs.append(probs)
    all_labels.append(y_val)

    fold_idx += 1


# ============================
# CONCATENAR PREDICCIONES
# ============================

all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

print(f"\nTotal predictions: {len(all_probs)}")


# ============================
# CURVA ROC
# ============================

fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

print(f"\nAUC = {roc_auc:.4f}")


# ============================
# GRAFICA ROC
# ============================

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

roc_png = "roc_curve_kfold.png"
plt.savefig(roc_png, dpi=300)
print(f"ROC curve plot saved → {roc_png}")
plt.show()


# ============================
# EVALUACIÓN DE THRESHOLDS FIJOS
# ============================

print("\n===== EVALUACIÓN DE THRESHOLDS =====")
results = []

for th in THRESHOLDS:
    preds = (all_probs >= th).astype(int)

    tp = np.sum((preds == 1) & (all_labels == 1))
    fp = np.sum((preds == 1) & (all_labels == 0))
    tn = np.sum((preds == 0) & (all_labels == 0))
    fn = np.sum((preds == 0) & (all_labels == 1))

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    results.append([th, tp, fp, tn, fn, precision, recall, f1])

df = pd.DataFrame(results, columns=[
    "threshold", "TP", "FP", "TN", "FN", "precision", "recall", "f1"
])

print(df.to_string(index=False))


# ============================
# GUARDAR ROC EN CSV
# ============================

roc_csv = "roc_curve_kfold.csv"
np.savetxt(roc_csv, np.column_stack([fpr, tpr, roc_thresholds]),
           delimiter=",", header="fpr,tpr,threshold", comments="")

print(f"\nROC curve data saved → {roc_csv}")
