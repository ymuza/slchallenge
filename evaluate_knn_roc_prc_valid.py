import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Paths
TRAIN_PATH = "outputs/embeddings.npy"
LABELS_PATH = "outputs/y_train_lenses.npy"
os.makedirs("outputs", exist_ok=True)

print("🔹 Cargando datos...")
X = np.load(TRAIN_PATH)
y = np.load(LABELS_PATH)

# Alinear longitudes por seguridad
m = min(len(X), len(y))
X, y = X[:m], y[:m]
print(f"✅ Datos: X={X.shape}, y={y.shape}")

# 1) Split estratificado 80/20
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Scaler + PCA (fit SOLO en train)
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)

n_comp = min(256, X_tr_s.shape[1])  # reducir algo pero sin perder demasiado
print(f"⚙️ Aplicando PCA → {n_comp} componentes")
pca = PCA(n_components=n_comp, random_state=42).fit(X_tr_s)
X_tr_p = pca.transform(X_tr_s)
X_va_p = pca.transform(X_va_s)

# 3) ROC y PRC para varios K
k_values = [1, 3, 5, 7, 9, 15]
plt.figure(figsize=(8, 8))
best_auc = -1
best_info = None

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr_p, y_tr)
    probs = clf.predict_proba(X_va_p)[:, 1]

    # ROC
    fpr, tpr, thr = roc_curve(y_va, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"K={k} (AUC={roc_auc:.3f})")

    # Mejor umbral (Youden J)
    j = tpr - fpr
    j_idx = np.argmax(j)
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_info = dict(k=k, thr=thr[j_idx], fpr=fpr[j_idx], tpr=tpr[j_idx])

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.scatter(best_info["fpr"], best_info["tpr"], c="red", s=80,
            label=f"Mejor thr (K={best_info['k']}, thr={best_info['thr']:.3f})")
plt.title("ROC en validación (sin fuga) – KNN lentes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_valid_multiK.png", dpi=200)
plt.close()
print(f"✅ ROC guardada: outputs/roc_valid_multiK.png (mejor K={best_info['k']}, thr={best_info['thr']:.3f}, AUC={best_auc:.3f})")

# 4) Precision–Recall
plt.figure(figsize=(8, 8))
best_ap = -1
best_k_pr = None
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k).fit(X_tr_p, y_tr)
    probs = clf.predict_proba(X_va_p)[:, 1]
    prec, rec, _ = precision_recall_curve(y_va, probs)
    ap = average_precision_score(y_va, probs)
    plt.plot(rec, prec, lw=2, label=f"K={k} (AP={ap:.3f})")
    if ap > best_ap:
        best_ap = ap
        best_k_pr = k

plt.title("Precision–Recall en validación – KNN lentes")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/prc_valid_multiK.png", dpi=200)
plt.close()
print(f"✅ PRC guardada: outputs/prc_valid_multiK.png (mejor K={best_k_pr}, AP={best_ap:.3f})")
