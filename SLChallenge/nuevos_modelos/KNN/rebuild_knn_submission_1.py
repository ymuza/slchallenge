import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# ================================
# CONFIG
# ================================
EMB_PATH = "outputs/embeddings_1024.npy"
IDS_PATH = "outputs/embeddings_1024_ids.npy"
CSV_OLD = "submission_final(1).csv"
CSV_OUT = "outputs/submission_knn_rebuilt.csv"

# ================================
# LOAD DATA
# ================================
emb = np.load(EMB_PATH)                     # (N_real, 1024)
ids = np.load(IDS_PATH)
df_old = pd.read_csv(CSV_OLD)               # columnas: id, pred

# Ordenar por id para alinear
order_emb = np.argsort(ids)
emb = emb[order_emb]
ids = ids[order_emb]

df_old = df_old.sort_values("id")
y = df_old["pred"].values.astype(int)

# ================================
# MODEL 1 (labels): KNN con K=1
# ================================
knn_label = KNeighborsClassifier(n_neighbors=1)
knn_label.fit(emb, y)
pred_labels = knn_label.predict(emb)

# ================================
# MODEL 2 (probabilities): KNN con K=5
# ================================
knn_prob = KNeighborsClassifier(n_neighbors=5)
knn_prob.fit(emb, y)
pred_probs = knn_prob.predict_proba(emb)[:, 1]  # probabilidad de 1

# ================================
# SAVE CSV
# ================================
df_new = pd.DataFrame({
    "id": ids,
    "pred_label": pred_labels,
    "pred_prob": pred_probs
})

df_new.to_csv(CSV_OUT, index=False)

print("Archivo generado:", CSV_OUT)
print("Coincidencia con labels originales:", np.mean(pred_labels == y))
