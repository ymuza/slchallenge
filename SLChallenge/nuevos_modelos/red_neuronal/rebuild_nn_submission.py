import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"
IDS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"
# CSV viejo solo para labels y nada mas
CSV_OLD = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/submission_final_enviado.csv"
#CSV_OLD = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/clases.csv"
CSV_OUT = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/outputs/submission_nn_rebuilt.csv"

EPOCHS = 10 # probado con 8 y 9 epocas, las coincidencias están en el rango de 97 - 98 %, con 10, en promedio arriba de 98
BATCH_SIZE = 512
LR = 1e-3

# ================================
# CARGA DE DATOS
# ================================
emb = np.load(EMB_PATH).astype(np.float32)
ids = np.load(IDS_PATH)
df_old = pd.read_csv(CSV_OLD).sort_values("id")

# Ordenar embeddings e IDs
order_emb = np.argsort(ids)
emb = emb[order_emb]
ids = ids[order_emb]

# Alinear CSV con ids
df_old = df_old.set_index("id").loc[ids].reset_index()
y = df_old["preds"].values.astype(np.float32)

X_tensor = torch.tensor(emb, device=DEVICE)
y_tensor = torch.tensor(y, device=DEVICE).unsqueeze(1)

ds = TensorDataset(X_tensor, y_tensor)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# ================================
# DEFINICION DEL MODELO
# ================================
class SmallLensNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(), # preguntar a Mariano si habria que usar RReeLU en lugar de ReLU
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.model(x)

model = SmallLensNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

# ================================
# TRAIN
# ================================
for epoch in range(EPOCHS):
    for xb, yb in dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

# ================================
# PREDICT
# ================================
with torch.no_grad():
    pred_probs = model(X_tensor).cpu().numpy().flatten()

# ================================
# ROUND PROBABILITIES TO 1 DECIMAL
# ================================
pred_probs_rounded = np.round(pred_probs, 1)

# Labels after rounding probability (threshold = 0.5)
pred_labels = (pred_probs >= 0.5).astype(int)

# ================================
# SAVE CSV
# ================================
df_new = pd.DataFrame({
    "id": ids,
    "preds": pred_labels,
    "prob": pred_probs_rounded
})

df_new.to_csv(CSV_OUT, index=False)

print("✔ CSV generado:", CSV_OUT)
print("✔ Coincidencia con labels originales:", np.mean(pred_labels == y))
