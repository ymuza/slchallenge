import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMB_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"
IDS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"
WEIGHTS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/weights/best_lens_classifier.pth"
CSV_OUT = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/outputs/submission_nn_inference.csv"

class SmallLensNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ===== Cargar embeddings e ids =====
emb = np.load(EMB_PATH).astype(np.float32)
ids = np.load(IDS_PATH)

# Convertir a tensor
X_tensor = torch.tensor(emb, device=DEVICE)

# ===== Cargar modelo y pesos =====
model = SmallLensNet().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

# ===== Hacer predicción =====
with torch.no_grad():
    pred_probs = model(X_tensor).cpu().numpy().flatten()

# Redondear probabilidades (opcional)
pred_probs_rounded = np.round(pred_probs, 1)

# Convertir a etiquetas con threshold 0.5
pred_labels = (pred_probs >= 0.5).astype(int)

# ===== Guardar salida =====
df_new = pd.DataFrame({
    "id": ids,
    "preds": pred_labels,
    "prob": pred_probs_rounded
})
df_new.to_csv(CSV_OUT, index=False)

print("✔ CSV generado:", CSV_OUT)
