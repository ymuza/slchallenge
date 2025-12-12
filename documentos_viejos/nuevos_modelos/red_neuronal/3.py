# inference_lensnet.py
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMB_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"        # << ajustá
IDS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"
WEIGHTS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/weights/best_lens_classifier.pth"         # << ajustá
CSV_OUT = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs/submission_knn_rebuilt3.csv"      # << ajustá

class ReconstructedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

def main():
    # Cargar embeddings e ids
    emb = np.load(EMB_PATH).astype(np.float32)
    ids = np.load(IDS_PATH)

    X_tensor = torch.tensor(emb, device=DEVICE)

    # Cargar modelo
    model = ReconstructedModel().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # Inferencia
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape: (N, 2)

    # Tomamos la probabilidad de la clase positiva (índice 1)
    prob_positive = probs[:, 1]
    # Por si querés redondear:
    prob_rounded = np.round(prob_positive, 3)

    # Etiqueta final con umbral 0.5
    labels = (prob_positive >= 0.5).astype(int)

    # Generar CSV
    df = pd.DataFrame({
        "id": ids,
        "preds": labels,
        "prob": prob_rounded
    })
    df.to_csv(CSV_OUT, index=False)
    print("✔ CSV generado:", CSV_OUT)

if __name__ == "__main__":
    main()
