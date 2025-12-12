import numpy as np
import torch
import torch.nn as nn
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/weights/lens_classifier_dino.pth"
TEST_EMB = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"
TEST_IDS = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"

OUTPUT_CSV = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/outputs/predictions_test.csv"


# ===== MODEL (same architecture as training) =====

class LensClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)


# ============================
# PREDICT
# ============================

def predict():
    print("📥 Loading test data...")
    X = np.load(TEST_EMB)
    ids = np.load(TEST_IDS)

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    print("📥 Loading model...")
    model = LensClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("🚀 Predicting...")
    all_probs = []
    all_preds = []

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1]  # prob de ser lente
        preds = (probs > 0.5).long()

        all_probs = probs.cpu().numpy()
        all_preds = preds.cpu().numpy()

    print("💾 Saving CSV...")

    df = pd.DataFrame({
        "id": ids,
        "preds": all_preds,
        "prob": np.round(all_probs, 6)
    })

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"🎉 Done! File saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    predict()
