import numpy as np
import torch

from train_lens_classifier2 import LensMLP


EMB_TEST = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"
IDS_TEST = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"
MODEL = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/weights/lens_classifier_mlp.pth"

OUT_CSV = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/outputs/mlp_predictions.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    print("📥 Loading test embeddings...")
    X = np.load(EMB_TEST)
    ids = np.load(IDS_TEST)

    # load scaler
    mean = np.load("mlp_scaler_mean.npy")
    scale = np.load("mlp_scaler_scale.npy")
    X = (X - mean) / scale

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # model
    model = LensMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model.eval()

    # predict
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    preds = (probs >= 0.9).astype(int)

    import pandas as pd
    df = pd.DataFrame({
        "id": ids,
        "preds": preds,
        "prob": probs
    })
    df.to_csv(OUT_CSV, index=False)

    print(f"💾 Predictions saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
