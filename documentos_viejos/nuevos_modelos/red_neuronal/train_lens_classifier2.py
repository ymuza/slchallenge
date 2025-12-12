import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ======================
# CONFIG
# ======================
EMB_TRAIN = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_aligned.npy"
Y_TRAIN   = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/y_train_lenses_aligned.npy"
MODEL_OUT = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/weights/lens_classifier_mlp.pth"

LR = 1e-3
BATCH = 256
EPOCHS = 25
VAL_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# MLP DEFINITIVO
# ======================
class LensMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)


# ======================
# MAIN TRAINING
# ======================
def main():

    print("📥 Loading training data...")
    X = np.load(EMB_TRAIN)
    y = np.load(Y_TRAIN).astype(np.float32)

    # ======================
    # SCALING (MUY IMPORTANTE)
    # ======================
    print("📏 Scaling embeddings...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # save scaler
    np.save("mlp_scaler_mean.npy", scaler.mean_)
    np.save("mlp_scaler_scale.npy", scaler.scale_)

    # tensor conversion
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # ======================
    # TRAIN/VAL SPLIT
    # ======================
    n_total = len(X)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(TensorDataset(X, y), [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    # ======================
    # MODEL
    # ======================
    model = LensMLP().to(DEVICE)
    pos_weight = torch.tensor([94399 / 5601], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 Training MLP...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        # validation
        model.eval()
        val_probs = []
        val_true = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs.flatten())
                val_true.extend(yb.numpy().flatten())

        auc = roc_auc_score(val_true, val_probs)

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss/n_train:.4f} | Val AUC: {auc:.4f}")

    # ======================
    # SAVE MODEL
    # ======================
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"💾 Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
