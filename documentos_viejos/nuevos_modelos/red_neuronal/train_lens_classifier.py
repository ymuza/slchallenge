import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_PATH = "//media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_embeddings_dino_224.npy"
LABELS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_labels_dino.npy"

MODEL_OUTPUT = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/red_neuronal/weights/lens_classifier_dino.pth"

BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 6  # early stopping


# ============================
# DATASET
# ============================

class EmbeddingsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# MLP OPTIMIZADO
# ============================

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
# TRAINING LOOP
# ============================

def train():
    print("📥 Loading data...")
    X = np.load(EMBEDDINGS_PATH)
    y = np.load(LABELS_PATH)

    print("Train size:", X.shape, "Labels:", y.shape)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y
    )

    train_ds = EmbeddingsDataset(X_train, y_train)
    val_ds = EmbeddingsDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LensClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_auc = 0
    patience_counter = 0

    print("🚀 Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                logits = model(Xb)
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                val_preds.extend(prob)
                val_labels.extend(yb.numpy())

        auc = roc_auc_score(val_labels, val_preds)

        scheduler.step(auc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")

        # EARLY STOPPING
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_OUTPUT)
            print(f"💾 Saved new best model! AUC={auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⛔ Early stopping triggered.")
                break

    print(f"🎉 Training complete. Best AUC = {best_auc:.4f}")


if __name__ == "__main__":
    train()
