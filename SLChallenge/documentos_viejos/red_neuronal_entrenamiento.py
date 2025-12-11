import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os

# ============================
# CONFIGURACIÓN
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_embeddings_dino_224.npy"
IDS_PATH        = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_ids_dino.npy"
LABELS_PATH     = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_labels_dino.npy"

BATCH_SIZE     = 256
LEARNING_RATE  = 0.0005
MAX_EPOCHS     = 70
PATIENCE       = 15

print(f"🔧 Usando dispositivo: {DEVICE}")

# Crear directorio outputs si no existe
os.makedirs("outputs", exist_ok=True)


# ============================
# CARGA DE DATOS CORRECTA
# ============================
print("📦 Cargando embeddings, IDs y etiquetas...")

embeddings = np.load(EMBEDDINGS_PATH)
ids        = np.load(IDS_PATH)       # Ya es un .npy
labels     = np.load(LABELS_PATH)    # Ya es un .npy

print(f"👉 Embeddings: {embeddings.shape}")
print(f"👉 IDs: {ids.shape}")
print(f"👉 Labels: {labels.shape}")

assert len(embeddings) == len(labels) == len(ids), "❌ ERROR: Los tamaños no coinciden."


# ============================
# TRAIN / VAL / TEST SPLIT
# ============================
X_temp, X_test, y_temp, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"\n📊 Split:")
print(f"   Train: {len(X_train)}")
print(f"   Val:   {len(X_val)}")
print(f"   Test:  {len(X_test)}")


# ============================
# DATASET
# ============================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


train_dataset = EmbeddingDataset(X_train, y_train)
val_dataset   = EmbeddingDataset(X_val, y_val)
test_dataset  = EmbeddingDataset(X_test, y_test)


# ============================
# WEIGHTED SAMPLER
# ============================
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================
# MODELO
# ============================
class LensClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.RReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.RReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)


model = LensClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n🧠 Modelo inicializado.")
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")


# ============================
# TRAIN + EVALUATE
# ============================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(y_batch).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            correct += preds.eq(y_batch).sum().item()
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (
        total_loss / len(loader),
        correct / len(loader.dataset),
        all_preds,
        all_labels,
        all_probs
    )


# ============================
# TRAINING LOOP
# ============================
print("\n🚀 Iniciando entrenamiento...\n")

best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(MAX_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "outputs/best_lens_classifier.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\n⏹️ Early stopping activado.")
            break


# ============================
# TEST SET
# ============================
print("\n📊 Evaluando modelo en TEST...\n")

model.load_state_dict(torch.load("outputs/best_lens_classifier.pth"))

test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
    model, test_loader, criterion
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nReporte de clasificación:")
print(classification_report(test_labels, test_preds, target_names=["No-Lente", "Lente"]))


# ============================
# VISUALIZACIÓN + ROC
# ============================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Loss curves
axes[0, 0].plot(history["train_loss"], label="Train Loss")
axes[0, 0].plot(history["val_loss"], label="Val Loss")
axes[0, 0].set_title("Curvas de Pérdida")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Accuracy curves
axes[0, 1].plot(history["train_acc"], label="Train Acc")
axes[0, 1].plot(history["val_acc"], label="Val Acc")
axes[0, 1].set_title("Curvas de Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
axes[1, 0].set_title("Matriz de Confusión")

# ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)
axes[1, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
axes[1, 1].plot([0, 1], [0, 1], "k--")
axes[1, 1].set_title("Curva ROC")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/nn_classifier_results.png", dpi=300)
print("\n💾 Gráfico guardado en outputs/nn_classifier_results.png")

# Save results JSON
results = {
    "test_accuracy": test_acc,
    "test_loss": test_loss,
    "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
    "history": history,
}
with open("outputs/nn_classifier_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Resultados guardados en outputs/nn_classifier_results.json")
print(f"\n🎯 ROC AUC Score: {roc_auc:.4f}")
