import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_PATH = "SLChallenge/outputs/embeddings.npy"
IDS_PATH = "SLChallenge/outputs/ids.npy"
LABELS_PATH = "SLChallenge/clases.csv"
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 15  # Early stopping

print(f"🔧 Usando dispositivo: {DEVICE}")

# --- CARGAR DATOS ---
print("📦 Cargando embeddings y etiquetas...")
embeddings = np.load(EMBEDDINGS_PATH)
ids = np.load(IDS_PATH)
labels_df = pd.read_csv(LABELS_PATH)

# Crear diccionario id -> label
label_dict = dict(zip(labels_df['id'], labels_df['is_lens']))

# Filtrar solo los IDs que tenemos embeddings
labels = np.array([label_dict[obj_id] for obj_id in ids if obj_id in label_dict])
valid_mask = np.array([obj_id in label_dict for obj_id in ids])
embeddings = embeddings[valid_mask]
ids = ids[valid_mask]

print(f"✅ Datos cargados: {len(embeddings)} muestras")
print(f"   Clase 0 (no-lente): {np.sum(labels == 0)} ({100 * np.mean(labels == 0):.1f}%)")
print(f"   Clase 1 (lente): {np.sum(labels == 1)} ({100 * np.mean(labels == 1):.1f}%)")

# --- SPLIT TRAIN/VAL/TEST ---
X_temp, X_test, y_temp, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"\n📊 Split de datos:")
print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# --- DATASET ---
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# Crear datasets
train_dataset = EmbeddingDataset(X_train, y_train)
val_dataset = EmbeddingDataset(X_val, y_val)
test_dataset = EmbeddingDataset(X_test, y_test)

# Weighted sampler para balancear clases
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- MODELO ---
class LensClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)


model = LensClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n🧠 Modelo creado:")
print(f"   Arquitectura: 1024 → 512 (Tanh) → 256 (Tanh) → 2")
print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")


# --- ENTRENAMIENTO ---
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels, all_probs


print("\n🚀 Iniciando entrenamiento...\n")

best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(MAX_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch + 1:3d}/{MAX_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'SLChallenge/outputs/best_lens_classifier.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping en época {epoch + 1}")
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('SLChallenge/outputs/best_lens_classifier.pth'))

# --- EVALUACIÓN EN TEST ---
print("\n📊 Evaluación en conjunto de test...\n")
test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nReporte de clasificación:")
print(classification_report(test_labels, test_preds, target_names=['No-Lente', 'Lente']))

# --- VISUALIZACIONES ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Curvas de entrenamiento
ax = axes[0, 0]
ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Época', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Curvas de Pérdida', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(history['train_acc'], label='Train Acc', linewidth=2)
ax.plot(history['val_acc'], label='Val Acc', linewidth=2)
ax.set_xlabel('Época', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Curvas de Accuracy', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. Matriz de confusión
ax = axes[1, 0]
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No-Lente', 'Lente'],
            yticklabels=['No-Lente', 'Lente'])
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')

# 3. Curva ROC
ax = axes[1, 1]
fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/nn_classifier_results.png', dpi=300, bbox_inches='tight')
print("\n💾 Gráficos guardados en outputs/nn_classifier_results.png")

# Guardar resultados
results = {
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'roc_auc': roc_auc,
    'confusion_matrix': cm.tolist(),
    'history': history
}

import json

with open('SLChallenge/outputs/nn_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Resultados guardados en outputs/nn_classifier_results.json")
print(f"\n🎯 ROC AUC Score: {roc_auc:.4f}")