import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/documentos_viejos/outputs/best_lens_classifier.pth"
TEST_EMBEDDINGS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"  # Embeddings de TEST
TEST_IDS_PATH = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"  # IDs de TEST
OUTPUT_CSV = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/documentos_viejos/outputs/predicciones.csv"
BATCH_SIZE = 512

print(f"🔧 Usando dispositivo: {DEVICE}")



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



print(" Cargando modelo entrenado...")
model = LensClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Modo evaluación
print(" Modelo cargado exitosamente")


print("\n Cargando embeddings de test...")
test_embeddings = np.load(TEST_EMBEDDINGS_PATH)
test_ids = np.load(TEST_IDS_PATH, allow_pickle=True)

print(f" Datos cargados:")
print(f"   Test embeddings: {test_embeddings.shape}")
print(f"   Test IDs: {len(test_ids)}")


print("\n Generando predicciones...")

all_predictions = []
all_probabilities = []

# Procesar en batches
num_batches = (len(test_embeddings) + BATCH_SIZE - 1) // BATCH_SIZE

with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Procesando batches"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(test_embeddings))

        batch_embeddings = test_embeddings[start_idx:end_idx]
        batch_tensor = torch.FloatTensor(batch_embeddings).to(DEVICE)

        # Predicción
        outputs = model(batch_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        # Guardar resultados
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probabilidad de clase "Lente"


print("\n Creando archivo CSV...")

results_df = pd.DataFrame({
    'id': test_ids,
    'is_lens': all_predictions,
    'prob_lens': all_probabilities
})

# Guardar
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Predicciones guardadas en: {OUTPUT_CSV}")
print(f"\n📊 Estadísticas de predicciones:")
print(f"   Total de objetos: {len(results_df)}")
print(f"   Predichos como Lente (1): {sum(all_predictions)} ({100 * np.mean(all_predictions):.2f}%)")
print(
    f"   Predichos como No-Lente (0): {len(all_predictions) - sum(all_predictions)} ({100 * (1 - np.mean(all_predictions)):.2f}%)")
print(f"\n Distribución de probabilidades:")
print(f"   Media: {np.mean(all_probabilities):.4f}")
print(f"   Mediana: {np.median(all_probabilities):.4f}")
print(f"   Min: {np.min(all_probabilities):.4f}")
print(f"   Max: {np.max(all_probabilities):.4f}")

# Mostrar primeras predicciones
print(f"\n Primeras 10 predicciones:")
print(results_df.head(10))