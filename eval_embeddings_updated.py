"""
eval_embeddings_updated.py

Este script genera **embeddings** de imágenes (lenses y non-lenses) usando el
modelo AstroCLIP cargado desde un checkpoint local.

🔹 Flujo:
1. Carga el modelo AstroCLIP desde el checkpoint .ckpt con Lightning.
2. Usa GalaxyBatchDataset para leer tripletas de imágenes FITS (r,g,i),
   normalizarlas y convertirlas a tensores.
3. Extrae embeddings con el encoder de imágenes del modelo.
4. Guarda:
   - outputs/embeddings_updated.npy → matriz [N, 1024]
   - outputs/ids_updated.txt → lista de IDs asociados

🔹 Conceptos ML:
- **Embeddings**: representaciones en espacio latente que capturan similitud semántica.
- **Transfer Learning / Zero-Shot**: usamos un modelo pre-entrenado (AstroCLIP)
  sin necesidad de reentrenarlo desde cero.
- **Batch Inference**: procesamos imágenes en lotes para aprovechar GPU/CPU y memoria.

"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from astroclip.models.astroclip import AstroClipModel
from tensor_dataset_indexmap import GalaxyBatchDataset

# ================================
# Configuración
# ================================
CHECKPOINT = "/home/yamil/doctorado/AstroCLIP/pre_trained_model/astroclip.ckpt"

LENSES_ROOT = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/hsc_lenses"
NONLENSES_ROOT = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_nonlenses/hsc_nonlenses"

LENSES_META = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/parameters_updated.fits"
NONLENSES_META = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_nonlenses/parameters.fits"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ================================
# 1. Cargar modelo desde checkpoint local
# ================================
print(f"🔹 Cargando modelo AstroCLIP desde checkpoint local: {CHECKPOINT}")
model = AstroClipModel.load_from_checkpoint(CHECKPOINT, map_location=DEVICE)
model.eval()
model.to(DEVICE)

# ================================
# 2. Dataset + DataLoader
# ================================
def extract_embeddings(root, meta, label):
    ds = GalaxyBatchDataset(root=root, meta_fits=meta, size=96)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_embs, all_ids, all_labels = [], [], []
    with torch.no_grad():
        for batch in dl:
            imgs = batch["image"].to(DEVICE)
            ids = batch["id"]
            z = batch["redshift"]
            # ⚡ generar embeddings
            emb = model(imgs, input_type="image")
            all_embs.append(emb.cpu().numpy())
            all_ids.extend(ids)
            all_labels.extend(z.numpy())

    return (
        np.concatenate(all_embs, axis=0),
        np.array(all_ids),
        np.array(all_labels),
    )

print("🔵 Procesando LENSES...")
emb_lens, ids_lens, z_lens = extract_embeddings(LENSES_ROOT, LENSES_META, label=1)

print("🟢 Procesando NON-LENSES...")
emb_non, ids_non, z_non = extract_embeddings(NONLENSES_ROOT, NONLENSES_META, label=0)

# ================================
# 3. Guardar embeddings + IDs
# ================================
all_embs = np.vstack([emb_lens, emb_non])
all_ids = np.concatenate([ids_lens, ids_non])
all_z = np.concatenate([z_lens, z_non])

np.save(os.path.join(OUT_DIR, "embeddings_updated.npy"), all_embs)
np.savetxt(os.path.join(OUT_DIR, "ids_updated.txt"), all_ids, fmt="%s")
np.save(os.path.join(OUT_DIR, "redshifts_updated.npy"), all_z)

print(f"✅ Embeddings guardados: {all_embs.shape}")
print(f"✅ IDs guardados: {all_ids.shape}")
print(f"✅ Redshifts guardados: {all_z.shape}")
