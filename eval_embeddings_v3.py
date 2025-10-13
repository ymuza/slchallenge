import os
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from tqdm import tqdm
import warnings

# ==============================
# CONFIGURACIÓN
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
#BASE_PATH = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip_test_dataset/test_dataset"
#OUTPUT_PATH = "outputs/embeddings_real.npy"
BASE_PATH = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip_test_dataset/test_dataset"
OUTPUT_PATH = "outputs/embeddings_real.npy"
IDS_PATH = "outputs/ids_real.npy"

print(f"🧠 Usando dispositivo: {DEVICE}")

# ==============================
# CARGA DE MODELO DINOv2
# ==============================
print("🔹 Cargando modelo DINOv2 (vitl14) desde PyTorch Hub...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
model.eval().to(DEVICE)
print("✅ Modelo DINOv2 cargado.")

# ==============================
# LISTADO DE OBJETOS
# ==============================
fits_files = sorted([f for f in os.listdir(BASE_PATH) if f.endswith("_r.fits")])
print(f"🔎 Se encontraron {len(fits_files)} objetos (banda r).")

embeddings = []
ids = []

# ==============================
# PROCESAMIENTO POR LOTES
# ==============================
for i in tqdm(range(0, len(fits_files), BATCH_SIZE)):
    batch_imgs = []
    batch_ids = []

    for f in fits_files[i:i + BATCH_SIZE]:
        base_id = f.replace("_r.fits", "")
        try:
            # Leer las tres bandas
            with fits.open(os.path.join(BASE_PATH, f), memmap=False) as hdul_r:
                img_r = hdul_r[1].data["band_r"][0].astype(np.float32)
            with fits.open(os.path.join(BASE_PATH, f.replace("_r.fits", "_g.fits")), memmap=False) as hdul_g:
                img_g = hdul_g[1].data["band_g"][0].astype(np.float32)
            with fits.open(os.path.join(BASE_PATH, f.replace("_r.fits", "_i.fits")), memmap=False) as hdul_i:
                img_i = hdul_i[1].data["band_i"][0].astype(np.float32)

            # Combinar en RGB y normalizar
            rgb = np.stack([img_r, img_g, img_i], axis=-1)
            rgb = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
            rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
            rgb = rgb / (torch.max(rgb) + 1e-6)

            batch_imgs.append(rgb)
            batch_ids.append(base_id)

        except Exception as e:
            warnings.warn(f"[WARN] Error procesando {base_id}: {e}")

    if not batch_imgs:
        continue

    try:
        batch_tensor = torch.cat(batch_imgs, dim=0).to(DEVICE)
        with torch.no_grad():
            feats = model(batch_tensor)

        # El modelo ya devuelve embeddings planos (no se usa adaptive_avg_pool2d)
        emb = feats.detach().cpu().numpy()
        embeddings.append(emb)
        ids.extend(batch_ids)

        # Liberar memoria GPU
        del batch_tensor, feats
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print("⚠️ GPU sin memoria, pasando procesamiento a CPU para este lote.")
        torch.cuda.empty_cache()
        try:
            batch_tensor = torch.cat(batch_imgs, dim=0).to("cpu")
            with torch.no_grad():
                feats = model.to("cpu")(batch_tensor)
            emb = feats.detach().numpy()
            embeddings.append(emb)
            ids.extend(batch_ids)
        except Exception as e:
            warnings.warn(f"[WARN] Error procesando en CPU {base_id}: {e}")

# ==============================
# GUARDADO DE RESULTADOS
# ==============================
if len(embeddings) == 0:
    raise RuntimeError("❌ No se generaron embeddings válidos.")

embeddings = np.vstack(embeddings)
np.save(OUTPUT_PATH, embeddings)
np.save(IDS_PATH, np.array(ids))

print(f"✅ Guardado: {OUTPUT_PATH}  →  shape={embeddings.shape}")
print(f"✅ Guardado: {IDS_PATH}  →  total IDs={len(ids)}")
print("🎉 Finalizado con éxito.")
