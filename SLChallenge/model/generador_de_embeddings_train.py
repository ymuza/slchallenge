import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from tqdm import tqdm
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from astroclip.models.astroclip import AstroClipModel
import cv2



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip.ckpt"

DIR_LENSES = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_lenses/hsc_lenses"
DIR_NONLENSES = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_nonlenses/hsc_nonlenses"


OUTPUT_EMB = "outputs/embeddings_train_v3.npy"
OUTPUT_LAB = "outputs/labels_train_v3.npy"

# === NORMALIZACIÓN DINO ===
DINO_MEAN = np.array([0.485, 0.456, 0.406])
DINO_STD  = np.array([0.229, 0.224, 0.225])

# Tamaño de imagen: múltiplo de 12, y que ya sabemos que te funcionaba con AstroCLIP
RES = 48


def clip_percentiles(img, low=0.5, high=99.5):
    vmin, vmax = np.percentile(img, [low, high])
    return np.clip(img, vmin, vmax)


def preprocess_fits(g_path, r_path, i_path):
    """
    Preprocesado:
      - leer columnas band_g / band_r / band_i
      - clipping percentil
      - asinh stretch
      - normalización por banda
      - resize a RES x RES
      - pseudo-RGB (i, r, g)
      - normalización DINO
    """
    g = fits.getdata(g_path)["band_g"][0]
    r = fits.getdata(r_path)["band_r"][0]
    i = fits.getdata(i_path)["band_i"][0]

    # Clipping
    g = clip_percentiles(g)
    r = clip_percentiles(r)
    i = clip_percentiles(i)

    # Asinh
    g = np.arcsinh(g / (np.std(g) + 1e-6))
    r = np.arcsinh(r / (np.std(r) + 1e-6))
    i = np.arcsinh(i / (np.std(i) + 1e-6))

    # Normalización local
    g = (g - g.mean()) / (g.std() + 1e-6)
    r = (r - r.mean()) / (r.std() + 1e-6)
    i = (i - i.mean()) / (i.std() + 1e-6)

    # Resize a RES x RES (48 x 48, múltiplo de 12)
    g = cv2.resize(g, (RES, RES), interpolation=cv2.INTER_AREA)
    r = cv2.resize(r, (RES, RES), interpolation=cv2.INTER_AREA)
    i = cv2.resize(i, (RES, RES), interpolation=cv2.INTER_AREA)

    # Pseudo-RGB: i → R, r → G, g → B
    rgb = np.stack([i, r, g], axis=0).astype(np.float32)

    # Normalización DINO
    for c in range(3):
        rgb[c] = (rgb[c] - DINO_MEAN[c]) / DINO_STD[c]

    return rgb


def extract_ids(folder):
    """
    Devuelve IDs tipo 'D2_L_00000000' a partir de archivos *_g.fits.
    """
    ids = []
    for f in os.listdir(folder):
        if f.endswith("_g.fits"):
            obj = f.replace("_g.fits", "")
            ids.append(obj)
    return sorted(ids)


def generate_embeddings():
    print(f"📥 Cargando modelo AstroCLIP desde: {CKPT_PATH}")
    model = AstroClipModel.load_from_checkpoint(CKPT_PATH)
    model = model.to(DEVICE)
    model.eval()

    print("📌 Extrayendo IDs...")
    lens_ids = extract_ids(DIR_LENSES)
    non_ids  = extract_ids(DIR_NONLENSES)

    print(f"✔ Lentes: {len(lens_ids)}")
    print(f"✔ No-lentes: {len(non_ids)}")

    all_objs = [(oid, 1) for oid in lens_ids] + [(oid, 0) for oid in non_ids]

    embeddings = []
    labels = []

    for obj_id, label in tqdm(all_objs, desc="🔎 Generando embeddings"):
        if label == 1:
            base = os.path.join(DIR_LENSES, obj_id)
        else:
            base = os.path.join(DIR_NONLENSES, obj_id)

        g_path = base + "_g.fits"
        r_path = base + "_r.fits"
        i_path = base + "_i.fits"

        img = preprocess_fits(g_path, r_path, i_path)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Usamos solo image_encoder, sin projector_image, para evitar rutas
            # internas que usan 'attentions[1]' y están dando problemas
            emb = model.image_encoder(img_tensor)
            embeddings.append(emb.cpu().numpy())
            labels.append(label)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels, dtype=np.int64)

    os.makedirs(os.path.dirname(OUTPUT_EMB), exist_ok=True)
    np.save(OUTPUT_EMB, embeddings)
    np.save(OUTPUT_LAB, labels)

    print("\n✅ Embeddings guardados en:", OUTPUT_EMB)
    print("📌 Labels guardados en:", OUTPUT_LAB)
    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)


if __name__ == "__main__":
    generate_embeddings()
