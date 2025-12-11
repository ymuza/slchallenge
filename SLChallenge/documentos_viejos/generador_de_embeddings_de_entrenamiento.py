import os
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm
import cv2

from astroclip.models.astroclip import AstroClipModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CAMBIA ESTA RUTA POR TU CHECKPOINT REAL
CKPT_PATH = "/home/yamil/doctorado/AstroCLIP/pre_trained_model/astroclip.ckpt"

# RUTAS DE TUS DATOS
FOLDER_LENS = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_lenses/hsc_lenses"
FOLDER_NONLENS = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_nonlenses/hsc_nonlenses"

OUTPUT_EMB = "outputs/embeddings_train_v2.npy"
OUTPUT_LAB = "outputs/labels_train_v2.npy"

# ===========================
# DINO / CLIP NORMALIZATION
# ===========================
DINO_MEAN = np.array([0.485, 0.456, 0.406])
DINO_STD  = np.array([0.229, 0.224, 0.225])


# ===========================
# FUNCIONES DE PREPROCESADO
# ===========================

def clip_percentiles(img, low=0.5, high=99.5):
    vmin, vmax = np.percentile(img, [low, high])
    return np.clip(img, vmin, vmax)


def preprocess_fits(g_path, r_path, i_path):
    # Leer tablas FITS (cada archivo tiene 1 fila de tabla con columna band_x)
    g = fits.getdata(g_path)["band_g"][0]
    r = fits.getdata(r_path)["band_r"][0]
    i = fits.getdata(i_path)["band_i"][0]

    # Clipping por percentiles (remueve outliers extremos)
    g = clip_percentiles(g)
    r = clip_percentiles(r)
    i = clip_percentiles(i)

    # Asinh stretch (realza estructuras débiles)
    g = np.arcsinh(g / (np.std(g) + 1e-6))
    r = np.arcsinh(r / (np.std(r) + 1e-6))
    i = np.arcsinh(i / (np.std(i) + 1e-6))

    # Normalización por banda
    g = (g - g.mean()) / (g.std() + 1e-6)
    r = (r - r.mean()) / (r.std() + 1e-6)
    i = (i - i.mean()) / (i.std() + 1e-6)

    # Resize a 64×64
    g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    r = cv2.resize(r, (64, 64), interpolation=cv2.INTER_AREA)
    i = cv2.resize(i, (64, 64), interpolation=cv2.INTER_AREA)

    # Pseudo-RGB: i → R, r → G, g → B
    rgb = np.stack([i, r, g], axis=0).astype(np.float32)

    # Normalización DINO (igual que backbone de AstroCLIP)
    for c in range(3):
        rgb[c] = (rgb[c] - DINO_MEAN[c]) / DINO_STD[c]

    return rgb


# ===========================
# EXTRAER IDs DE OBJETOS
# ===========================

def extract_ids(folder):
    ids = []
    for f in os.listdir(folder):
        if f.endswith("_g.fits"):
            # Ej: "D2_L_00000000_g.fits" -> "D2_L_00000000"
            obj = f.replace("_g.fits", "")
            ids.append(obj)
    return sorted(ids)


# ===========================
# GENERAR EMBEDDINGS
# ===========================

def generate_embeddings():
    print(f"📥 Cargando modelo AstroCLIP desde: {CKPT_PATH}")

    model = AstroClipModel.load_from_checkpoint(CKPT_PATH)
    model = model.to(DEVICE)
    model.eval()

    print("📌 Extrayendo IDs…")

    lens_ids = extract_ids(FOLDER_LENS)
    non_ids  = extract_ids(FOLDER_NONLENS)

    print(f"✔ Lentes: {len(lens_ids)}")
    print(f"✔ No-lentes: {len(non_ids)}")

    all_objs = [(oid, 1) for oid in lens_ids] + [(oid, 0) for oid in non_ids]

    embeddings = []
    labels = []

    for obj_id, label in tqdm(all_objs, desc="🔎 Generando embeddings"):
        if label == 1:
            base = os.path.join(FOLDER_LENS, obj_id)
        else:
            base = os.path.join(FOLDER_NONLENS, obj_id)

        g_path = base + "_g.fits"
        r_path = base + "_r.fits"
        i_path = base + "_i.fits"

        img = preprocess_fits(g_path, r_path, i_path)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Forward AstroCLIP
        with torch.no_grad():
            emb = model.image_encoder(img_tensor)
            emb = model.projector_image(emb)
            embeddings.append(emb.cpu().numpy())
            labels.append(label)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)

    # Guardar
    os.makedirs(os.path.dirname(OUTPUT_EMB), exist_ok=True)

    np.save(OUTPUT_EMB, embeddings)
    np.save(OUTPUT_LAB, labels)

    print("\n✅ Embeddings guardados en:", OUTPUT_EMB)
    print("📌 Labels guardados en:", OUTPUT_LAB)


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    generate_embeddings()
