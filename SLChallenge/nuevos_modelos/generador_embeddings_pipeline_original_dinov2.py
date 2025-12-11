import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
import re


# ============================
# CONFIG
# ============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LENS_DIR = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_lenses/hsc_lenses"
NONLENS_DIR = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_nonlenses/hsc_nonlenses"

OUTPUT_EMB = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_embeddings_dino.npy"
OUTPUT_LABELS = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_labels_dino.npy"
OUTPUT_IDS = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_ids_dino.npy"

# Igual que en tu generador original:
TARGET_SIZE = 518

# mean / std EXACTOS del pipeline anterior
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.25, 0.25, 0.25]

# Sólo r,g,i → pero orden final debe ser RGB = r,g,i
BANDS = ["r", "g", "i"]


# ============================
# REGEX PARA DETECTAR IDS
# ============================

pattern = re.compile(r"^(D2_[LN]_\d{8})_([griyz])\.fits$")

def get_unique_ids(directory):
    bands = defaultdict(set)

    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if not m:
            continue
        base = m.group(1)
        band = m.group(2)
        bands[base].add(band)

    # sólo aceptamos los que tengan r,g,i
    valid = [b for b, bb in bands.items() if {"r","g","i"}.issubset(bb)]
    valid.sort()
    return valid



# ============================
# CARGA EXACTA DE BANDAS (como en tu script viejo)
# ============================

def load_rgb_fits(id_base: str, directory: str):
    imgs = []

    for band in BANDS:
        filename = f"{id_base}_{band}.fits"
        path = os.path.join(directory, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        table = fits.getdata(path)

        # Si es tabla FITS (caso original)
        if isinstance(table, np.ndarray) and table.dtype.names is not None:
            key = f"band_{band}"
            img = table[key][0].astype(np.float32)
        else:
            # Imágenes standard
            img = table.astype(np.float32)

        # Normalización básica como antes
        maxv = np.nanmax(img)
        if maxv > 0:
            img = img / maxv

        img = np.nan_to_num(img)

        imgs.append(img)

    # Stack final: (3,H,W)
    arr = np.stack(imgs, axis=2)  # (H,W,3) para torchvision
    return arr



# ============================
# TRANSFORMACIONES EXACTAS DEL PIPELINE ORIGINAL
# ============================

preprocess = transforms.Compose([
    transforms.ToTensor(),  # convierte (H,W,3) → (3,H,W), escala [0,1]
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])



# ============================
# DINOv2 CARGA EXACTA
# ============================

def load_dino():
    print("📥 Loading DINOv2 ViT-L/14 (exact model)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    model.eval()
    return model.to(DEVICE)



# ============================
# EXTRACCIÓN EMBEDDINGS (idéntica a la original)
# ============================

def extract_embedding(model, img_np):
    """
    img_np: (H, W, 3)
    """

    tensor = preprocess(img_np)  # → (3,518,518)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = model(tensor)

        # si devuelve mapa espacial → average pooling
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, (1,1)).squeeze()

        emb = feats.cpu().numpy().astype(np.float32)
        return emb



# ============================
# PROGRAMA PRINCIPAL
# ============================

def main():

    # CARGAR IDS
    lens_ids = get_unique_ids(LENS_DIR)
    nonlens_ids = get_unique_ids(NONLENS_DIR)

    print(f"Lenses: {len(lens_ids)}")
    print(f"Non-lenses: {len(nonlens_ids)}")
    print(f"Total: {len(lens_ids) + len(nonlens_ids)}")

    all_ids = lens_ids + nonlens_ids
    all_labels = [1]*len(lens_ids) + [0]*len(nonlens_ids)

    # CARGAR DINO
    dino = load_dino()

    embeddings = []

    for obj_id, label in tqdm(zip(all_ids, all_labels), total=len(all_ids), desc="Extracting DINO embeddings"):
        directory = LENS_DIR if label == 1 else NONLENS_DIR

        img_np = load_rgb_fits(obj_id, directory)  # (H,W,3)
        emb = extract_embedding(dino, img_np)       # (1024,)
        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_ids = np.array(all_ids)

    print("Final embeddings shape:", embeddings.shape)

    np.save(OUTPUT_EMB, embeddings)
    np.save(OUTPUT_LABELS, all_labels)
    np.save(OUTPUT_IDS, all_ids)

    print("🎉 DONE!")
    print(f"Saved embeddings → {OUTPUT_EMB}")
    print(f"Saved labels     → {OUTPUT_LABELS}")
    print(f"Saved ids        → {OUTPUT_IDS}")


if __name__ == "__main__":
    main()
