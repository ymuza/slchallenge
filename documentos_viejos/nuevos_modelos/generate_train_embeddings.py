import os
import numpy as np
from astropy.io import fits
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import re
from torch.utils.data import Dataset, DataLoader

# ============================================
# CONFIG
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LENS_DIR = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_lenses/hsc_lenses"
NONLENS_DIR = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_nonlenses/hsc_nonlenses"

OUTPUT_EMB = "train_embeddings_dino_224.npy"
OUTPUT_LABELS = "train_labels.npy"
OUTPUT_IDS = "train_ids.npy"

# Tamaño óptimo para DINOv2
TARGET_SIZE = 224

# Normalización EXACTA de tus scripts originales
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.25, 0.25, 0.25]

# Orden correcto para HSC → RGB = r,g,i
BANDS = ["r", "g", "i"]

BATCH_SIZE = 48       # Para 8GB VRAM; bajar si hay OOM
NUM_WORKERS = 4       # Ajustar según CPU


# ============================================
# REGEX PARA DETECTAR IDS
# ============================================

pattern = re.compile(r"^(D2_[LN]_\d{8})_([griyz])\.fits$")

def get_unique_ids(directory):
    band_dict = defaultdict(set)

    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if not m:
            continue
        base_id = m.group(1)
        band = m.group(2)
        band_dict[base_id].add(band)

    valid = [bid for bid, bands in band_dict.items() if {"r","g","i"}.issubset(bands)]
    valid.sort()
    return valid


# ============================================
# LECTURA DE FITS
# ============================================

def load_rgb_fits(id_base, directory):
    imgs = []

    for band in BANDS:
        filename = f"{id_base}_{band}.fits"
        path = os.path.join(directory, filename)

        table = fits.getdata(path)

        # Caso FITS tabla (como tus archivos)
        if hasattr(table, "dtype") and table.dtype.names:
            img = table[f"band_{band}"][0].astype(np.float32)
        else:
            img = table.astype(np.float32)

        # Normalización simple como el pipeline original
        maxv = np.nanmax(img)
        if maxv > 0:
            img = img / maxv

        img = np.nan_to_num(img)
        imgs.append(img)

    # Devuelve (H, W, 3)
    return np.stack(imgs, axis=2)


# ============================================
# PREPROCESS EXACTO DEL PIPELINE ORIGINAL
# ============================================

preprocess = transforms.Compose([
    transforms.ToTensor(),  # (H,W,3) → (3,H,W)
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


# ============================================
# CARGAR MODELO DINOv2
# ============================================

def load_dino():
    print("📥 Loading DINOv2 ViT-L/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    model.eval()
    return model.to(DEVICE)


# ============================================
# DATASET + BATCHING
# ============================================

class HSCDataset(Dataset):
    def __init__(self, lens_ids, nonlens_ids):
        self.ids = lens_ids + nonlens_ids
        self.labels = [1]*len(lens_ids) + [0]*len(nonlens_ids)
        self.paths = [LENS_DIR if lab==1 else NONLENS_DIR for lab in self.labels]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_np = load_rgb_fits(self.ids[idx], self.paths[idx])
        img_tensor = preprocess(img_np)
        return img_tensor, self.labels[idx], self.ids[idx]


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    return imgs, labels, ids


# ============================================
# MAIN
# ============================================

def main():

    lens_ids = get_unique_ids(LENS_DIR)
    nonlens_ids = get_unique_ids(NONLENS_DIR)

    print(f"Lenses: {len(lens_ids)}")
    print(f"Non-Lenses: {len(nonlens_ids)}")
    print(f"Total: {len(lens_ids)+len(nonlens_ids)}")

    dataset = HSCDataset(lens_ids, nonlens_ids)

    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        collate_fn=collate_fn)

    model = load_dino()

    all_emb = []
    all_labels = []
    all_ids = []

    print("🚀 Extracting DINO embeddings with batch-size =", BATCH_SIZE)

    for imgs, labels, ids in tqdm(loader):

        imgs = imgs.to(DEVICE)

        with torch.no_grad():
            feats = model(imgs)     # (B,1024) or (B,14,14,1024)

            if feats.ndim == 4:
                feats = F.adaptive_avg_pool2d(feats, (1,1)).squeeze()

        feats = feats.cpu().numpy()
        all_emb.append(feats)
        all_labels.extend(labels)
        all_ids.extend(ids)

    all_emb = np.concatenate(all_emb, axis=0)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_ids = np.array(all_ids)

    print("Final embeddings shape:", all_emb.shape)

    np.save(OUTPUT_EMB, all_emb)
    np.save(OUTPUT_LABELS, all_labels)
    np.save(OUTPUT_IDS, all_ids)

    print(f"🎉 DONE! Saved:")
    print(" →", OUTPUT_EMB)
    print(" →", OUTPUT_LABELS)
    print(" →", OUTPUT_IDS)


if __name__ == "__main__":
    main()
