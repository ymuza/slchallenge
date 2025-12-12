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

# ============================
# CONFIG
# ============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = "/ruta/a/tu/carpeta_de_test"  # <-- cambialo por tu ruta real
OUTPUT_EMB = "test_embeddings_dino_224.npy"
OUTPUT_IDS = "test_ids.npy"

TARGET_SIZE = 224
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.25, 0.25, 0.25]

BANDS = ["r","g","i"]
BATCH_SIZE = 48
NUM_WORKERS = 4

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


def load_rgb_fits(id_base, directory):
    imgs = []
    for band in BANDS:
        filename = f"{id_base}_{band}.fits"
        path = os.path.join(directory, filename)
        table = fits.getdata(path)
        if hasattr(table, "dtype") and table.dtype.names:
            img = table[f"band_{band}"][0].astype(np.float32)
        else:
            img = table.astype(np.float32)
        maxv = np.nanmax(img)
        if maxv > 0:
            img = img / maxv
        img = np.nan_to_num(img)
        imgs.append(img)
    return np.stack(imgs, axis=2)  # (H,W,3)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


class TestDataset(Dataset):
    def __init__(self, ids, directory):
        self.ids = ids
        self.directory = directory

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        obj_id = self.ids[idx]
        img_np = load_rgb_fits(obj_id, self.directory)
        img_t = preprocess(img_np)
        return img_t, obj_id


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    ids  = [b[1] for b in batch]
    return imgs, ids


def load_dino():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    model.eval()
    return model.to(DEVICE)


def main():
    print("🔍 Detectando IDs en test folder...")
    test_ids = get_unique_ids(TEST_DIR)
    print(f"Found {len(test_ids)} objects in test set")

    ds = TestDataset(test_ids, TEST_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = load_dino()

    all_emb = []
    all_ids = []

    for imgs, ids in tqdm(loader, desc="Extracting test embeddings"):
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            feats = model(imgs)
            if feats.ndim == 4:
                feats = F.adaptive_avg_pool2d(feats, (1,1)).squeeze()
        embs = feats.cpu().numpy()
        all_emb.append(embs)
        all_ids.extend(ids)

    all_emb = np.concatenate(all_emb, axis=0)
    all_ids = np.array(all_ids, dtype=str)

    print("Embeddings shape:", all_emb.shape)
    print("IDs shape:", all_ids.shape)

    np.save(OUTPUT_EMB, all_emb)
    np.save(OUTPUT_IDS, all_ids)

    print("✅ Saved test embeddings:", OUTPUT_EMB)
    print("✅ Saved test ids:", OUTPUT_IDS)


if __name__ == "__main__":
    main()
