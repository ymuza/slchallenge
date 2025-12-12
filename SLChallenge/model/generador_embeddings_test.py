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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip.ckpt"

DIR_TEST = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/test_dataset/test_dataset_updated"  # <-- AJUSTAR

OUTPUT_EMB = "outputs/embeddings_test_rgb.npy"
OUTPUT_IDS = "outputs/ids_test_rgb.txt"

BATCH_SIZE = 64
NUM_WORKERS = 4

def pad_to_48(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    if H == 41 and W == 41:
        padded = np.zeros((48, 48), dtype=np.float32)
        top = (48 - H) // 2
        left = (48 - W) // 2
        padded[top:top+H, left:left+W] = img
        return padded
    elif H == 48 and W == 48:
        return img.astype(np.float32)
    else:
        raise ValueError(f"Imagen con shape inesperado: {img.shape}")

class TestFITS_RGB(Dataset):
    def __init__(self, dir_test):
        self.dir = dir_test
        self.samples = []

        for fname in sorted(os.listdir(self.dir)):
            if fname.endswith("_g.fits"):
                base = fname.replace("_g.fits", "")
                self.samples.append(base)

        print(f"Total imágenes TEST detectadas: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def load_fits_test(self, path):
        img = fits.getdata(path)

        if img is None:
            raise ValueError(f"El archivo {path} no contiene datos.")

        if img.ndim != 2:
            raise ValueError(f"El archivo {path} no es una imagen 2D válida. Shape: {img.shape}")

        img = img.astype(np.float32)

        return pad_to_48(img)

    def __getitem__(self, idx):
        base = self.samples[idx]

        # cargar g,r,i
        img_g = self.load_fits_test(os.path.join(self.dir, f"{base}_g.fits"))
        img_r = self.load_fits_test(os.path.join(self.dir, f"{base}_r.fits"))
        img_i = self.load_fits_test(os.path.join(self.dir, f"{base}_i.fits"))

        # pseudo-RGB
        img = np.stack([img_i, img_r, img_g], axis=0)

        return torch.tensor(img, dtype=torch.float32), base


def main():
    print(f"Usando dispositivo: {DEVICE}")
    model = AstroClipModel.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
    model.eval().to(DEVICE)

    dataset = TestFITS_RGB(DIR_TEST)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    embeddings = []
    ids = []

    print("\n🔄 Generando embeddings TEST...\n")

    progress = tqdm(loader, desc="Procesando batches (test)", unit="batch", ncols=100)

    for batch_imgs, batch_ids in progress:
        batch_imgs = batch_imgs.to(DEVICE)

        with torch.no_grad():
            emb = model.image_encoder(batch_imgs)

        embeddings.append(emb.cpu().numpy())
        ids.extend(batch_ids)

        progress.set_postfix({
            "Emb": emb.shape,
            "Batch": len(batch_imgs)
        })

    print("\n📦 Guardando resultados...")

    embeddings = np.concatenate(embeddings, axis=0)
    os.makedirs(os.path.dirname(OUTPUT_EMB), exist_ok=True)

    np.save(OUTPUT_EMB, embeddings)
    with open(OUTPUT_IDS, "w") as f:
        for oid in ids:
            f.write(oid + "\n")

    print("\n🎉 Embeddings TEST generados correctamente:")
    print(" →", OUTPUT_EMB)
    print(" →", OUTPUT_IDS)

if __name__ == "__main__":
    main()
