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

# ======================
# CONFIGURACIÓN
# ======================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip.ckpt"

DIR_LENSES = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_lenses/hsc_lenses"
DIR_NONLENSES = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/training_datasets/hsc_nonlenses/hsc_nonlenses"

OUTPUT_EMB = "outputs/embeddings_train_rgb.npy"
OUTPUT_LAB = "outputs/labels_train_rgb.npy"
OUTPUT_IDS = "outputs/ids_train_rgb.txt"
BATCH_SIZE = 64
NUM_WORKERS = 4

# ======================
# FUNCIONES AUXILIARES
# ======================

def pad_to_48(img: np.ndarray) -> np.ndarray:
    """Si la imagen es 41x41, la centra en 48x48 con padding cero."""
    H, W = img.shape
    if H == 41 and W == 41:
        new_img = np.zeros((48, 48), dtype=np.float32)
        top = (48 - H) // 2
        left = (48 - W) // 2
        new_img[top:top+H, left:left+W] = img.astype(np.float32)
        return new_img
    elif H == 48 and W == 48:
        return img.astype(np.float32)
    else:
        raise ValueError(f"Imagen con shape inesperado: {img.shape}")

class GalaxyFITS_RGB(Dataset):
    def __init__(self, dir_lenses, dir_nonlenses):
        self.samples = []

        # cargar LENTES
        for fname in sorted(os.listdir(dir_lenses)):
            if fname.endswith("_g.fits"):
                base = fname.replace("_g.fits", "")
                self.samples.append((os.path.join(dir_lenses, base), 1))

        # cargar NO-LENTES
        for fname in sorted(os.listdir(dir_nonlenses)):
            if fname.endswith("_g.fits"):
                base = fname.replace("_g.fits", "")
                self.samples.append((os.path.join(dir_nonlenses, base), 0))

        print(f"Total LENTES: {sum(1 for _, l in self.samples if l == 1)}")
        print(f"Total NO-LENTES: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"TOTAL ENTRENAMIENTO: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def load_fits(self, path):
        data = fits.getdata(path)
        filename = os.path.basename(path)
        band_letter = filename.split("_")[-1].replace(".fits", "")
        col = f"band_{band_letter}"

        if col not in data.dtype.names:
            raise ValueError(f"Columna {col} no existe en {path}. Columnas: {data.dtype.names}")

        img = data[col][0]
        return pad_to_48(img)

    def __getitem__(self, idx):
        base_path, label = self.samples[idx]

        # cargar bandas g, r, i
        g = self.load_fits(f"{base_path}_g.fits")
        r = self.load_fits(f"{base_path}_r.fits")
        i = self.load_fits(f"{base_path}_i.fits")

        # formar pseudo-RGB
        img = np.stack([i, r, g], axis=0)  # (3,48,48)

        return torch.tensor(img, dtype=torch.float32), label, os.path.basename(base_path)


def main():
    print(f"Usando dispositivo: {DEVICE}")
    model = AstroClipModel.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
    model.eval().to(DEVICE)

    dataset = GalaxyFITS_RGB(DIR_LENSES, DIR_NONLENSES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    embeddings = []
    labels = []
    ids = []

    print("\n🔄 Generando embeddings...\n")

    # tqdm para barra de progreso elegante
    progress = tqdm(loader, desc="Procesando batches", unit="batch", ncols=100)

    for batch_imgs, batch_labels, batch_ids in progress:
        batch_imgs = batch_imgs.to(DEVICE)

        with torch.no_grad():
            emb = model.image_encoder(batch_imgs)

        embeddings.append(emb.cpu().numpy())
        labels.append(batch_labels.numpy())
        ids.extend(batch_ids)

        # Mostrar métricas en vivo
        progress.set_postfix({
            "Emb.shape": emb.shape,
            "Batch_size": len(batch_imgs)
        })

    print("\n📦 Guardando resultados...")

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    os.makedirs(os.path.dirname(OUTPUT_EMB), exist_ok=True)

    np.save(OUTPUT_EMB, embeddings)
    np.save(OUTPUT_LAB, labels)

    with open(OUTPUT_IDS, "w") as f:
        for oid in ids:
            f.write(oid + "\n")

    print("\n🎉 Embeddings generados correctamente:")
    print(" →", OUTPUT_EMB)
    print(" →", OUTPUT_LAB)
    print(" →", OUTPUT_IDS)
    print("\n🚀 Proceso finalizado.\n")


if __name__ == "__main__":
    main()
