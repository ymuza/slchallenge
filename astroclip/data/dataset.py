# astroclip/data/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from astropy.table import Table

# ---------------------------------------------------------------------
# Utilidad para cargar imágenes FITS y generar triplets
# ---------------------------------------------------------------------
def load_fits_image(path, size=41, strict=False):
    """Carga una imagen FITS como tensor normalizado (3 bandas RGB-like)."""
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)
    except Exception as e:
        if strict:
            raise e
        return None

    # Normalizar y recortar al tamaño esperado
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=0)  # convertir monocanal a 3 canales
    elif data.ndim == 3 and data.shape[0] >= 3:
        data = data[:3]  # solo 3 canales
    else:
        return None

    # Redimensionar al centro si es más grande que `size`
    h, w = data.shape[1], data.shape[2]
    if h > size and w > size:
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        data = data[:, start_h:start_h + size, start_w:start_w + size]

    tensor = torch.from_numpy(data)
    return tensor


def build_triplets(root):
    """Construye lista de tripletas (id, path, band)."""
    triplets = []
    for fname in sorted(os.listdir(root)):
        if fname.endswith(".fits"):
            obj_id = fname.split("_")[0] + "_" + fname.split("_")[1] + "_" + fname.split("_")[2]
            band = fname.split("_")[-1].replace(".fits", "")
            triplets.append((obj_id, os.path.join(root, fname), band))
    return triplets


# ---------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------
class AstroClipDataset(Dataset):
    def __init__(self, root, meta_fits, size=41, strict_fits=False):
        super().__init__()
        self.triplets = build_triplets(root)
        self.size = size
        self.strict = strict_fits

        # cargar catálogo y quedarnos con zlens
        cat = Table.read(meta_fits, format="fits")
        self.zlens = np.array(cat["zlens"], dtype=np.float32)

        # Índices únicos por objeto
        self.ids = sorted(list({t[0] for t in self.triplets}))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        obj_id = self.ids[idx]

        # seleccionar triplets correspondientes al objeto
        obj_triplets = [t for t in self.triplets if t[0] == obj_id]
        images = []
        for _, path, _ in obj_triplets:
            img = load_fits_image(path, self.size, self.strict)
            if img is not None:
                images.append(img)

        if not images:
            raise RuntimeError(f"No se pudieron cargar imágenes para {obj_id}")

        # Usamos la primera imagen válida
        image = images[0]
        redshift = torch.tensor(self.zlens[idx], dtype=torch.float32)
        flag = torch.tensor(True, dtype=torch.bool)

        return {
            "image": image,
            "redshift": redshift,
            "flag": flag,
            "id": obj_id,
        }
