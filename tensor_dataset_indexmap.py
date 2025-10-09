import os, re, glob, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from astropy.table import Table

"""
Lee imágenes FITS en tripletas (r, g, i).

Normaliza cada banda con robust_asinh_normalize.

Hace center_crop o resize a 96×96 para que sea compatible con el backbone DINOv2 (patch size = 12).

Devuelve un diccionario con:

{
  "image": tensor(3,H,W),
  "redshift": z,
  "is_lensed": flag,
  "id": base_id
}
Con esto se construyeron dataloaders tanto para lentes como para no-lentes.
"""




# --- Normalización ---
def robust_asinh_normalize(img, clip_lo=1.0, clip_hi=99.0, eps=1e-6):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(img)
    mad = np.median(np.abs(img - med)) + eps
    x = (img - med) / (1.4826 * mad)
    x = np.arcsinh(x)
    lo, hi = np.percentile(x, [clip_lo, clip_hi])
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def center_crop(arr, out_size):
    h, w = arr.shape
    if h == out_size and w == out_size:
        return arr
    top = max((h - out_size) // 2, 0)
    left = max((w - out_size) // 2, 0)
    return arr[top:top+out_size, left:left+out_size]

def extract_base_id(path):
    name = os.path.basename(path)
    return re.sub(r'_(r|g|i)\.(fits|fit|fz)$', '', name, flags=re.IGNORECASE)

def build_triplets(root, bands=('r','g','i')):
    files = glob.glob(os.path.join(root, "*.fits"))
    buckets = {}
    for p in files:
        m = re.search(r'_(r|g|i)\.(fits|fit|fz)$', os.path.basename(p).lower())
        if not m: continue
        band = m.group(1)
        base_id = extract_base_id(p)
        d = buckets.setdefault(base_id, {})
        d[band] = p
    triplets = []
    for base_id, dd in sorted(buckets.items()):
        if all(b in dd for b in bands):
            triplets.append((base_id, dd['r'], dd['g'], dd['i']))
    return triplets

# --- Dataset ---
class GalaxyBatchDataset(Dataset):
    def __init__(self, root, meta_fits, size=96, strict_fits=False):
        super().__init__()
        self.triplets = build_triplets(root)
        self.size = size
        self.strict = strict_fits

        # cargar catálogo y quedarnos con zlens (si existe)
        try:
            cat = Table.read(meta_fits, format="fits")
            self.zlens = np.array(cat["zlens"], dtype=np.float32)
        except Exception as e:
            warnings.warn(f"[WARN] No se pudo leer catálogo {meta_fits}: {e}")
            self.zlens = np.zeros(len(self.triplets), dtype=np.float32)

        # chequeo de consistencia
        if len(self.triplets) != len(self.zlens):
            warnings.warn(
                f"Atención: {len(self.triplets)} imágenes vs {len(self.zlens)} filas en catálogo"
            )

    def __len__(self):
        return len(self.triplets)

    def _safe_getdata(self, path):
        try:
            with fits.open(path, memmap=True) as hdul:
                data = hdul[0].data
                if isinstance(data, np.ndarray) and data.ndim == 2:
                    return np.array(data, dtype=np.float32)
                if data is None and len(hdul) > 1:
                    data = hdul[1].data
                if hasattr(data, "dtype") and data.dtype.names:
                    for name in data.dtype.names:
                        if "band" in name.lower():
                            return np.array(data[name][0], dtype=np.float32)
        except Exception as e:
            if self.strict: raise
            warnings.warn(f"[WARN] Saltando FITS: {path} ({e})")
            return None

    def __getitem__(self, idx):
        base_id, r_path, g_path, i_path = self.triplets[idx]
        arrs = []
        for p in (r_path,g_path,i_path):
            a = self._safe_getdata(p)
            if a is None:
                a = np.zeros((self.size,self.size), dtype=np.float32)
            a = robust_asinh_normalize(a)
            arrs.append(a)

        # construir tensor [3,H,W]
        img = torch.from_numpy(np.stack(arrs,axis=0)).contiguous()

        # 🔧 Forzar resize a self.size (ej: 96x96)
        if img.shape[1] != self.size or img.shape[2] != self.size:
            img = F.interpolate(
                img.unsqueeze(0),  # [1,3,H,W]
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        # redshift tomado de la fila correspondiente
        z = self.zlens[idx] if idx < len(self.zlens) else 0.0
        is_lens = 1  # todo este dataset son lentes

        return {
            "image": img,
            "redshift": torch.tensor(z, dtype=torch.float32),
            "is_lensed": torch.tensor(is_lens, dtype=torch.bool),
            "id": base_id
        }

# --- Demo rápido ---
if __name__ == "__main__":
    ROOT = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/hsc_lenses"
    META = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/parameters.fits"

    ds = GalaxyBatchDataset(root=ROOT, meta_fits=META, size=96)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    batch = next(iter(loader))
    print("Images:", batch["image"].shape)       # [32, 3, 96, 96]
    print("Redshifts:", batch["redshift"][:5])  # primeros valores
    print("Flags:", batch["is_lensed"][:5])     # todos True
    print("IDs:", batch["id"][:5])
