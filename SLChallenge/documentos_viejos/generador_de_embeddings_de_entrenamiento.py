import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from astropy.io import fits
from astroclip.models.astroclip import AstroClipModel
from SLChallenge.documentos_viejos.tensor_dataset_indexmap import GalaxyBatchDataset


"""Carga el modelo preentrenado AstroCLIP (astroclip.ckpt).
Extrae embeddings de dimensión 1024 para todas las imágenes (lenses + non-lenses).
Exporta a:
outputs/embeddings.npy → matriz (100000, 1024).
outputs/labels.npy → 1 = lente, 0 = no-lente.
outputs/ids.txt → IDs de cada objeto.
✅ Así se obtiene un espacio latente donde comparar y entrenar clasificadores."""



# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "/home/yamil/doctorado/AstroCLIP/pre_trained_model/astroclip.ckpt"

# Paths
LENSES_ROOT = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/hsc_lenses"
NONLENSES_ROOT = "/media/yamil/nvmeBlue/challenge_data/images/hsc_nonlenses/hsc_nonlenses"
META_PATH = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/parameters.fits"

# ---------------- HELPERS ----------------
def get_lens_id_from_fits(path):
    """Extrae el Lens ID desde el HDU1 del FITS."""
    try:
        with fits.open(path, memmap=True) as hdul:
            if len(hdul) > 1 and "Lens ID" in hdul[1].columns.names:
                return str(hdul[1].data["Lens ID"][0])
    except Exception as e:
        print(f"[WARN] No se pudo leer Lens ID de {path}: {e}")
    return None

def extract_embeddings(root, label, model):
    ds = GalaxyBatchDataset(root=root, meta_fits=None, size=96)  # no necesita catálogo aquí
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    all_emb, all_ids, all_labels = [], [], []

    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        outs = model(imgs, input_type="image").detach().cpu().numpy()
        all_emb.append(outs)

        # Extraemos Lens IDs verdaderos desde los FITS
        for fid in batch["id"]:
            fname = os.path.join(root, fid + "_r.fits")  # usamos canal r
            lid = get_lens_id_from_fits(fname)
            all_ids.append(lid if lid is not None else fid)

        all_labels.extend([label] * imgs.size(0))

    return np.vstack(all_emb), np.array(all_ids), np.array(all_labels)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("✅ Cargando modelo...")
    model = AstroClipModel.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
    model.eval().to(DEVICE)
    print("✅ Modelo cargado correctamente")

    print("🔵 Procesando LENSES...")
    emb_lens, ids_lens, labels_lens = extract_embeddings(LENSES_ROOT, 1, model)

    print("🟢 Procesando NON-LENSES...")
    emb_non, ids_non, labels_non = extract_embeddings(NONLENSES_ROOT, 0, model)

    # Concatenamos
    embeddings = np.vstack([emb_lens, emb_non])
    ids = np.concatenate([ids_lens, ids_non])
    labels = np.concatenate([labels_lens, labels_non])

    # Guardamos
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/embeddings.npy", embeddings)
    np.save("outputs/labels.npy", labels)
    np.savetxt("outputs/ids.txt", ids, fmt="%s")

    print(f"✅ Embeddings finales: {embeddings.shape}")
    print(f"   - Lentes: {sum(labels==1)} | No-lentes: {sum(labels==0)}")
    print("💾 Guardados en outputs/embeddings.npy, outputs/labels.npy y outputs/ids.txt")
