import os
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from tqdm import tqdm
import warnings

# ==============================
# CONFIGURACIÓN
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BASE_PATH = "/media/yamil/b5ef7208-1e9c-40d4-ab93-390950eedbce/astroclip_test_dataset/test_dataset_updated"
OUTPUT_PATH = "outputs/embeddings_real_updated.npy"
IDS_PATH = "outputs/ids_real_updated.npy"

print(f"🧠 Usando dispositivo: {DEVICE}")


# ==============================
# FUNCIÓN PARA LEER FITS ROBUSTO
# ==============================
def read_fits_band(filepath, band_name):
    """
    Lee una banda FITS de forma robusta, manejando diferentes estructuras
    """
    try:
        with fits.open(filepath, memmap=False) as hdul:
            # Inspeccionar estructura
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None:
                    data = hdu.data

                    # Caso 1: Datos en formato tabla con columnas
                    if hasattr(data, 'columns') and band_name in data.columns.names:
                        band_data = data[band_name]
                        # Si tiene dimensión extra, tomar el primer elemento
                        if band_data.ndim > 2:
                            return band_data[0].astype(np.float32)
                        return band_data.astype(np.float32)

                    # Caso 2: Datos directos como array (imagen)
                    elif isinstance(data, np.ndarray):
                        if data.ndim == 2:  # Imagen 2D directa
                            return data.astype(np.float32)
                        elif data.ndim == 3 and data.shape[0] == 1:  # [1, H, W]
                            return data[0].astype(np.float32)
                        elif data.ndim == 3:  # [H, W, C] o similar
                            return data[:, :, 0].astype(np.float32)

            raise ValueError(f"No se encontró banda '{band_name}' en ninguna HDU")

    except Exception as e:
        raise ValueError(f"Error leyendo {filepath}: {e}")


# ==============================
# INSPECCIÓN INICIAL
# ==============================
print("\n🔍 Inspeccionando estructura del primer archivo FITS...")
fits_files = sorted([f for f in os.listdir(BASE_PATH) if f.endswith("_r.fits")])

if len(fits_files) == 0:
    raise RuntimeError("❌ No se encontraron archivos *_r.fits en el directorio")

# Inspeccionar primer archivo
test_file = os.path.join(BASE_PATH, fits_files[0])
print(f"📄 Archivo de prueba: {fits_files[0]}")

with fits.open(test_file, memmap=False) as hdul:
    hdul.info()
    print("\n📊 Estructura de HDUs:")
    for i, hdu in enumerate(hdul):
        print(f"  HDU {i}: {hdu.__class__.__name__}")
        if hasattr(hdu, 'data') and hdu.data is not None:
            if hasattr(hdu.data, 'columns'):
                print(f"    Columnas: {hdu.data.columns.names}")
                for col in hdu.data.columns.names:
                    print(f"      - {col}: shape={hdu.data[col].shape}, dtype={hdu.data[col].dtype}")
            elif isinstance(hdu.data, np.ndarray):
                print(f"    Array shape: {hdu.data.shape}, dtype={hdu.data.dtype}")

print("\n🧪 Probando lectura de bandas...")
try:
    img_r = read_fits_band(test_file, "band_r")
    print(f"✅ band_r leída: shape={img_r.shape}")

    img_g = read_fits_band(test_file.replace("_r.fits", "_g.fits"), "band_g")
    print(f"✅ band_g leída: shape={img_g.shape}")

    img_i = read_fits_band(test_file.replace("_r.fits", "_i.fits"), "band_i")
    print(f"✅ band_i leída: shape={img_i.shape}")
except Exception as e:
    print(f"❌ Error en lectura de prueba: {e}")
    print("\n⚠️  Por favor, revisa la estructura de tus archivos FITS.")
    raise

print(f"\n🔎 Se encontraron {len(fits_files)} objetos (banda r).")
print("🚀 Iniciando procesamiento completo...\n")

# ==============================
# CARGA DE MODELO DINOv2
# ==============================
print("🔹 Cargando modelo DINOv2 (vitl14) desde PyTorch Hub...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
model.eval().to(DEVICE)
print("✅ Modelo DINOv2 cargado.")

embeddings = []
ids = []
errors = []

# ==============================
# PROCESAMIENTO POR LOTES
# ==============================
for i in tqdm(range(0, len(fits_files), BATCH_SIZE), desc="Procesando"):
    batch_imgs = []
    batch_ids = []

    for f in fits_files[i:i + BATCH_SIZE]:
        base_id = f.replace("_r.fits", "")
        try:
            # Leer las tres bandas
            img_r = read_fits_band(os.path.join(BASE_PATH, f), "band_r")
            img_g = read_fits_band(os.path.join(BASE_PATH, f.replace("_r.fits", "_g.fits")), "band_g")
            img_i = read_fits_band(os.path.join(BASE_PATH, f.replace("_r.fits", "_i.fits")), "band_i")

            # Combinar en RGB y normalizar
            rgb = np.stack([img_r, img_g, img_i], axis=-1)
            rgb = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
            rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
            rgb = rgb / (torch.max(rgb) + 1e-6)

            batch_imgs.append(rgb)
            batch_ids.append(base_id)

        except Exception as e:
            errors.append((base_id, str(e)))
            if len(errors) <= 5:  # Mostrar solo los primeros 5 errores
                warnings.warn(f"[WARN] Error procesando {base_id}: {e}")

    if not batch_imgs:
        continue

    try:
        batch_tensor = torch.cat(batch_imgs, dim=0).to(DEVICE)
        with torch.no_grad():
            feats = model(batch_tensor)

        emb = feats.detach().cpu().numpy()
        embeddings.append(emb)
        ids.extend(batch_ids)

        # Liberar memoria GPU
        del batch_tensor, feats
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print("⚠️ GPU sin memoria, pasando procesamiento a CPU para este lote.")
        torch.cuda.empty_cache()
        try:
            batch_tensor = torch.cat(batch_imgs, dim=0).to("cpu")
            with torch.no_grad():
                feats = model.to("cpu")(batch_tensor)
            emb = feats.detach().numpy()
            embeddings.append(emb)
            ids.extend(batch_ids)
        except Exception as e:
            warnings.warn(f"[WARN] Error procesando en CPU: {e}")

# ==============================
# GUARDADO DE RESULTADOS
# ==============================
if len(embeddings) == 0:
    print(f"\n❌ ERROR: No se generaron embeddings válidos.")
    print(f"   Total de errores: {len(errors)}")
    if errors:
        print("\n   Primeros errores:")
        for obj_id, error in errors[:10]:
            print(f"     - {obj_id}: {error}")
    raise RuntimeError("❌ No se generaron embeddings válidos.")

embeddings = np.vstack(embeddings)
np.save(OUTPUT_PATH, embeddings)
np.save(IDS_PATH, np.array(ids))

print(f"\n✅ Guardado: {OUTPUT_PATH} → shape={embeddings.shape}")
print(f"✅ Guardado: {IDS_PATH} → total IDs={len(ids)}")
print(f"⚠️  Objetos con errores: {len(errors)}/{len(fits_files)} ({len(errors) / len(fits_files) * 100:.2f}%)")
print("🎉 Finalizado con éxito.")