import numpy as np
from astropy.io import fits
import os

# --- Rutas de los catálogos ---
LENSES_PATH = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/parameters.fits"
NONLENSES_PATH = "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_nonlenses/parameters.fits"

# --- Leer cantidad de objetos ---
print("📂 Leyendo catálogos...")
with fits.open(LENSES_PATH) as f1:
    n_lenses = len(f1[1].data)
with fits.open(NONLENSES_PATH) as f2:
    n_nonlenses = len(f2[1].data)

print(f"✅ Lentes: {n_lenses}, No-lentes: {n_nonlenses}")

# --- Crear etiquetas ---
# 1 = lente, 0 = no lente
y_lenses = np.ones(n_lenses, dtype=int)
y_nonlenses = np.zeros(n_nonlenses, dtype=int)

# Concatenar en el mismo orden que los embeddings
y_train = np.concatenate([y_lenses, y_nonlenses])

print(f"🧠 Total etiquetas generadas: {len(y_train)}")
print(f"   → {np.sum(y_train)} lentes ({100*np.sum(y_train)/len(y_train):.2f}%)")

# --- Guardar ---
os.makedirs("outputs", exist_ok=True)
np.save("outputs/y_train_lenses.npy", y_train)
print("💾 Guardado en outputs/y_train_lenses.npy ✅")
