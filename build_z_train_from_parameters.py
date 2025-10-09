"""
Abre ambos FITS (lenses y nonlenses).
Detecta la columna correcta del redshift (usualmente zlens o z_lens).
Combina ambos arrays en uno solo.
Lo guarda como outputs/z_train.npy.
"""
import numpy as np
from astropy.io import fits
import os

os.makedirs("outputs", exist_ok=True)

paths = [
    "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_lenses/parameters.fits",
    "/media/yamil/nvmeBlue/challenge_data/test_images/hsc_nonlenses/parameters.fits",
]

z_all = []

for path in paths:
    with fits.open(path, memmap=False) as hdul:
        cols = [c.lower() for c in hdul[1].columns.names]
        print(f"📂 Leyendo {path}")
        print("  Columnas disponibles:", cols)

        if "zlens" in cols:
            z = hdul[1].data["zlens"]
            print(f"  → Usando columna 'zlens' ({len(z)} entradas)")
        elif "z_central" in cols:
            z = hdul[1].data["z_central"]
            print(f"  → Usando columna 'z_central' ({len(z)} entradas)")
        elif "photoz" in cols:
            z = hdul[1].data["photoz"]
            print(f"  → Usando columna 'photoz' ({len(z)} entradas)")
        elif "z" in cols:
            z = hdul[1].data["z"]
            print(f"  → Usando columna 'z' ({len(z)} entradas)")
        else:
            raise ValueError(f"No se encontró columna de redshift en {path}")

        z_all.append(np.array(z, dtype=np.float32))

z_train = np.concatenate(z_all)
print(f"\n✅ Total de redshifts cargados: {len(z_train)}")

np.save("outputs/z_train.npy", z_train)
print("💾 Guardado en outputs/z_train.npy")
