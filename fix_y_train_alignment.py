import numpy as np

X_train = np.load("outputs/embeddings.npy")
y_train = np.load("outputs/y_train_lenses.npy")

print(f"Antes → X_train: {X_train.shape}, y_train: {y_train.shape}")

# Alinear
min_len = min(len(X_train), len(y_train))
X_train = X_train[:min_len]
y_train = y_train[:min_len]

# Guardar versiones corregidas
np.save("outputs/embeddings_aligned.npy", X_train)
np.save("outputs/y_train_lenses_aligned.npy", y_train)

print(f"✅ Después → X_train: {X_train.shape}, y_train: {y_train.shape}")
print("💾 Archivos guardados en outputs/ (aligned)")
