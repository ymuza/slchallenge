import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_train = np.load("outputs/embeddings_test.npy")
labels = np.load("outputs/labels_test.npy")
X_test = np.load("outputs/embeddings_test.npy")  # los del conjunto real

pca = PCA(n_components=2)
proj = pca.fit_transform(np.vstack([X_train, X_test]))
train_proj, test_proj = proj[:len(X_train)], proj[len(X_train):]

plt.figure(figsize=(7,7))
plt.scatter(train_proj[labels==0,0], train_proj[labels==0,1], alpha=0.3, label="Non-lens train", s=5)
plt.scatter(train_proj[labels==1,0], train_proj[labels==1,1], alpha=0.3, label="Lens train", s=5)
plt.scatter(test_proj[:,0], test_proj[:,1], alpha=0.3, label="Real test", s=5, c="green")
plt.legend(); plt.title("PCA de embeddings (train vs real)")
plt.tight_layout()
plt.savefig("outputs/embeddings_space_verification.png", dpi=150)
print("✅ Gráfico guardado en outputs/embeddings_space_verification.png")

