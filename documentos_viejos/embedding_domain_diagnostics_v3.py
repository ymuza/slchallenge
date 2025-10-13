# embedding_domain_diagnostics_v3.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# ==============================
# ⚙️ CONFIGURACIÓN INICIAL
# ==============================
os.makedirs("../outputs", exist_ok=True)
TRAIN_PATH = "../outputs/embeddings.npy"  # AstroCLIP train (1024D)
TEST_PATH = "../outputs/embeddings_test.npy"  # DINOv2 real (768D)
PDF_PATH = "../outputs/domain_diagnostics_report.pdf"
PLOT_PATH = "../outputs/domain_diagnostics_v3.png"

print("🔹 Cargando embeddings...")
X_train = np.load(TRAIN_PATH)
X_test = np.load(TEST_PATH)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# ==============================
# 🧠 REDUCCIÓN PCA COHERENTE
# ==============================
if X_train.shape[1] != X_test.shape[1]:
    target_dim = min(X_train.shape[1], X_test.shape[1])
    print(f"⚙️ Aplicando PCA conjunta a {target_dim} dimensiones...")
    pca = PCA(n_components=target_dim, random_state=42)
    concat = np.concatenate([
        X_train[:, :target_dim] if X_train.shape[1] > target_dim else X_train,
        X_test[:, :target_dim] if X_test.shape[1] > target_dim else X_test
    ], axis=0)
    pca.fit(concat)
    X_train = pca.transform(X_train[:, :pca.n_features_in_])
    X_test = pca.transform(X_test[:, :pca.n_features_in_])
    print(f"✅ Nuevas formas: Train {X_train.shape}, Test {X_test.shape}")
else:
    print("✅ Dimensiones ya coinciden, no se aplica PCA.")

# ==============================
# 📊 ESTADÍSTICAS BÁSICAS
# ==============================
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
test_mean = X_test.mean(axis=0)
test_std = X_test.std(axis=0)

mean_diff = np.abs(train_mean - test_mean)
std_diff = np.abs(train_std - test_std)
mean_diff_global = mean_diff.mean()
std_diff_global = std_diff.mean()
l2_distance = np.linalg.norm(train_mean - test_mean)

print(f"\n📊 Diferencias promedio por dimensión:")
print(f"  • Media abs diff = {mean_diff_global:.6f}")
print(f"  • STD abs diff   = {std_diff_global:.6f}")
print(f"\n📏 Distancia L2 entre medias: {l2_distance:.6f}")

# --- Dimensiones más distintas ---
top_dims = np.argsort(mean_diff)[-10:][::-1]
top_table = [(i, mean_diff[i], std_diff[i]) for i in top_dims]
print("\n🔎 Top 10 dimensiones con mayor diferencia media:")
for i, d_m, d_s in top_table:
    print(f"  Dim {i:3d} → Δmean={d_m:.5f}, Δstd={d_s:.5f}")

# ==============================
# 📈 GRÁFICOS
# ==============================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(train_mean, color="blue", label="Train mean", fill=True, alpha=0.4)
sns.kdeplot(test_mean, color="orange", label="Test mean", fill=True, alpha=0.4)
plt.title("Distribución de medias por dimensión")
plt.xlabel("Valor medio")
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(train_std, color="blue", label="Train std", fill=True, alpha=0.4)
sns.kdeplot(test_std, color="orange", label="Test std", fill=True, alpha=0.4)
plt.title("Distribución de desviaciones estándar por dimensión")
plt.xlabel("Desviación estándar")
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
plt.close()
print(f"\n📈 Gráfico guardado en: {PLOT_PATH}")

# ==============================
# 🚨 DETECCIÓN DE OUTLIERS
# ==============================
threshold = 3 * train_std.mean()
outlier_dims = np.where(mean_diff > threshold)[0]
print(f"\n⚠️ Dimensiones fuera de rango esperado: {len(outlier_dims)} / {X_train.shape[1]}")
if len(outlier_dims) > 0:
    print(f"Ejemplo: {outlier_dims[:10]}")
else:
    print("✅ No se detectaron outliers dimensionales.")

# ==============================
# 🧾 EXPORTACIÓN A PDF CIENTÍFICO
# ==============================
print("\n📝 Generando informe PDF...")

c = canvas.Canvas(PDF_PATH, pagesize=letter)
width, height = letter
c.setFont("Helvetica-Bold", 14)
c.drawString(50, height - 50, "Embedding Domain Diagnostics Report")
c.setFont("Helvetica", 11)
c.drawString(50, height - 80, f"Train shape: {X_train.shape}")
c.drawString(50, height - 95, f"Test shape: {X_test.shape}")
c.drawString(50, height - 115, f"Mean abs diff: {mean_diff_global:.6f}")
c.drawString(50, height - 130, f"STD abs diff: {std_diff_global:.6f}")
c.drawString(50, height - 145, f"L2 distance: {l2_distance:.6f}")
c.drawString(50, height - 165, f"Outlier dims: {len(outlier_dims)}")

c.setFont("Helvetica-Bold", 12)
c.drawString(50, height - 190, "Top 10 dimensiones más distintas:")
c.setFont("Helvetica", 10)
y = height - 210
for i, d_m, d_s in top_table:
    c.drawString(60, y, f"Dim {i:3d} → Δmean={d_m:.5f}, Δstd={d_s:.5f}")
    y -= 15

# Inserta la figura de comparación
if os.path.exists(PLOT_PATH):
    c.drawImage(PLOT_PATH, 50, 100, width=500, height=300)

c.save()
print(f"✅ Informe PDF generado en: {PDF_PATH}")

print("\n🎯 Diagnóstico completo: dominio entrenado vs real analizado y exportado.")
