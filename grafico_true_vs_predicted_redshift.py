"""
Gráfico de True vs. Predicted Redshift usando KNN Regression
Similar al ejemplo de AstroML
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# ==============================
# CONFIGURACIÓN
# ==============================
EMBEDDINGS_TRAIN = "outputs/embeddings.npy"
Y_TRAIN = "outputs/y_train_lenses.npy"
Z_TRAIN = "outputs/z_train.npy"
OUTPUT_DIR = "outputs/roc_analysis_k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros
K_REDSHIFT = 10  # Número de vecinos para predicción de redshift
TEST_SIZE = 0.2  # Porcentaje para test
RANDOM_STATE = 42

print("🔹 Cargando datos de entrenamiento...")
X_train = np.load(EMBEDDINGS_TRAIN)
y_train = np.load(Y_TRAIN)
z_train = np.load(Z_TRAIN)

print(f"📋 X_train shape: {X_train.shape}")
print(f"📋 y_train shape: {y_train.shape}")
print(f"📋 z_train shape: {z_train.shape}")

# Verificar y ajustar tamaños
n_samples = min(len(X_train), len(y_train), len(z_train))
if len(X_train) != len(y_train) or len(X_train) != len(z_train):
    print(f"\n⚠️  Desajuste detectado en número de muestras!")
    print(f"   🔧 Truncando a las primeras {n_samples} muestras consistentes...")
    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]
    z_train = z_train[:n_samples]

print(f"\n✅ Datos alineados: {n_samples} muestras")

# ==============================
# SPLIT Y NORMALIZACIÓN
# ==============================
print("\n🔹 Dividiendo datos en train/test...")
X_tr, X_te, z_tr, z_te = train_test_split(
    X_train, z_train,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=None  # No stratify para regresión
)

print(f"✅ Train: {len(X_tr)} muestras")
print(f"✅ Test:  {len(X_te)} muestras")

# Normalizar
print("🔹 Normalizando embeddings...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# ==============================
# ENTRENAR KNN REGRESSOR
# ==============================
print(f"\n🔹 Entrenando KNN Regressor (K={K_REDSHIFT})...")
knn_reg = KNeighborsRegressor(
    n_neighbors=K_REDSHIFT,
    weights='distance',
    n_jobs=-1
)
knn_reg.fit(X_tr_scaled, z_tr)
print("✅ Modelo entrenado")

# Predicciones
print("🔹 Generando predicciones...")
z_pred = knn_reg.predict(X_te_scaled)

# ==============================
# MÉTRICAS
# ==============================
mse = mean_squared_error(z_te, z_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(z_te, z_pred)
r2 = r2_score(z_te, z_pred)

print("\n" + "="*60)
print("MÉTRICAS DE PREDICCIÓN DE REDSHIFT")
print("="*60)
print(f"RMSE: {rmse:.5f}")
print(f"MAE:  {mae:.5f}")
print(f"R²:   {r2:.5f}")
print("="*60)

# ==============================
# VISUALIZACIÓN
# ==============================
print("\n🔹 Generando gráfico...")

fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot
ax.scatter(z_te, z_pred, alpha=0.5, s=20, color='#4A90E2', edgecolors='none')

# Línea de identidad (predicción perfecta)
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect prediction')

# Etiquetas y título
ax.set_xlabel('True Redshift', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Redshift', fontsize=14, fontweight='bold')
ax.set_title('True vs. Predicted Redshift', fontsize=16, fontweight='bold')

# Agregar métricas en el gráfico
textstr = f'K = {K_REDSHIFT}\n'
textstr += f'RMSE = {rmse:.5f}\n'
textstr += f'MAE = {mae:.5f}\n'
textstr += f'R² = {r2:.5f}\n'
textstr += f'N = {len(z_te)}'

props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

# Grid y límites fijos entre 0 y 1
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='lower right', fontsize=11)

# Límites fijos 0 a 1
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()

# Guardar
output_path = os.path.join(OUTPUT_DIR, 'true_vs_predicted_redshift.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"📊 Gráfico guardado: {output_path}")

# ==============================
# ANÁLISIS DE RESIDUOS
# ==============================
print("\n🔹 Generando análisis de residuos...")

residuals = z_pred - z_te

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Residuos vs True Redshift
axes[0].scatter(z_te, residuals, alpha=0.5, s=20, color='#E74C3C', edgecolors='none')
axes[0].axhline(y=0, color='k', linestyle='--', lw=2)
axes[0].set_xlabel('True Redshift', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Residuals (Predicted - True)', fontsize=12, fontweight='bold')
axes[0].set_title('Residuals vs. True Redshift', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# 2. Histograma de residuos
axes[1].hist(residuals, bins=50, color='#9B59B6', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='k', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

# Estadísticas en histograma
mean_res = np.mean(residuals)
std_res = np.std(residuals)
textstr_res = f'Mean = {mean_res:.5f}\nStd = {std_res:.5f}'
props_res = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
axes[1].text(0.05, 0.95, textstr_res, transform=axes[1].transAxes,
             fontsize=11, verticalalignment='top', bbox=props_res, family='monospace')

plt.tight_layout()

# Guardar
residuals_path = os.path.join(OUTPUT_DIR, 'residuals_analysis.png')
plt.savefig(residuals_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"📊 Análisis de residuos guardado: {residuals_path}")

print("\n✅ Análisis de redshift completado!")