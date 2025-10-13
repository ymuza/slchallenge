import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
K_REDSHIFT = 5  # Número de vecinos para predicción de redshift
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

print("\n" + "=" * 60)
print("MÉTRICAS DE PREDICCIÓN DE REDSHIFT")
print("=" * 60)
print(f"RMSE: {rmse:.5f}")
print(f"MAE:  {mae:.5f}")
print(f"R²:   {r2:.5f}")
print("=" * 60)

# ==============================
# VISUALIZACIÓN CON SEABORN (ISODENSIDAD)
# ==============================
print("\n🔹 Generando gráfico de isodensidad con seaborn...")

# Configurar estilo seaborn
sns.set_style("whitegrid")
sns.set_palette("viridis")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ==============================
# GRÁFICO 1: SCATTER PLOT CON KDE (ISODENSIDAD)
# ==============================
# Crear un jointplot style manualmente en el primer subplot
scatter = axes[0].scatter(z_te, z_pred, alpha=0.3, s=10, c='blue', edgecolors='none')

# Línea de identidad (predicción perfecta)
axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect prediction')

# Agregar contornos de densidad (isodensidad)
# Filtrar datos dentro del rango [0,1] para evitar problemas con KDE
mask = (z_te >= 0) & (z_te <= 1) & (z_pred >= 0) & (z_pred <= 1)
z_te_filtered = z_te[mask]
z_pred_filtered = z_pred[mask]

if len(z_te_filtered) > 0:
    sns.kdeplot(
        x=z_te_filtered,
        y=z_pred_filtered,
        ax=axes[0],
        levels=8,  # Número de niveles de contorno
        fill=False,  # Contornos sin relleno
        colors='darkred',
        linewidths=1.5,
        alpha=0.8
    )

axes[0].set_xlabel('True Redshift', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Predicted Redshift', fontsize=14, fontweight='bold')
axes[0].set_title('True vs. Predicted Redshift\n(Con contornos de densidad)', fontsize=14, fontweight='bold')

# Agregar métricas en el gráfico
textstr = f'K = {K_REDSHIFT}\n'
textstr += f'RMSE = {rmse:.5f}\n'
textstr += f'MAE = {mae:.5f}\n'
textstr += f'R² = {r2:.5f}\n'
textstr += f'N = {len(z_te)}'

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')

axes[0].set_aspect('equal', adjustable='box')
axes[0].legend(loc='lower right', fontsize=11)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3)

# ==============================
# GRÁFICO 2: HEXBIN PLOT CON CONTOURS (ALTERNATIVA)
# ==============================
# Hexbin con contornos superpuestos
hexbin = axes[1].hexbin(z_te, z_pred, gridsize=30, cmap='Blues', alpha=0.7, mincnt=1)

# Línea de identidad
axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect prediction')

# Agregar contornos de densidad
if len(z_te_filtered) > 0:
    sns.kdeplot(
        x=z_te_filtered,
        y=z_pred_filtered,
        ax=axes[1],
        levels=5,
        fill=False,
        colors='red',
        linewidths=2,
        alpha=0.8
    )

axes[1].set_xlabel('True Redshift', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Predicted Redshift', fontsize=14, fontweight='bold')
axes[1].set_title('True vs. Predicted Redshift\n(Hexbin + Contornos)', fontsize=14, fontweight='bold')

# Barra de color para hexbin
cbar = plt.colorbar(hexbin, ax=axes[1])
cbar.set_label('Densidad', fontsize=12)

axes[1].set_aspect('equal', adjustable='box')
axes[1].legend(loc='lower right', fontsize=11)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.tight_layout()

# Guardar gráfico de isodensidad
isodensity_path = os.path.join(OUTPUT_DIR, 'true_vs_predicted_redshift_isodensity.png')
plt.savefig(isodensity_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"📊 Gráfico de isodensidad guardado: {isodensity_path}")

# ==============================
# GRÁFICO ADICIONAL CORREGIDO: SOLO CONTOUR PLOT
# ==============================
print("\n🔹 Generando gráfico solo de contornos (versión corregida)...")

plt.figure(figsize=(10, 8))

# Crear un contour plot usando hist2d y contornos
if len(z_te_filtered) > 0:
    # Método 1: Usar hist2d para crear una grilla de densidad
    H, xedges, yedges = np.histogram2d(z_te_filtered, z_pred_filtered, bins=30)

    # Calcular los centros de los bins
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Crear contour plot
    contour = plt.contourf(xcenters, ycenters, H.T, levels=15, alpha=0.7, cmap='viridis')

    # Agregar líneas de contorno
    plt.contour(xcenters, ycenters, H.T, levels=15, colors='black', alpha=0.3, linewidths=0.5)

    # Scatter plot de algunos puntos para referencia (opcional)
    plt.scatter(z_te, z_pred, alpha=0.1, s=2, color='blue', edgecolors='none')
else:
    # Fallback si no hay datos filtrados
    plt.scatter(z_te, z_pred, alpha=0.5, s=20, color='blue', edgecolors='none')

# Línea de identidad
plt.plot([0, 1], [0, 1], 'r--', lw=3, label='Perfect prediction', alpha=0.8)

plt.xlabel('True Redshift', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Redshift', fontsize=14, fontweight='bold')
plt.title('True vs. Predicted Redshift\n(Contour Plot - Densidad)', fontsize=16, fontweight='bold')

# Barra de color
if len(z_te_filtered) > 0:
    plt.colorbar(contour, label='Densidad')

# Métricas
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props, family='monospace')

plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()

# Guardar contour plot
contour_path = os.path.join(OUTPUT_DIR, 'true_vs_predicted_redshift_contour.png')
plt.savefig(contour_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"📊 Gráfico de contornos guardado: {contour_path}")

# ==============================
# GRÁFICO EXTRA: JOINTPLOT DE SEABORN
# ==============================
print("\n🔹 Generando jointplot de seaborn...")

try:
    # Crear un jointplot que combina scatter, histogramas y KDE
    g = sns.jointplot(
        x=z_te,
        y=z_pred,
        kind='scatter',
        alpha=0.5,
        s=10,
        color='blue',
        marginal_kws=dict(bins=30, fill=True),
        height=8
    )

    # Agregar línea de identidad
    g.ax_joint.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')
    g.ax_joint.legend()

    # Agregar contornos de densidad
    sns.kdeplot(
        x=z_te,
        y=z_pred,
        ax=g.ax_joint,
        levels=6,
        fill=False,
        colors='red',
        linewidths=1.5
    )

    # Agregar métricas
    g.ax_joint.text(0.05, 0.95, textstr, transform=g.ax_joint.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, family='monospace')

    g.ax_joint.set_xlim([0, 1])
    g.ax_joint.set_ylim([0, 1])
    g.ax_joint.set_aspect('equal')

    g.set_axis_labels('True Redshift', 'Predicted Redshift', fontsize=12, fontweight='bold')
    g.fig.suptitle('True vs. Predicted Redshift\n(JointPlot)', fontsize=14, fontweight='bold')
    g.fig.tight_layout()

    # Guardar jointplot
    jointplot_path = os.path.join(OUTPUT_DIR, 'true_vs_predicted_redshift_jointplot.png')
    plt.savefig(jointplot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"📊 Jointplot guardado: {jointplot_path}")

except Exception as e:
    print(f"⚠️  Error generando jointplot: {e}")

print("\n✅ Análisis de redshift con gráficos de isodensidad completado!")