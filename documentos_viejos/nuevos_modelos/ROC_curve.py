"""
Análisis de curvas ROC para diferentes valores de K en KNN
Basado en: https://www.astroml.org/astroML-notebooks/chapter9/astroml_chapter9_Classification.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import os

# ==============================
# CONFIGURACIÓN
# ==============================

EMB_TEST = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_fixed.npy"
IDS_TEST = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/embeddings_test_ids_fixed.npy"


EMBEDDINGS_TRAIN = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/train_embeddings_dino_224.npy"
Y_TRAIN = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/y_train_lenses_aligned.npy"
Z_TRAIN = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/embeddings/z_train.npy"
OUTPUT_DIR = "/media/yamil/nvmeBlue/deep k-Correct/deep-KCorrect/SLChallenge/nuevos_modelos/KNN/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Valores de K a probar
K_VALUES = [3, 5, 7, 10, 15, 20, 30, 50, 100]
N_FOLDS = 5  # Para cross-validation

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
    print(f"   X_train: {len(X_train)} | y_train: {len(y_train)} | z_train: {len(z_train)}")
    print(f"   🔧 Truncando a las primeras {n_samples} muestras consistentes...")
    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]
    z_train = z_train[:n_samples]

print(f"\n✅ Datos alineados:")
print(f"   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   z_train shape: {z_train.shape}")
print(f"   Proporción de lentes: {y_train.mean():.4f}")

# Normalización (opcional pero recomendado para KNN)
print("🔹 Normalizando embeddings...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==============================
# CROSS-VALIDATION CON DIFERENTES K
# ==============================
print(f"\n🔹 Realizando {N_FOLDS}-fold cross-validation para K={K_VALUES}...")

results = {}
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for k in K_VALUES:
    print(f"\n  📊 Evaluando K={k}...")

    all_y_true = []
    all_y_proba = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Entrenar KNN
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        knn.fit(X_tr, y_tr)

        # Predecir probabilidades
        y_proba = knn.predict_proba(X_val)[:, 1]

        all_y_true.extend(y_val)
        all_y_proba.extend(y_proba)

        print(f"    Fold {fold + 1}/{N_FOLDS} completado")

    # Calcular ROC
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_proba)
    roc_auc = auc(fpr, tpr)

    results[k] = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'y_true': np.array(all_y_true),
        'y_proba': np.array(all_y_proba)
    }

    print(f"    ✅ K={k} → AUC = {roc_auc:.4f}")

# ==============================
# VISUALIZACIÓN: CURVAS ROC
# ==============================
print("\n🔹 Generando visualizaciones...")

# 1. Todas las curvas ROC juntas
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(K_VALUES)))

for (k, res), color in zip(results.items(), colors):
    plt.plot(res['fpr'], res['tpr'], lw=2, color=color,
             label=f'K={k} (AUC={res["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves para diferentes valores de K - KNN', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_all_path = os.path.join(OUTPUT_DIR, 'roc_curves_all_k.png')
plt.savefig(roc_all_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"📈 Curvas ROC guardadas: {roc_all_path}")

# 2. AUC vs K
plt.figure(figsize=(10, 6))
k_vals = list(results.keys())
auc_vals = [results[k]['auc'] for k in k_vals]

plt.plot(k_vals, auc_vals, 'o-', linewidth=2, markersize=8, color='#2E86AB')
plt.xlabel('Número de vecinos (K)', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.title('AUC vs K - Rendimiento del clasificador KNN', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.axhline(y=max(auc_vals), color='r', linestyle='--', alpha=0.5,
            label=f'Mejor AUC = {max(auc_vals):.4f} (K={k_vals[np.argmax(auc_vals)]})')
plt.legend(fontsize=10)
plt.tight_layout()
auc_k_path = os.path.join(OUTPUT_DIR, 'auc_vs_k.png')
plt.savefig(auc_k_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"📈 AUC vs K guardado: {auc_k_path}")

# 3. Comparación TPR a diferentes FPR
fpr_thresholds = [0.01, 0.05, 0.1, 0.2]
plt.figure(figsize=(10, 6))

for fpr_thresh in fpr_thresholds:
    tpr_at_fpr = []
    for k in k_vals:
        fpr = results[k]['fpr']
        tpr = results[k]['tpr']
        # Encontrar TPR al FPR más cercano
        idx = np.argmin(np.abs(fpr - fpr_thresh))
        tpr_at_fpr.append(tpr[idx])

    plt.plot(k_vals, tpr_at_fpr, 'o-', linewidth=2, markersize=6,
             label=f'TPR @ FPR={fpr_thresh}')

plt.xlabel('Número de vecinos (K)', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('TPR a diferentes niveles de FPR vs K', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
tpr_fpr_path = os.path.join(OUTPUT_DIR, 'tpr_at_fpr_vs_k.png')
plt.savefig(tpr_fpr_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"📈 TPR vs K guardado: {tpr_fpr_path}")

# ==============================
# RESUMEN DE RESULTADOS
# ==============================
print("\n" + "=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)

best_k = k_vals[np.argmax(auc_vals)]
best_auc = max(auc_vals)

print(f"\n🏆 Mejor configuración:")
print(f"   K = {best_k}")
print(f"   AUC = {best_auc:.4f}")

print(f"\n📊 Resultados detallados por K:")
print(f"{'K':<8} {'AUC':<10} {'TPR@FPR=0.05':<15} {'TPR@FPR=0.1':<15}")
print("-" * 50)

for k in k_vals:
    fpr = results[k]['fpr']
    tpr = results[k]['tpr']
    auc_val = results[k]['auc']

    # TPR a FPR=0.05
    idx_005 = np.argmin(np.abs(fpr - 0.05))
    tpr_005 = tpr[idx_005]

    # TPR a FPR=0.1
    idx_01 = np.argmin(np.abs(fpr - 0.1))
    tpr_01 = tpr[idx_01]

    print(f"{k:<8} {auc_val:<10.4f} {tpr_005:<15.4f} {tpr_01:<15.4f}")

print("\n✅ Análisis completado. Resultados guardados en:", OUTPUT_DIR)

# ==============================
# GUARDAR RESULTADOS NUMÉRICOS
# ==============================
results_summary = {
    'k_values': k_vals,
    'auc_scores': auc_vals,
    'best_k': best_k,
    'best_auc': best_auc
}

np.save(os.path.join(OUTPUT_DIR, 'results_summary.npy'), results_summary)
print(f"💾 Resumen numérico guardado: {os.path.join(OUTPUT_DIR, 'results_summary.npy')}")