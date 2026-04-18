"""
setup_check.py — EJECUTAR PRIMERO
===================================
Verifica que todo está correctamente configurado antes de lanzar el pipeline.

Uso en terminal de VS Code:
    python setup_check.py
"""

import os
import sys

print("=" * 60)
print("  VERIFICACIÓN DE CONFIGURACIÓN DEL TFG MEJORADO")
print("=" * 60)

errors = []
warnings = []

# ─────────────────────────────────────────────
# 1. Python
# ─────────────────────────────────────────────
print(f"\n[1] Python: {sys.version}")
if sys.version_info < (3, 8):
    errors.append("Python >= 3.8 requerido")
else:
    print("    ✓ Versión OK")

# ─────────────────────────────────────────────
# 2. Dependencias
# ─────────────────────────────────────────────
print("\n[2] Dependencias:")

try:
    import torch
    print(f"    ✓ torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    ✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        warnings.append("CUDA no disponible — entrenará en CPU (mucho más lento)")
        print("    ⚠ CUDA no disponible (usará CPU)")
except ImportError:
    errors.append("torch no instalado → pip install torch")
    print("    ✗ torch NO encontrado")

try:
    import numpy as np
    print(f"    ✓ numpy {np.__version__}")
except ImportError:
    errors.append("numpy no instalado → pip install numpy")
    print("    ✗ numpy NO encontrado")

try:
    import scipy
    print(f"    ✓ scipy {scipy.__version__}")
except ImportError:
    errors.append("scipy no instalado → pip install scipy")
    print("    ✗ scipy NO encontrado")

try:
    import matplotlib
    print(f"    ✓ matplotlib {matplotlib.__version__}")
except ImportError:
    warnings.append("matplotlib no instalado (opcional, para gráficas)")
    print("    ⚠ matplotlib NO encontrado (opcional)")

# ─────────────────────────────────────────────
# 3. Archivos del proyecto
# ─────────────────────────────────────────────
print("\n[3] Archivos del proyecto:")
required_files = [
    "model.py",
    "data_utils.py",
    "incremental_engine.py",
    "evaluation.py",
    "main_pipeline.py",
]
for f in required_files:
    if os.path.exists(f):
        print(f"    ✓ {f}")
    else:
        errors.append(f"{f} no encontrado en el directorio actual")
        print(f"    ✗ {f} — NO ENCONTRADO")

# ─────────────────────────────────────────────
# 4. Datasets
# ─────────────────────────────────────────────
print("\n[4] Datasets:")

radioml_path = "data/RML2016.10a_dict.pkl"
wisig_path = "data/ManyRX.pkl"

# Buscar RadioML en varias ubicaciones posibles
radioml_found = False
for path in [radioml_path, "RML2016.10a_dict.pkl", "../data/RML2016.10a_dict.pkl"]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    ✓ RadioML encontrado: {path} ({size_mb:.1f} MB)")
        radioml_found = True
        if path != radioml_path:
            print(f"      → Actualiza CONFIG['radioml_path'] = '{path}' en main_pipeline.py")
        break
if not radioml_found:
    errors.append(f"RadioML no encontrado. Colócalo en {radioml_path}")
    print(f"    ✗ RadioML NO encontrado")
    print(f"      Coloca RML2016.10a_dict.pkl en la carpeta data/")

# Buscar WiSig
wisig_found = False
for path in [wisig_path, "ManyRX.pkl", "../data/ManyRX.pkl"]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    ✓ WiSig encontrado: {path} ({size_mb:.1f} MB)")
        wisig_found = True
        if path != wisig_path:
            print(f"      → Actualiza CONFIG['wisig_path'] = '{path}' en main_pipeline.py")
        break
if not wisig_found:
    warnings.append("WiSig no encontrado — Fases 5 y 6 se saltarán")
    print(f"    ⚠ WiSig NO encontrado (Fases 5-6 se saltarán)")
    print(f"      Descárgalo de: https://cores.ee.ucla.edu/downloads/datasets/wisig/")

# Crear carpeta data si no existe
if not os.path.exists("data"):
    os.makedirs("data")
    print("    → Carpeta data/ creada. Coloca los datasets ahí.")

# ─────────────────────────────────────────────
# 5. Test rápido de importación
# ─────────────────────────────────────────────
print("\n[5] Test de importación:")
try:
    from model import SNREstimatorCNN, ResNetSNR, count_parameters
    
    m1 = SNREstimatorCNN()
    m2 = ResNetSNR()
    print(f"    ✓ CNN Baseline:  {count_parameters(m1):>8,} parámetros")
    print(f"    ✓ ResNet-SNR:    {count_parameters(m2):>8,} parámetros")
    
    # Test forward pass
    import torch
    x = torch.randn(2, 2, 128)
    with torch.no_grad():
        out1 = m1(x)
        out2 = m2(x)
    print(f"    ✓ Forward pass OK (baseline: {out1[0]:.2f}, resnet: {out2[0]:.2f})")
    
    from data_utils import IQDataset, NormalizationStrategy
    print("    ✓ data_utils importado")
    
    from incremental_engine import ReplayBuffer, EWC
    print("    ✓ incremental_engine importado")
    
    from evaluation import predict_unlabeled, degradation_test
    print("    ✓ evaluation importado")
    
except Exception as e:
    errors.append(f"Error de importación: {e}")
    print(f"    ✗ Error: {e}")

# ─────────────────────────────────────────────
# RESULTADO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print("  ✗ HAY ERRORES — corrige antes de ejecutar:")
    for e in errors:
        print(f"    → {e}")
else:
    print("  ✓ TODO CORRECTO — listo para ejecutar")

if warnings:
    print("\n  Avisos (no bloquean la ejecución):")
    for w in warnings:
        print(f"    ⚠ {w}")

if not errors:
    print(f"""
  SIGUIENTE PASO:
  ───────────────
  En la terminal de VS Code, ejecuta:

    python main_pipeline.py

  O abre main_pipeline.py y ejecútalo por secciones
  (cada FASE está separada con comentarios claros).
  
  Tiempo estimado con GPU: ~15-25 min
  Tiempo estimado con CPU: ~2-4 horas
""")
else:
    print(f"""
  PARA INSTALAR DEPENDENCIAS:
  ───────────────────────────
    pip install -r requirements.txt

  Si usas conda:
    conda install pytorch numpy scipy matplotlib -c pytorch
""")
