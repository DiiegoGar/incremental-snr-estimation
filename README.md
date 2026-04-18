# TFG Mejorado — Estimación Incremental de SNR en Señales WiFi

## Estructura del Proyecto

```
tfg_mejorado/
├── model.py                  # Arquitecturas: CNN Baseline + ResNet-1D-SNR
├── data_utils.py             # Carga de datos, normalización, pseudo-labeling
├── incremental_engine.py     # Motor CL: Herding Replay + Feature-KD + EWC
├── evaluation.py             # Evaluación cross-domain, M2M4, calibración
├── main_pipeline.py          # Pipeline maestro (ejecutar paso a paso)
├── README.md                 # Este archivo
└── data/                     # (Crear y colocar los datasets aquí)
    ├── RML2016.10a_dict.pkl
    └── ManyRX.pkl
```

## Mejoras Implementadas vs Notebook Original

| Aspecto | Original | Mejorado |
|---------|----------|----------|
| Arquitectura | CNN 3 capas (45K params) | ResNet-1D 4 bloques (350K params) |
| Normalización | Z-score global (falla cross-domain) | InstanceNorm interna (domain-agnostic) |
| Replay Buffer | Selección aleatoria | Herding selection (tipo iCaRL) |
| Knowledge Distillation | Solo output-level | Output + Feature-level KD |
| Regularización | Ninguna | EWC (Elastic Weight Consolidation) |
| Optimizador | Adam + StepLR | AdamW + Cosine Annealing |
| Métricas CL | Solo forgetting | AA, BWT, Forgetting completo |
| WiSig | Zero-shot sin cuantificar | Tarea 5 con pseudo-labels + M2M4 |
| Evaluación | Cualitativa | Spearman ρ, sensibilidad, calibración |

## Cómo Usar

### Opción 1: Como script (ejecutar completo)
```bash
cd tfg_mejorado
python main_pipeline.py
```

### Opción 2: Como notebook (paso a paso)
Copiar cada sección de `main_pipeline.py` como celdas separadas
en un Jupyter notebook. Las secciones están delimitadas por:
```python
# ################################################################
# FASE X: ...
# ################################################################
```

### Opción 3: Importar módulos en tu notebook existente
```python
import sys
sys.path.insert(0, 'ruta/a/tfg_mejorado')

from model import ResNetSNR
from data_utils import load_radioml, build_task_data
from incremental_engine import run_incremental_pipeline, print_cl_metrics
from evaluation import degradation_test, compare_with_m2m4
```

## Requisitos
```
torch >= 2.0
numpy
scipy
```

## Configuración
Ajustar `CONFIG` en `main_pipeline.py`:
- `radioml_path`: ruta al pickle de RadioML 2016.10a
- `wisig_path`: ruta al pickle de WiSig ManyRX
- `buffer_capacity`: tamaño del replay buffer (10000 por defecto)
- `lambda_kd`, `lambda_feat`, `lambda_ewc`: pesos de las losses

## Fases de Ejecución

1. **FASE 1**: Carga RadioML + WiSig, prepara tareas incrementales
2. **FASE 2**: Entrenamiento estático (upper bound de rendimiento)
3. **FASE 3**: 4 estrategias incrementales (sin replay → completo)
4. **FASE 4**: Comparación y métricas CL estándar
5. **FASE 5**: Adaptación a WiSig con pseudo-labels (Tarea 5)
6. **FASE 6**: Evaluación cuantitativa cross-domain (M2M4, degradación)
