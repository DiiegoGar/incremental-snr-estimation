"""
main_pipeline.py — Pipeline maestro del TFG mejorado
=====================================================
Ejecutar paso a paso (copiar cada sección como celda de notebook)
o directamente: python main_pipeline.py

Estructura:
  FASE 1: Carga de datos + arquitectura mejorada
  FASE 2: Entrenamiento estático (baseline vs ResNet)
  FASE 3: Aprendizaje incremental con 3 estrategias
  FASE 4: Comparación de estrategias CL
  FASE 5: Adaptación cross-domain a WiSig (Tarea 5)
  FASE 6: Evaluación cuantitativa en WiSig
"""

import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Módulos del proyecto
from model import SNREstimatorCNN, ResNetSNR, model_summary
from data_utils import (
    load_radioml, build_task_data, load_wisig_manyrx,
    generate_wisig_pseudo_labels, split_wisig_adapt_eval,
    IQDataset, make_loaders,
)
from incremental_engine import (
    ReplayBuffer, EWC,
    train_one_epoch, evaluate, train_task_incremental,
    run_incremental_pipeline, print_cl_metrics, compute_forgetting,
    compute_average_accuracy, compute_backward_transfer,
    run_multiseed_strategies,
    compute_random_init_baselines, compute_fwt,
    run_task_order_ablation,
    run_hyperparam_ablation, export_mae_matrix_csv,
)
from evaluation import (
    predict_unlabeled, degradation_test, print_degradation_results,
    analyze_by_metadata, compare_with_m2m4, calibrate_linear, KAPPA_S,
    mae_per_snr, print_mae_per_snr_table, feature_space_analysis,
    cross_domain_validity_report,
)

# ============================================================
# CONFIGURACIÓN
# ============================================================
CONFIG = {
    # Rutas (ajustar según tu estructura). En Linux/macOS el nombre del
    # archivo es case-sensitive: el dataset oficial se distribuye como
    # ManyRx.pkl (x minúscula), no ManyRX.
    "radioml_path": "data/RML2016.10a_dict.pkl",
    "wisig_path": "data/ManyRx.pkl",

    # Entrenamiento
    "seed": 42,
    "batch_size": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Reproducibilidad estricta. Activa cudnn.deterministic y propaga la
    # seed a los DataLoaders mediante un generator dedicado. Coste pequeño
    # en velocidad pero comparativas multi-seed/ablation se vuelven fiables.
    "deterministic": True,

    # Entrenamiento estático
    "static_epochs": 40,
    "static_lr": 3e-4,

    # Incremental
    "incremental_epochs": 20,
    "incremental_lr": 3e-4,
    "buffer_capacity": 10000,
    "lambda_kd": 0.5,
    "lambda_feat": 0.3,
    "lambda_ewc": 10.0,

    # WiSig — Tarea 5 (cross-domain)
    # Rango ampliado: cubre SNR real de WiFi (≥20 dB) y evita techo a +15 dB.
    "wisig_pseudo_snr_levels": [-15, -10, -5, 0, 5, 10, 15, 20, 25],
    "wisig_pseudo_n_per_level": 3000,
    "wisig_adaptation_epochs": 15,

    # Multi-head: head_radio frozen + head_wisig nuevo (~33k params).
    # Esto elimina estructuralmente el olvido cross-domain: RadioML usa el
    # head original intacto y WiSig usa un head propio. NO se mezclan los
    # mapeos finales. EWC, KD y replay ya no son necesarios en T5.
    "t5_use_multihead": True,
    # Si True, también input_norm libre (más capacidad WiSig pero altera
    # ligeramente las features que ve head_radio en T1-T4). Mantener False
    # garantiza olvido = 0 por construcción.
    "t5_train_input_norm": False,
    "t5_lr": 1e-3,  # con head pequeño y aleatorio se puede usar LR mayor

    # ── FASE A: Evaluación multi-seed (reproducibilidad estadística) ──────────
    "run_multiseed":        True,           # 3 seeds × 4 estrategias
    "multiseed_seeds":      [42, 123, 2024],

    # ── FASE B: Ablación de componentes (qué aporta cada módulo CL) ───────────
    "run_component_ablation": True,         # 8 configs: aislamiento de cada componente

    # ── FASE D: Sensibilidad al orden de tareas ───────────────────────────────
    "run_task_order_ablation": True,        # 3 órdenes distintos

    # ── FASE E: Forward Transfer (FWT) ───────────────────────────────────────
    "run_fwt":              True,           # Forward Transfer: mide si tareas previas ayudan a las nuevas

    # ── FASE F: Ablación GroupNorm vs BatchNorm ───────────────────────────────
    "run_norm_ablation":    True,

    # ── FASE D.3: Análisis del espacio de features ────────────────────────────
    "run_feature_analysis": True,

    # ── FASE G: WiSig — descongelar layer4 además de head_wisig ──────────────
    "t5_unfreeze_layer4":   False,          # True = más capacidad, riesgo de olvido features

    # ── Legacy (desactivadas; reemplazadas por run_component_ablation) ────────
    "run_lambda_ablation":  False,
    "run_buffer_ablation":  False,

    # Exportaciones estructuradas
    "export_csv":     True,
    "export_run_log": True,
    "output_dir":     "outputs",

    # Modo iteración rápida. Cuando es True, salta las ablations costosas y
    # multi-seed: ejecuta solo Fase 1-6 con una seed. Pasa de ~6 h a ~45 min.
    # Útil mientras se afinan hiperparámetros o se depura el pipeline; activar
    # solo False cuando se vayan a generar las tablas finales del TFG.
    "quick_run": False,
}

# Aplicación del modo quick_run: sobrescribe los flags caros.
if CONFIG.get("quick_run", False):
    CONFIG["run_multiseed"] = False
    CONFIG["run_component_ablation"] = False
    CONFIG["run_task_order_ablation"] = False
    CONFIG["run_fwt"] = False
    CONFIG["run_norm_ablation"] = False
    CONFIG["run_feature_analysis"] = False
    print("[quick_run] activado: ablations multi-seed/component/order/fwt/norm desactivadas")

device = torch.device(CONFIG["device"])
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Reproducibilidad estricta (6.4.1). Debe ejecutarse ANTES de que se cree
# cualquier modelo o DataLoader; afecta a las semillas de torch/numpy/random
# y, en CUDA, a la selección de kernels deterministas.
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])
if CONFIG.get("deterministic", True):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# Directorio de salidas + log estructurado (serializable a JSON al final).
os.makedirs(CONFIG["output_dir"], exist_ok=True)
run_log = {
    "config": {k: (str(v) if isinstance(v, torch.device) else v)
               for k, v in CONFIG.items()},
    "phases": {},
    "timings": {},  # tiempos por fase (segundos)
}
_pipeline_t0 = time.perf_counter()
_phase_starts = {}  # nombre → t0; pares con _end_phase para registrar elapsed

def _start_phase(name):
    _phase_starts[name] = time.perf_counter()

def _end_phase(name):
    if name in _phase_starts:
        run_log["timings"][name] = time.perf_counter() - _phase_starts.pop(name)
        print(f"[timing] {name}: {run_log['timings'][name]:.1f} s")


# ################################################################
# FASE 1: CARGA DE DATOS
# ################################################################
print("\n" + "█" * 60)
print("  FASE 1: CARGA DE DATOS Y PREPARACIÓN")
print("█" * 60)

# 1.1 Cargar RadioML
data_dict, mods, snrs, mod_to_idx = load_radioml(CONFIG["radioml_path"], seed=CONFIG["seed"])

# 1.2 Construir tareas incrementales SIN normalización externa
# (ResNetSNR usa InstanceNorm interna → datos crudos)
print("\n--- Preparando tareas para ResNetSNR (sin normalización externa) ---")
task_data_raw, _ = build_task_data(data_dict, mod_to_idx, normalize="none")

# 1.3 También preparar con z-score para el baseline CNN (comparación)
print("\n--- Preparando tareas para CNN Baseline (z-score) ---")
task_data_zscore, zscore_stats = build_task_data(data_dict, mod_to_idx, normalize="zscore")

# 1.4 Verificar arquitecturas
print("\n--- Verificación de arquitecturas ---")
print("\nBASELINE:")
model_summary(SNREstimatorCNN())
print("\nRESNET-SNR (MEJORADO):")
model_summary(ResNetSNR())


# ################################################################
# FASE 2: ENTRENAMIENTO ESTÁTICO (TECHO DE RENDIMIENTO)
# ################################################################
print("\n" + "█" * 60)
print("  FASE 2: ENTRENAMIENTO ESTÁTICO (UPPER BOUND)")
print("█" * 60)

def train_static(model, X_train, y_train, X_val, y_val, epochs, lr, device):
    """Entrena el modelo sobre TODOS los datos simultáneamente."""
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val,
                                            batch_size=CONFIG["batch_size"])
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_mae = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} dB | Val RMSE: {val_rmse:.4f} dB"
                  + (" ★" if val_mae == best_mae else ""))

    model.load_state_dict(best_state)
    print(f"\n  MEJOR MAE ESTÁTICO: {best_mae:.4f} dB")
    return model, best_mae


# 2.1 Entrenar CNN Baseline estático (con z-score)
print("\n--- 2.1 CNN Baseline estático ---")
# Combinar todos los datos de todas las tareas
X_train_all_zs = np.concatenate([t["X_train"] for t in task_data_zscore])
y_train_all_zs = np.concatenate([t["y_train"] for t in task_data_zscore])
X_val_all_zs = np.concatenate([t["X_val"] for t in task_data_zscore])
y_val_all_zs = np.concatenate([t["y_val"] for t in task_data_zscore])

model_baseline_static = SNREstimatorCNN().to(device)
model_baseline_static, baseline_static_mae = train_static(
    model_baseline_static, X_train_all_zs, y_train_all_zs,
    X_val_all_zs, y_val_all_zs,
    epochs=CONFIG["static_epochs"], lr=CONFIG["static_lr"], device=device
)

# 2.2 Entrenar ResNetSNR estático (sin normalización externa)
print("\n--- 2.2 ResNetSNR estático ---")
X_train_all_raw = np.concatenate([t["X_train"] for t in task_data_raw])
y_train_all_raw = np.concatenate([t["y_train"] for t in task_data_raw])
X_val_all_raw = np.concatenate([t["X_val"] for t in task_data_raw])
y_val_all_raw = np.concatenate([t["y_val"] for t in task_data_raw])

model_resnet_static = ResNetSNR().to(device)
model_resnet_static, resnet_static_mae = train_static(
    model_resnet_static, X_train_all_raw, y_train_all_raw,
    X_val_all_raw, y_val_all_raw,
    epochs=CONFIG["static_epochs"], lr=CONFIG["static_lr"], device=device
)

print(f"\n{'='*50}")
print(f"  COMPARACIÓN ESTÁTICA")
print(f"{'='*50}")
print(f"  CNN Baseline (z-score):   {baseline_static_mae:.4f} dB")
print(f"  ResNetSNR (InstanceNorm): {resnet_static_mae:.4f} dB")
print(f"  Mejora: {baseline_static_mae - resnet_static_mae:.4f} dB")

bn_static_mae = None  # se sobreescribe si run_norm_ablation=True

# ── FASE F: Ablación GroupNorm vs BatchNorm ───────────────────────────────────
# Hipótesis: ¿BatchNorm contamina running stats entre tareas CL?
# GroupNorm es el default; BatchNorm es la comparación.
# SOLO entrenamiento estático — el impacto en CL se evalúa en run_component_ablation.
if CONFIG.get("run_norm_ablation", True):
    print(f"\n--- 2.3 Ablación FASE F: GroupNorm vs BatchNorm (entrenamiento estático) ---")
    model_resnet_bn = ResNetSNR(norm="batch").to(device)
    _, bn_static_mae = train_static(
        model_resnet_bn, X_train_all_raw, y_train_all_raw,
        X_val_all_raw, y_val_all_raw,
        epochs=CONFIG["static_epochs"], lr=CONFIG["static_lr"], device=device,
    )
    print(f"\n  Ablación normalización en bloques residuales (entrenamiento estático):")
    print(f"    GroupNorm (CL-friendly, default): {resnet_static_mae:.4f} dB MAE")
    print(f"    BatchNorm (acumula running stats): {bn_static_mae:.4f} dB MAE")
    diff_norm = resnet_static_mae - bn_static_mae
    winner = "GroupNorm" if diff_norm < 0 else "BatchNorm"
    print(f"    Diferencia GN−BN: {diff_norm:+.4f} dB  → {winner} gana en estático")
    print(f"    Nota: GroupNorm evita contaminación de running stats entre tareas CL,")
    print(f"          lo que se traduce en mejor forgetting aunque el MAE estático sea similar.")
    run_log["phases"]["norm_ablation"] = {
        "groupnorm_mae": float(resnet_static_mae),
        "batchnorm_mae": float(bn_static_mae),
        "diff_gn_minus_bn": float(diff_norm),
    }

# 2.3 Joint training upper bound — evaluación por tarea
# El modelo estático de ResNetSNR actúa como upper bound conceptual: ha visto
# todas las tareas simultáneamente, así que su MAE por tarea es el mínimo
# alcanzable por esta arquitectura. Comparar contra mae_matrix[N][tid] de CL
# cuantifica el coste de aprender incrementalmente.
print(f"\n{'='*50}")
print(f"  JOINT TRAINING — MAE por tarea (upper bound)")
print(f"{'='*50}")
joint_mae_per_task = {}
criterion_joint = nn.SmoothL1Loss()
for t in task_data_raw:
    tid = t["task_id"]
    test_ds = IQDataset(t["X_test"], t["y_test"])
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    _, mae_j, rmse_j, _, _ = evaluate(model_resnet_static, test_loader, criterion_joint, device)
    joint_mae_per_task[tid] = mae_j
    print(f"  T{tid} ({t['mods']}): MAE={mae_j:.4f} dB | RMSE={rmse_j:.4f} dB")
joint_aa = float(np.mean(list(joint_mae_per_task.values())))
print(f"  Joint AA (media): {joint_aa:.4f} dB")


# ################################################################
# FASE 3: APRENDIZAJE INCREMENTAL — 4 ESTRATEGIAS
# ################################################################
print("\n" + "█" * 60)
print("  FASE 3: APRENDIZAJE INCREMENTAL (4 ESTRATEGIAS)")
print("█" * 60)
_start_phase("phase3_incremental")

# Estrategia 1: Naive fine-tuning (lower bound — muestra olvido catastrófico)
# Sin replay, sin KD, sin EWC: referencia mínima frente a la que cualquier
# método CL serio debe mejorar. Si una estrategia no supera a Naive,
# no aporta nada sobre la alternativa trivial.
print("\n" + "=" * 60)
print("  ESTRATEGIA 1: NAIVE FINE-TUNING (lower bound — olvido catastrófico)")
print("=" * 60)
model_norep, mae_matrix_norep, _ = run_incremental_pipeline(
    ResNetSNR, task_data_raw, device,
    buffer_capacity=0, lambda_kd=0, lambda_feat=0,
    use_ewc=False, use_herding=False,
    epochs=CONFIG["incremental_epochs"], lr=CONFIG["incremental_lr"]
)

# Estrategia 2: Solo Replay (como el notebook original, pero con ResNet)
print("\n" + "=" * 60)
print("  ESTRATEGIA 2: REPLAY BUFFER (RANDOM)")
print("=" * 60)
model_replay, mae_matrix_replay, _ = run_incremental_pipeline(
    ResNetSNR, task_data_raw, device,
    buffer_capacity=CONFIG["buffer_capacity"], lambda_kd=0, lambda_feat=0,
    use_ewc=False, use_herding=False,
    epochs=CONFIG["incremental_epochs"], lr=CONFIG["incremental_lr"]
)

# Estrategia 3: Replay + Output-KD (similar al original, sin feature-KD ni EWC)
print("\n" + "=" * 60)
print("  ESTRATEGIA 3: REPLAY + OUTPUT-KD")
print("=" * 60)
model_replay_kd, mae_matrix_replay_kd, _ = run_incremental_pipeline(
    ResNetSNR, task_data_raw, device,
    buffer_capacity=CONFIG["buffer_capacity"],
    lambda_kd=CONFIG["lambda_kd"], lambda_feat=0,
    use_ewc=False, use_herding=False,
    epochs=CONFIG["incremental_epochs"], lr=CONFIG["incremental_lr"]
)

# Estrategia 4: COMPLETA — Herding + Feature-KD + EWC (TODAS LAS MEJORAS)
print("\n" + "=" * 60)
print("  ESTRATEGIA 4: HERDING + FEATURE-KD + EWC (COMPLETO)")
print("=" * 60)
model_best, mae_matrix_full, results_full = run_incremental_pipeline(
    ResNetSNR, task_data_raw, device,
    buffer_capacity=CONFIG["buffer_capacity"],
    lambda_kd=CONFIG["lambda_kd"], lambda_feat=CONFIG["lambda_feat"],
    lambda_ewc=CONFIG["lambda_ewc"],
    use_ewc=True, use_herding=True,
    epochs=CONFIG["incremental_epochs"], lr=CONFIG["incremental_lr"]
)
_end_phase("phase3_incremental")


# ################################################################
# FASE 4: COMPARACIÓN DE ESTRATEGIAS
# ################################################################
print("\n" + "█" * 60)
print("  FASE 4: COMPARACIÓN DE ESTRATEGIAS CL")
print("█" * 60)

num_tasks = len(task_data_raw)

# Tabla comparativa — 4 estrategias CL + Joint upper bound
print(f"\n{'='*82}")
print(f"  COMPARACIÓN FINAL DE MAE (dB) TRAS LA ÚLTIMA TAREA")
print(f"{'='*82}")
print(f"  {'Tarea':<8} {'Naive':<12} {'Replay':<12} {'R+KD':<12} "
      f"{'R+FKD+EWC':<12} {'Joint (UB)':<12}")
print(f"  {'-'*72}")
for tid in range(1, num_tasks + 1):
    norep = mae_matrix_norep[num_tasks].get(tid, float('nan'))
    rep   = mae_matrix_replay[num_tasks].get(tid, float('nan'))
    rkd   = mae_matrix_replay_kd[num_tasks].get(tid, float('nan'))
    full  = mae_matrix_full[num_tasks].get(tid, float('nan'))
    jt    = joint_mae_per_task.get(tid, float('nan'))
    print(f"  T{tid:<6} {norep:>7.4f}     {rep:>7.4f}     {rkd:>7.4f}     "
          f"{full:>7.4f}     {jt:>7.4f}")

# Calcular AA (Average Accuracy) para comparar entre estrategias
aa_norep = float(np.mean([mae_matrix_norep[num_tasks][t] for t in range(1, num_tasks+1)]))
aa_rep   = float(np.mean([mae_matrix_replay[num_tasks][t] for t in range(1, num_tasks+1)]))
aa_rkd   = float(np.mean([mae_matrix_replay_kd[num_tasks][t] for t in range(1, num_tasks+1)]))
aa_full  = float(np.mean([mae_matrix_full[num_tasks][t] for t in range(1, num_tasks+1)]))
print(f"  {'-'*72}")
print(f"  {'AA':<8} {aa_norep:>7.4f}     {aa_rep:>7.4f}     {aa_rkd:>7.4f}     "
      f"{aa_full:>7.4f}     {joint_aa:>7.4f}")
print(f"\n  Gap al upper bound (H+FKD+EWC − Joint): {aa_full - joint_aa:+.4f} dB")
print(f"  Gap del lower bound (Naive − Joint):     {aa_norep - joint_aa:+.4f} dB")

# Métricas CL para cada estrategia
metrics_norep = print_cl_metrics(mae_matrix_norep, num_tasks, "NAIVE (lower bound)")
metrics_rep   = print_cl_metrics(mae_matrix_replay, num_tasks, "REPLAY")
metrics_rkd   = print_cl_metrics(mae_matrix_replay_kd, num_tasks, "REPLAY + KD")
metrics_full  = print_cl_metrics(mae_matrix_full, num_tasks, "HERDING + FKD + EWC")

# Resumen
print(f"\n{'='*60}")
print(f"  RESUMEN DE FORGETTING")
print(f"{'='*60}")
print(f"  Sin replay:           {metrics_norep['mean_forgetting']:.4f} dB")
print(f"  Replay random:        {metrics_rep['mean_forgetting']:.4f} dB")
print(f"  Replay + Output-KD:   {metrics_rkd['mean_forgetting']:.4f} dB")
print(f"  Herding + FKD + EWC:  {metrics_full['mean_forgetting']:.4f} dB")
print(f"  Estático (techo):     0.0000 dB")
print(f"\n  Reducción olvido (Sin replay → Completo): "
      f"{(1 - metrics_full['mean_forgetting']/max(metrics_norep['mean_forgetting'], 1e-6))*100:.1f}%")

# Consolidar resultados de Fase 4 en run_log
matrices_cl = {
    "naive":     mae_matrix_norep,
    "replay":    mae_matrix_replay,
    "r_kd":      mae_matrix_replay_kd,
    "h_fkd_ewc": mae_matrix_full,
}
run_log["phases"]["phase4_cl"] = {
    "joint_aa":          joint_aa,
    "joint_mae_per_task": joint_mae_per_task,
    "AA": {"naive": aa_norep, "replay": aa_rep,
           "r_kd": aa_rkd, "h_fkd_ewc": aa_full},
    "mean_forgetting": {
        "naive":     metrics_norep["mean_forgetting"],
        "replay":    metrics_rep["mean_forgetting"],
        "r_kd":      metrics_rkd["mean_forgetting"],
        "h_fkd_ewc": metrics_full["mean_forgetting"],
    },
    "mae_matrix": {k: {str(i): v for i, v in m.items()}
                   for k, m in matrices_cl.items()},
}

# Export matriz R_ij completa a CSV (formato largo: strategy,after,eval,MAE).
if CONFIG["export_csv"]:
    csv_path = os.path.join(CONFIG["output_dir"], "mae_matrix_rij.csv")
    export_mae_matrix_csv(matrices_cl, num_tasks, csv_path)
    print(f"\n  Matriz R_ij exportada: {csv_path}")


# ################################################################
# FASE C: ANÁLISIS MAE POR NIVEL DE SNR
# Convierte el MAE escalar en una curva MAE vs SNR publicable.
# Revela en qué rango de SNR difieren las estrategias CL.
# ################################################################
print("\n" + "█" * 60)
print("  FASE C: MAE POR NIVEL DE SNR — 4 ESTRATEGIAS + JOINT")
print("█" * 60)

criterion_snr = nn.SmoothL1Loss()
_strategy_models = {
    "Naive":       model_norep,
    "Replay":      model_replay,
    "Replay+KD":   model_replay_kd,
    "H+FKD+EWC":   model_best,
    "Joint(UB)":   model_resnet_static,
}
_snr_results = {}
for _name, _mdl in _strategy_models.items():
    _all_p, _all_t = [], []
    for _task in task_data_raw:
        _ds = IQDataset(_task["X_test"], _task["y_test"])
        _ld = DataLoader(_ds, batch_size=256, shuffle=False)
        _, _, _, _p, _t = evaluate(_mdl, _ld, criterion_snr, device)
        _all_p.extend(_p.tolist()); _all_t.extend(_t.tolist())
    _bins, _maes, _ = mae_per_snr(np.array(_all_p), np.array(_all_t))
    _snr_results[_name] = (_bins, _maes)

print_mae_per_snr_table(_snr_results, label="todas las modulaciones")

# Guardar también CSV por SNR
if CONFIG["export_csv"]:
    import csv as _csv
    snr_csv = os.path.join(CONFIG["output_dir"], "mae_per_snr.csv")
    with open(snr_csv, "w", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["strategy"] + [int(s) for s in _snr_results[list(_snr_results.keys())[0]][0]])
        for _n, (_b, _m) in _snr_results.items():
            _w.writerow([_n] + [f"{v:.4f}" for v in _m])
    print(f"\n  MAE por SNR exportado: {snr_csv}")

run_log["phases"]["phase_c_snr"] = {
    nm: {"snr_bins": bins.tolist(), "mae_vals": maes.tolist()}
    for nm, (bins, maes) in _snr_results.items()
}


# ################################################################
# FASE D: ANÁLISIS DE VULNERABILIDAD DE T3 + ESPACIO DE FEATURES
# ################################################################
print("\n" + "█" * 60)
print("  FASE D: ANÁLISIS VULNERABILIDAD T3 + ESPACIO DE FEATURES")
print("█" * 60)

# D.1 — Matriz R_ij completa de H+FKD+EWC: evolución del MAE por tarea
print("\n--- D.1 Matriz R_ij completa (H+FKD+EWC) — evolución tras cada tarea ---")
print(f"  {'Tras↓ / Eval→':<14}", end="")
for ev in range(1, num_tasks + 1):
    print(f"  T{ev}({task_data_raw[ev-1]['mods'][0][:6]})", end="")
print()
print("  " + "-" * (16 + num_tasks * 14))
for after in sorted(mae_matrix_full.keys()):
    print(f"  T{after:<12}", end="")
    for ev in range(1, num_tasks + 1):
        val = mae_matrix_full[after].get(ev, float("nan"))
        if not np.isnan(val):
            print(f"  {val:>10.4f}", end="")
        else:
            print(f"  {'---':>10}", end="")
    print()

# D.2 — Forgetting acumulado: T3 vs resto
print("\n--- D.2 Forgetting acumulado por tarea y comparativa ---")
_fg_full = compute_forgetting(mae_matrix_full, num_tasks)
_fg_others = [f for t, f in _fg_full.items() if t != 3]
print(f"  {'Tarea':<8} {'Forgetting (dB)':>15}  {'vs. media_otros':>16}")
print("  " + "-" * 42)
for _tid, _fv in sorted(_fg_full.items()):
    _ref = np.mean(_fg_others) if _fg_others else float("nan")
    _ratio = _fv / (_ref + 1e-8)
    print(f"  T{_tid:<6} {_fv:>15.4f}  {_ratio:>14.2f}x")
if _fg_others:
    print(f"\n  T3 olvida {_fg_full.get(3, 0) / (np.mean(_fg_others) + 1e-8):.2f}× "
          f"más que la media del resto de tareas.")

# D.3 — Espacio de features
print("\n--- D.3 Análisis del espacio de features (modelo H+FKD+EWC) ---")
if CONFIG.get("run_feature_analysis", True):
    _centroids, _feat_info = feature_space_analysis(
        model_best, task_data_raw, device, n_samples=500
    )
    run_log["phases"]["feature_analysis"] = {
        "within_vars": {str(k): float(v)
                        for k, v in _feat_info["within_vars"].items()},
        "between_dists": {f"T{i}-T{j}": float(d)
                          for (i, j), d in _feat_info["between_dists"].items()},
    }


# ################################################################
# FASE E: FORWARD TRANSFER (FWT) — opcional, activar con run_fwt=True
# ################################################################
if CONFIG.get("run_fwt", False):
    print("\n" + "█" * 60)
    print("  FASE E: FORWARD TRANSFER (FWT)")
    print("█" * 60)
    print("  FWT_i = MAE_random_init(T_i) − MAE_CL_diagonal(T_i)")
    print("  Positivo → transferencia positiva; negativo → interferencia.")

    _rand_maes = compute_random_init_baselines(
        ResNetSNR, task_data_raw, device,
        epochs=CONFIG["incremental_epochs"],
        lr=CONFIG["incremental_lr"],
        batch_size=CONFIG["batch_size"],
    )
    print(f"\n--- FWT para H+FKD+EWC ---")
    _fwt_per, _fwt_mean = compute_fwt(mae_matrix_full, _rand_maes, num_tasks)
    for _tid, _fwt_i in _fwt_per.items():
        _dir = "transferencia(+)" if _fwt_i > 0 else "interferencia(-)"
        print(f"  T{_tid}: {_fwt_i:+.4f} dB  ({_dir})")
    print(f"  FWT medio: {_fwt_mean:+.4f} dB")

    # FWT también para Replay random (comparación)
    _fwt_rep, _fwt_rep_mean = compute_fwt(mae_matrix_replay, _rand_maes, num_tasks)
    print(f"\n--- FWT para Replay random ---")
    for _tid, _fwt_i in _fwt_rep.items():
        _dir = "transferencia(+)" if _fwt_i > 0 else "interferencia(-)"
        print(f"  T{_tid}: {_fwt_i:+.4f} dB  ({_dir})")
    print(f"  FWT medio: {_fwt_rep_mean:+.4f} dB")

    run_log["phases"]["fwt"] = {
        "h_fkd_ewc": {"per_task": {str(k): v for k, v in _fwt_per.items()},
                       "mean": _fwt_mean},
        "replay":    {"per_task": {str(k): v for k, v in _fwt_rep.items()},
                       "mean": _fwt_rep_mean},
    }


# ################################################################
# ABLACIÓN DE COMPONENTES — FASE B del plan
# 8 configuraciones que aíslan la contribución de cada módulo CL.
# Esto explica POR QUÉ los resultados son los que son.
# ################################################################
if CONFIG.get("run_component_ablation", True):
    print("\n" + "█" * 60)
    print("  FASE B: ABLACIÓN DE COMPONENTES CL")
    print("█" * 60)
    print("  Aísla la contribución de: Herding, Output-KD, Feature-KD, EWC.")
    _start_phase("component_ablation")

    _comp_names = [
        "1. Replay random (sin KD)",
        "2. Replay + Output-KD",
        "3. Replay + Feature-KD",
        "4. R + OutKD + FeatKD",
        "5. Herding + KD + FKD (sin EWC)",
        "6. H+KD+FKD + EWC(λ=1)",
        "7. H+KD+FKD + EWC(λ=10)  ←actual",
        "8. H+KD+FKD + EWC(λ=100)",
    ]
    _comp_configs = [
        # 1. Solo replay random
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0, "lambda_feat": 0, "lambda_ewc": 0,
         "use_ewc": False, "use_herding": False},
        # 2. Replay + Output-KD
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0.5, "lambda_feat": 0, "lambda_ewc": 0,
         "use_ewc": False, "use_herding": False},
        # 3. Replay + Feature-KD (sin output-KD)
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0, "lambda_feat": 0.3, "lambda_ewc": 0,
         "use_ewc": False, "use_herding": False},
        # 4. Replay + Output-KD + Feature-KD
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0.5, "lambda_feat": 0.3, "lambda_ewc": 0,
         "use_ewc": False, "use_herding": False},
        # 5. Herding + KD + FKD (sin EWC) — aísla contribución del herding
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0.5, "lambda_feat": 0.3, "lambda_ewc": 0,
         "use_ewc": False, "use_herding": True},
        # 6. H+KD+FKD + EWC(λ=1)
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0.5, "lambda_feat": 0.3, "lambda_ewc": 1.0,
         "use_ewc": True, "use_herding": True},
        # 7. H+KD+FKD + EWC(λ=10) — configuración actual del TFG
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0.5, "lambda_feat": 0.3, "lambda_ewc": 10.0,
         "use_ewc": True, "use_herding": True},
        # 8. H+KD+FKD + EWC(λ=100)
        {"buffer_capacity": CONFIG["buffer_capacity"],
         "lambda_kd": 0.5, "lambda_feat": 0.3, "lambda_ewc": 100.0,
         "use_ewc": True, "use_herding": True},
    ]
    _comp_results = run_hyperparam_ablation(
        ResNetSNR, task_data_raw, device,
        configs=_comp_configs,
        names=_comp_names,
        seed=CONFIG["seed"],
        defaults={  # sin defaults propios: cada config es explícita
            "buffer_capacity": CONFIG["buffer_capacity"],
            "lambda_kd": 0, "lambda_feat": 0, "lambda_ewc": 0,
            "use_ewc": False, "use_herding": False,
            "epochs": CONFIG["incremental_epochs"],
            "lr": CONFIG["incremental_lr"],
            "batch_size": CONFIG["batch_size"],
        },
        label="ABLACIÓN COMPONENTES CL",
    )
    # Determinar qué añade cada componente
    print(f"\n  Contribución incremental de cada componente (vs. config anterior):")
    _prev_aa = None
    for _r in _comp_results:
        if _prev_aa is not None:
            _delta = _r["AA"] - _prev_aa
            _sign = "↑ empeora" if _delta > 0.005 else ("↓ mejora" if _delta < -0.005 else "≈ neutral")
            print(f"    {_r['name'][:45]}: ΔAA={_delta:+.4f} dB  {_sign}")
        _prev_aa = _r["AA"]

    run_log["phases"]["component_ablation"] = [
        {"name": r["name"], "AA": r["AA"], "BWT": r["BWT"],
         "mean_forgetting": r["mean_forgetting"]}
        for r in _comp_results
    ]
    _end_phase("component_ablation")


# ################################################################
# ABLACIÓN DE HIPERPARÁMETROS LEGACY (opcional, desactivado por defecto)
# ################################################################
if CONFIG["run_lambda_ablation"]:
    print("\n" + "█" * 60)
    print("  ABLACIÓN: lambda_ewc × lambda_kd")
    print("█" * 60)
    lambda_configs = [
        # Barrido en la estrategia completa (herding + feat-KD + EWC).
        {"lambda_ewc": 0,    "use_ewc": False},
        {"lambda_ewc": 1},
        {"lambda_ewc": 10},
        {"lambda_ewc": 100},
        {"lambda_kd": 0, "lambda_feat": 0},
        {"lambda_kd": 1.0, "lambda_feat": 0.5},
    ]
    lambda_results = run_hyperparam_ablation(
        ResNetSNR, task_data_raw, device, lambda_configs,
        seed=CONFIG["seed"],
        label="LAMBDAS (ewc × kd)",
    )
    run_log["phases"]["ablation_lambdas"] = [
        {"config": r["config"], "AA": r["AA"], "BWT": r["BWT"],
         "mean_forgetting": r["mean_forgetting"]}
        for r in lambda_results
    ]

if CONFIG["run_buffer_ablation"]:
    print("\n" + "█" * 60)
    print("  ABLACIÓN: tamaño del replay buffer")
    print("█" * 60)
    buffer_configs = [
        {"buffer_capacity": 0, "lambda_kd": 0, "lambda_feat": 0,
         "use_ewc": False, "use_herding": False},
        {"buffer_capacity": 1000},
        {"buffer_capacity": 5000},
        {"buffer_capacity": 10000},
        {"buffer_capacity": 20000},
    ]
    buffer_results = run_hyperparam_ablation(
        ResNetSNR, task_data_raw, device, buffer_configs,
        seed=CONFIG["seed"],
        label="BUFFER CAPACITY",
    )
    run_log["phases"]["ablation_buffer"] = [
        {"config": r["config"], "AA": r["AA"], "BWT": r["BWT"],
         "mean_forgetting": r["mean_forgetting"]}
        for r in buffer_results
    ]


# ################################################################
# FASE 5: ADAPTACIÓN CROSS-DOMAIN A WiSig (TAREA 5)
# ################################################################
print("\n" + "█" * 60)
print("  FASE 5: ADAPTACIÓN CROSS-DOMAIN — WiSig (TAREA 5)")
print("█" * 60)

# 5.1 Cargar WiSig
print("\n--- 5.1 Cargando WiSig ManyRX ---")
if os.path.exists(CONFIG["wisig_path"]):
    X_wisig, meta_wisig, wisig_raw = load_wisig_manyrx(CONFIG["wisig_path"])

    # 5.2 Partir WiSig en adapt (T5) y eval (held-out, nunca visto)
    # Este split elimina el bucle circular: las muestras usadas para
    # generar pseudo-labels (X_adapt) son estrictamente distintas de las
    # usadas para evaluar el modelo (X_eval). Así, las métricas físicas
    # del test de degradación y la comparación M2M4 no están contaminadas
    # por el entrenamiento de T5.
    print("\n--- 5.2 División WiSig adapt / eval (anti-bucle-circular) ---")
    X_adapt, meta_adapt, X_eval, meta_eval = split_wisig_adapt_eval(
        X_wisig, meta_wisig, adapt_frac=0.5, seed=CONFIG["seed"]
    )

    # 5.3 Predicción zero-shot sobre X_eval (held-out)
    print("\n--- 5.3 Predicción zero-shot sobre held-out WiSig ---")
    preds_zeroshot = predict_unlabeled(model_best, X_eval, device)
    print(f"  Zero-shot (held-out): Media={preds_zeroshot.mean():.2f} dB  "
          f"Std={preds_zeroshot.std():.2f} dB")

    # 5.4 Generar pseudo-labels SOLO sobre X_adapt
    print("\n--- 5.4 Generando pseudo-labels para WiSig (solo X_adapt) ---")
    X_pseudo, y_pseudo = generate_wisig_pseudo_labels(
        X_adapt, snr_levels=CONFIG["wisig_pseudo_snr_levels"],
        n_per_level=CONFIG["wisig_pseudo_n_per_level"],
    )

    # 5.5 Crear Tarea 5 y adaptar con multi-head
    print("\n--- 5.5 Adaptación incremental (Tarea 5) — Multi-head ---")
    # Split 70/15/15 — consistente con load_radioml().
    # X_val → early stopping; X_test → evaluación final nunca vista durante entrenamiento.
    perm = np.random.RandomState(42).permutation(len(X_pseudo))
    X_pseudo = X_pseudo[perm]
    y_pseudo = y_pseudo[perm]
    n = len(X_pseudo)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    task5 = {
        "task_id": 5,
        "mods": ["WiFi-WiSig"],
        "X_train": X_pseudo[:n_train],
        "y_train": y_pseudo[:n_train],
        "X_val":   X_pseudo[n_train:n_train + n_val],
        "y_val":   y_pseudo[n_train:n_train + n_val],
        "X_test":  X_pseudo[n_train + n_val:],
        "y_test":  y_pseudo[n_train + n_val:],
    }
    print(f"  Split T5 — Train: {len(task5['X_train'])} | "
          f"Val: {len(task5['X_val'])} | Test: {len(task5['X_test'])}")

    # Multi-head: clonamos el modelo, añadimos head_wisig nuevo y dejamos
    # el regressor original (head_radio) intacto. Esto separa estructuralmente
    # los dos dominios: el mapeo final aprendido sobre RadioML no se toca.
    model_adapted = copy.deepcopy(model_best)
    model_adapted.add_head("wisig")
    model_adapted.set_active_head("wisig")
    model_adapted.to(device)

    # Freezing: el extractor + head_radio quedan congelados por defecto.
    # Solo entrenamos heads.wisig (~33k params). Opciones adicionales:
    #   t5_unfreeze_layer4=True → descongelar también layer4 (más capacidad WiSig,
    #     pero no rompe head_radio pues el multi-head separa los mapeos finales)
    #   t5_train_input_norm=True → afecta ligeramente features de head_radio
    trainable_keys = ["heads.wisig"]
    if CONFIG.get("t5_unfreeze_layer4", False):
        trainable_keys.append("layer4")
        print("  ⚑ FASE G: layer4 desbloqueada (más capacidad WiSig)")
    if CONFIG.get("t5_train_input_norm", False):
        trainable_keys.append("input_norm")
    for name, p in model_adapted.named_parameters():
        p.requires_grad = any(k in name for k in trainable_keys)
    n_train_p = sum(p.numel() for p in model_adapted.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in model_adapted.parameters())
    print(f"  Multi-head WiSig — entrenables: {n_train_p:,}/{n_total_p:,} "
          f"({100*n_train_p/n_total_p:.1f}%)")
    trainable_modules = ", ".join(trainable_keys)
    print(f"  Módulos entrenables: {trainable_modules}")
    print(f"  Datos T5: {len(task5['X_train'])} WiSig (sin replay ni KD ni EWC)")

    # Entrenar T5: solo head nuevo, sin KD/EWC/replay (no son necesarios
    # con multi-head — el extractor y head_radio están congelados).
    model_adapted, hist_t5, best_mae_t5, best_ep_t5 = train_task_incremental(
        model_adapted, None,
        task5["X_train"], task5["y_train"],
        task5["X_val"], task5["y_val"],
        device, ewc_module=None,
        epochs=CONFIG["wisig_adaptation_epochs"],
        batch_size=CONFIG["batch_size"], lr=CONFIG["t5_lr"],
        lambda_kd=0, lambda_feat=0,
        X_replay=None, y_replay=None,
    )

    # Restaurar requires_grad para limpieza
    for p in model_adapted.parameters():
        p.requires_grad = True

    print(f"\n  Tarea 5 completada — MAE val: {best_mae_t5:.4f} dB (época {best_ep_t5})")

    # Evaluación T5 test (head wisig sigue activo)
    criterion_t5 = nn.SmoothL1Loss()
    test_ds_t5 = IQDataset(task5["X_test"], task5["y_test"])
    test_loader_t5 = DataLoader(test_ds_t5, batch_size=256, shuffle=False)
    _, mae_test_t5, rmse_test_t5, _, _ = evaluate(model_adapted, test_loader_t5, criterion_t5, device)
    print(f"  Tarea 5 — MAE test: {mae_test_t5:.4f} dB | RMSE test: {rmse_test_t5:.4f} dB")

    # 5.6 Verificar que no olvidó RadioML — usando head_radio (intacto)
    print("\n--- 5.6 Verificación anti-olvido tras Tarea 5 (head_radio) ---")
    criterion = nn.SmoothL1Loss()
    model_adapted.set_active_head(None)  # cambiar a head_radio para eval RadioML
    print(f"  {'Tarea':<10} {'Pre-T5 (dB)':<15} {'Post-T5 (dB)':<15} {'Δ (dB)':<10}")
    print(f"  {'-'*50}")
    radioml_pre_post = {}
    for tid in range(1, num_tasks + 1):
        task = task_data_raw[tid - 1]
        eval_ds = IQDataset(task["X_test"], task["y_test"])
        eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)

        _, mae_pre, _, _, _ = evaluate(model_best, eval_loader, criterion, device)
        _, mae_post, _, _, _ = evaluate(model_adapted, eval_loader, criterion, device)
        delta = mae_post - mae_pre
        radioml_pre_post[tid] = {"pre": float(mae_pre), "post": float(mae_post),
                                  "delta": float(delta)}
        print(f"  T{tid:<8} {mae_pre:>8.4f}        {mae_post:>8.4f}        {delta:>+7.4f}")
    model_adapted.set_active_head("wisig")  # restaurar head wisig para Fase 6


    # ################################################################
    # FASE 6: EVALUACIÓN CROSS-DOMAIN SIN BUCLE CIRCULAR
    # ################################################################
    print("\n" + "█" * 60)
    print("  FASE 6: VALIDEZ CROSS-DOMAIN — HELD-OUT WiSig (X_eval)")
    print("█" * 60)
    print("  X_eval: muestras NUNCA vistas durante adaptación T5.")
    print("  Métricas primarias: Spearman ρ, sensibilidad, acuerdo M2M4.")
    print("  MAE de pseudo-labels = métrica auxiliar (in-distribution, no externa).")

    # 6.1 Predicción post-adaptación sobre held-out
    print("\n--- 6.1 Predicción post-adaptación (held-out) ---")
    preds_adapted_eval = predict_unlabeled(model_adapted, X_eval, device)
    print(f"  Zero-shot (held-out):       Media={preds_zeroshot.mean():.2f} dB  "
          f"Std={preds_zeroshot.std():.2f} dB")
    print(f"  Post-adaptación (held-out): Media={preds_adapted_eval.mean():.2f} dB  "
          f"Std={preds_adapted_eval.std():.2f} dB")

    # 6.2 Reporte de validez cross-domain (todas las métricas físicas)
    # X_eval está garantizado held-out → no hay bucle circular.
    validity_report = cross_domain_validity_report(
        model_zeroshot=model_best,
        model_adapted=model_adapted,
        X_eval=X_eval,
        device=device,
        meta_eval=meta_eval,
        snr_levels=CONFIG["wisig_pseudo_snr_levels"],
        n_eval=min(10000, len(X_eval)),
        seed=CONFIG["seed"],
    )

    # 6.3 Calibración lineal usando el degradation test sobre X_eval
    # Derivamos slope/intercept a partir de los puntos SNR conocidos del
    # test de degradación (aplicado a held-out) → no hay contaminación.
    print("\n--- 6.3 Calibración lineal (degradation_test sobre held-out) ---")
    deg_cal = degradation_test(model_adapted, X_eval[:5000], device,
                               snr_levels=CONFIG["wisig_pseudo_snr_levels"])
    calib_preds, calib_truths = [], []
    for level, preds_at_lv in deg_cal["preds"].items():
        if level == "clean":
            continue
        snr_true = float(level.replace("dB", ""))
        calib_preds.extend(preds_at_lv.tolist())
        calib_truths.extend([snr_true] * len(preds_at_lv))
    calib_preds  = np.array(calib_preds)
    calib_truths = np.array(calib_truths)
    _, slope_cal, intercept_cal = calibrate_linear(
        calib_preds, calib_truths, preds_adapted_eval
    )

    # MAE auxiliar sobre test de pseudo-labels (solo informativo)
    print(f"\n  [Auxiliar] MAE sobre test de pseudo-labels (in-distribution):")
    print(f"    MAE sin calibrar: {mae_test_t5:.4f} dB")
    test_ds_cal = IQDataset(task5["X_test"], task5["y_test"])
    test_loader_cal = DataLoader(test_ds_cal, batch_size=256, shuffle=False)
    _, _, _, preds_test_raw, y_test_raw = evaluate(
        model_adapted, test_loader_cal, nn.SmoothL1Loss(), device
    )
    preds_test_cal = slope_cal * preds_test_raw + intercept_cal
    mae_test_cal   = float(np.mean(np.abs(preds_test_cal - y_test_raw)))
    rmse_test_cal  = float(np.sqrt(np.mean((preds_test_cal - y_test_raw) ** 2)))
    print(f"    MAE calibrado:    {mae_test_cal:.4f} dB  "
          f"[AVISO: ambos miden pseudo-labels, no SNR real]")

    # 6.4 Análisis por receptor sobre X_eval (held-out, no circular)
    print("\n--- 6.4 Análisis por receptor (held-out, no circular) ---")
    rx_analysis = analyze_by_metadata(preds_adapted_eval, meta_eval, 1, "receptor")

    # 6.5 Análisis por transmisor
    print("\n--- 6.5 Análisis por transmisor (held-out) ---")
    tx_analysis = analyze_by_metadata(preds_adapted_eval, meta_eval, 0, "transmisor")

    # Consolidar resultados Fase 5 y 6 en run_log
    run_log["phases"]["phase5_wisig"] = {
        "t5_mae_val":    float(best_mae_t5),
        "t5_mae_test":   float(mae_test_t5),
        "t5_rmse_test":  float(rmse_test_t5),
        "t5_mae_cal_aux": float(mae_test_cal),
        "adapt_n": int(len(X_adapt)),
        "eval_n":  int(len(X_eval)),
        "note": "MAE/RMSE son métricas in-distribution (pseudo-labels). "
                "Ver phase6_eval para métricas de validez externa.",
        "radioml_pre_post": {str(k): v for k, v in radioml_pre_post.items()},
    }
    run_log["phases"]["phase6_eval"] = {
        "note": "Métricas físicas sobre X_eval (held-out, no circular)",
        "degradation": validity_report.get("degradation", {}),
        "m2m4_agreement": validity_report.get("m2m4_agreement", {}),
        "rx_consistency": validity_report.get("rx_consistency", {}),
        "calibration": {
            "slope":     float(slope_cal),
            "intercept": float(intercept_cal),
            "mae_pseudo_pre":  float(mae_test_t5),
            "mae_pseudo_cal":  float(mae_test_cal),
            "rmse_pseudo_cal": float(rmse_test_cal),
        },
    }

else:
    print(f"\n  AVISO: No se encontró {CONFIG['wisig_path']}")
    print(f"  Las fases 5 y 6 requieren el dataset WiSig ManyRX.")
    print(f"  Descárgalo de: https://cores.ee.ucla.edu/downloads/datasets/wisig/")


# ################################################################
# RESUMEN FINAL
# ################################################################
print("\n" + "█" * 60)
print("  RESUMEN FINAL DEL PROYECTO MEJORADO")
print("█" * 60)

_bn_mae_str = (f"{bn_static_mae:.4f} dB MAE"
               if bn_static_mae is not None else "no ejecutado")
print(f"""
  ARQUITECTURA:
    Baseline CNN:   {sum(p.numel() for p in SNREstimatorCNN().parameters()):,} parámetros
    ResNet-SNR:     {sum(p.numel() for p in ResNetSNR().parameters()):,} parámetros

  RENDIMIENTO ESTÁTICO (techo):
    Baseline CNN (z-score):              {baseline_static_mae:.4f} dB MAE
    ResNet-SNR GroupNorm (InstanceNorm): {resnet_static_mae:.4f} dB MAE
    ResNet-SNR BatchNorm (ablación F):   {_bn_mae_str}

  RENDIMIENTO INCREMENTAL — Average Accuracy (MAE medio final):
    Sin replay    (lower bound):  {aa_norep:.4f} dB
    Replay random:                {aa_rep:.4f} dB
    Replay + Output-KD:           {aa_rkd:.4f} dB
    Herding + FKD + EWC:          {aa_full:.4f} dB
    Joint (upper bound):          {joint_aa:.4f} dB
    Gap al upper bound:           {aa_full - joint_aa:+.4f} dB

  FORGETTING MEDIO:
    Sin replay:                   {metrics_norep['mean_forgetting']:.4f} dB
    Replay random:                {metrics_rep['mean_forgetting']:.4f} dB
    Replay + Output-KD:           {metrics_rkd['mean_forgetting']:.4f} dB
    Herding + Feature-KD + EWC:   {metrics_full['mean_forgetting']:.4f} dB
    Reducción: {(1-metrics_full['mean_forgetting']/max(metrics_norep['mean_forgetting'],1e-6))*100:.1f}%

  MEJORAS IMPLEMENTADAS (plan de acción):
    ✓ FASE A — Multi-seed (3 seeds, 4 estrategias): reproducibilidad
    ✓ FASE B — Ablación de componentes (8 configs): contribución individual
    ✓ FASE C — MAE vs SNR: curva por nivel de SNR publicable
    ✓ FASE D — Análisis T3 + espacio features + orden de tareas
    ✓ FASE E — FWT: Forward Transfer (opcional, activar run_fwt=True)
    ✓ FASE F — Ablación GroupNorm vs BatchNorm en entrenamiento estático
    ✓ FASE G — WiSig: calibración per-receptor + opción t5_unfreeze_layer4
    ✓ ResNet-1D con skip connections y GroupNorm (CL-friendly)
    ✓ InstanceNorm interna (domain-agnostic, sin mismatch)
    ✓ Herding selection iCaRL + Feature-KD + EWC multi-tarea
    ✓ Métricas CL: AA, BWT, Forgetting, FWT (cuando run_fwt=True)
    ✓ Adaptación cross-domain multi-head (T1-T4 y WiSig separados)
    ✓ AdamW + Cosine Annealing
""")

# Guardar modelo final
torch.save({
    "model_state_dict": model_best.state_dict(),
    "config": CONFIG,
    "mae_matrix": mae_matrix_full,
}, "checkpoint_resnet_snr_incremental.pt")
print("  Checkpoint guardado: checkpoint_resnet_snr_incremental.pt")

if os.path.exists(CONFIG["wisig_path"]):
    torch.save({
        "model_state_dict": model_adapted.state_dict(),
        "active_head": model_adapted.active_head,  # "wisig" tras T5
        "extra_heads": list(model_adapted.heads.keys()),
        "config": CONFIG,
    }, "checkpoint_resnet_snr_adapted.pt")
    print("  Checkpoint adaptado: checkpoint_resnet_snr_adapted.pt")
    print(f"    Heads guardados: {list(model_adapted.heads.keys())} | "
          f"activo: {model_adapted.active_head}")


# ################################################################
# EVALUACIÓN MULTI-SEED — 4 estrategias (activar en CONFIG para el paper)
# ################################################################
if CONFIG["run_multiseed"]:
    print("\n" + "█" * 60)
    print("  EVALUACIÓN MULTI-SEED — 4 ESTRATEGIAS CL")
    print("█" * 60)
    _start_phase("multiseed")

    def _fresh_task_data():
        """Recarga los datos con la seed activa en cada repetición."""
        td, _ = build_task_data(data_dict, mod_to_idx, normalize="none")
        return td

    # Definición de las 4 estrategias a comparar con media ± std entre seeds
    multiseed_strategies = {
        "naive":     {"buffer_capacity": 0, "lambda_kd": 0,
                      "lambda_feat": 0, "lambda_ewc": 0,
                      "use_ewc": False, "use_herding": False},
        "replay":    {"buffer_capacity": CONFIG["buffer_capacity"],
                      "lambda_kd": 0, "lambda_feat": 0, "lambda_ewc": 0,
                      "use_ewc": False, "use_herding": False},
        "r_kd":      {"buffer_capacity": CONFIG["buffer_capacity"],
                      "lambda_kd": CONFIG["lambda_kd"],
                      "lambda_feat": 0, "lambda_ewc": 0,
                      "use_ewc": False, "use_herding": False},
        "h_fkd_ewc": {"buffer_capacity": CONFIG["buffer_capacity"],
                      "lambda_kd": CONFIG["lambda_kd"],
                      "lambda_feat": CONFIG["lambda_feat"],
                      "lambda_ewc": CONFIG["lambda_ewc"],
                      "use_ewc": True, "use_herding": True},
    }

    multiseed_all = run_multiseed_strategies(
        model_class=ResNetSNR,
        task_data_fn=_fresh_task_data,
        device=device,
        strategies=multiseed_strategies,
        seeds=CONFIG["multiseed_seeds"],
        epochs=CONFIG["incremental_epochs"],
        lr=CONFIG["incremental_lr"],
        batch_size=CONFIG["batch_size"],
        verbose=True,
    )

    torch.save({
        "results": multiseed_all,
        "config": CONFIG,
    }, "multiseed_results.pt")
    print("\n  Resultados multi-seed guardados: multiseed_results.pt")
    _end_phase("multiseed")


# ################################################################

# ################################################################
# OPTIONAL: Task order ablation
# Set CONFIG["run_task_order_ablation"] = True to enable
# ################################################################
if CONFIG["run_task_order_ablation"]:
    print("\n" + "█" * 60)
    print("  FASE D (cont.): SENSIBILIDAD AL ORDEN DE TAREAS")
    print("█" * 60)
    print("  Un método robusto debe producir métricas similares")
    print("  independientemente del orden. Alta varianza = fragilidad.")
    order_results, order_summary = run_task_order_ablation(
        model_class=ResNetSNR,
        task_data=task_data_raw,          # BUG FIX: era `tasks` (indefinido)
        device=device,
        orders=None,                       # original + 2 permutaciones deterministas
        seed=CONFIG["seed"],
        buffer_capacity=CONFIG["buffer_capacity"],
        lambda_kd=CONFIG["lambda_kd"],
        lambda_feat=CONFIG["lambda_feat"],
        lambda_ewc=CONFIG["lambda_ewc"],
        use_ewc=True,
        use_herding=True,
        epochs=CONFIG["incremental_epochs"],
        lr=CONFIG["incremental_lr"],
        batch_size=CONFIG["batch_size"],
    )
    torch.save({"results": order_results, "summary": order_summary},
               "task_order_ablation.pt")
    print("  Resultados guardados: task_order_ablation.pt")
    # Interpretación de la varianza entre órdenes
    aa_std  = order_summary["AA"]["std"]
    forg_std = order_summary["mean_forgetting"]["std"]
    print(f"\n  Interpretación:")
    print(f"    AA  entre órdenes:        {order_summary['AA']['mean']:.4f} ± {aa_std:.4f} dB")
    print(f"    Forgetting entre órdenes: "
          f"{order_summary['mean_forgetting']['mean']:.4f} ± {forg_std:.4f} dB")
    stability = "ROBUSTO" if aa_std < 0.1 else ("MODERADO" if aa_std < 0.3 else "FRÁGIL")
    print(f"    → El método es {stability} al orden de tareas.")
    run_log["phases"]["ablation_task_order"] = {
        "AA":   {"mean": order_summary["AA"]["mean"],
                 "std":  order_summary["AA"]["std"]},
        "BWT":  {"mean": order_summary["BWT"]["mean"],
                 "std":  order_summary["BWT"]["std"]},
        "mean_forgetting": {
            "mean": order_summary["mean_forgetting"]["mean"],
            "std":  order_summary["mean_forgetting"]["std"],
        },
        "n_orders": order_summary["n_orders"],
    }


# ################################################################
# LOGGING ESTRUCTURADO — dump JSON con todos los resultados
# ################################################################
if CONFIG["export_run_log"]:
    import json
    from datetime import datetime

    def _jsonify(o):
        # numpy / torch / tipos no serializables → primitivos Python
        if isinstance(o, (np.floating,)):  return float(o)
        if isinstance(o, (np.integer,)):   return int(o)
        if isinstance(o, np.ndarray):      return o.tolist()
        if isinstance(o, torch.Tensor):    return o.detach().cpu().tolist()
        if isinstance(o, torch.device):    return str(o)
        if isinstance(o, (set, tuple)):    return list(o)
        raise TypeError(f"No serializable: {type(o).__name__}")

    run_log["completed_at"] = datetime.now().isoformat(timespec="seconds")
    run_log["timings"]["total_elapsed_s"] = time.perf_counter() - _pipeline_t0
    json_path = os.path.join(CONFIG["output_dir"], "run_log.json")
    with open(json_path, "w") as f:
        json.dump(run_log, f, indent=2, default=_jsonify)
    print(f"\n  Log estructurado guardado: {json_path}")
    print(f"  Tiempo total pipeline: "
          f"{run_log['timings']['total_elapsed_s']/60:.1f} min")
