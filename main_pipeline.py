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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Módulos del proyecto
from model import SNREstimatorCNN, ResNetSNR, model_summary
from data_utils import (
    load_radioml, build_task_data, load_wisig_manyrx,
    generate_wisig_pseudo_labels, add_awgn,
    NormalizationStrategy, IQDataset, make_loaders
)
from incremental_engine import (
    ReplayBuffer, EWC,
    train_one_epoch, evaluate, train_task_incremental,
    run_incremental_pipeline, print_cl_metrics, compute_forgetting,
    compute_average_accuracy, compute_backward_transfer,
    run_multiseed, run_multiseed_strategies,
    compute_random_init_baselines, compute_fwt,
    run_task_order_ablation,
    run_hyperparam_ablation, export_mae_matrix_csv,
)
from evaluation import (
    predict_unlabeled, degradation_test, print_degradation_results,
    analyze_by_metadata, compare_with_m2m4, calibrate_linear, KAPPA_S
)

# ============================================================
# CONFIGURACIÓN
# ============================================================
CONFIG = {
    # Rutas (ajustar según tu estructura)
    "radioml_path": "data/RML2016.10a_dict.pkl",
    "wisig_path": "data/ManyRX.pkl",

    # Entrenamiento
    "seed": 42,
    "batch_size": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

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

    # Evaluación multi-seed (activar para el paper; desactivar para desarrollo rápido)
    "run_multiseed": False,
    "multiseed_seeds": [42, 123, 2024],

    # Ablación sobre orden de tareas (activar para análisis de sensibilidad al orden)
    "run_task_order_ablation": False,

    # Ablación de hiperparámetros (opcional, para el paper)
    "run_lambda_ablation": False,
    "run_buffer_ablation": False,

    # Exportaciones estructuradas (CSV de R_ij + JSON de log completo)
    "export_csv":     True,
    "export_run_log": True,
    "output_dir":     "outputs",
}

device = torch.device(CONFIG["device"])
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Directorio de salidas + log estructurado (serializable a JSON al final).
os.makedirs(CONFIG["output_dir"], exist_ok=True)
run_log = {
    "config": {k: (str(v) if isinstance(v, torch.device) else v)
               for k, v in CONFIG.items()},
    "phases": {},
}


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

# Estrategia 1: Naive fine-tuning (lower bound — muestra olvido catastrófico)
# Sin replay, sin KD, sin EWC: referencia mínima frente a la que cualquier
# método CL serio debe mejorar. Si una estrategia no supera a Naive,
# no aporta nada sobre la alternativa trivial.
print("\n" + "=" * 60)
print("  ESTRATEGIA 1: NAIVE FINE-TUNING (lower bound — olvido catastrófico)")
print("=" * 60)
_, mae_matrix_norep, _ = run_incremental_pipeline(
    ResNetSNR, task_data_raw, device,
    buffer_capacity=0, lambda_kd=0, lambda_feat=0,
    use_ewc=False, use_herding=False,
    epochs=CONFIG["incremental_epochs"], lr=CONFIG["incremental_lr"]
)

# Estrategia 2: Solo Replay (como el notebook original, pero con ResNet)
print("\n" + "=" * 60)
print("  ESTRATEGIA 2: REPLAY BUFFER (RANDOM)")
print("=" * 60)
_, mae_matrix_replay, _ = run_incremental_pipeline(
    ResNetSNR, task_data_raw, device,
    buffer_capacity=CONFIG["buffer_capacity"], lambda_kd=0, lambda_feat=0,
    use_ewc=False, use_herding=False,
    epochs=CONFIG["incremental_epochs"], lr=CONFIG["incremental_lr"]
)

# Estrategia 3: Replay + Output-KD (similar al original, sin feature-KD ni EWC)
print("\n" + "=" * 60)
print("  ESTRATEGIA 3: REPLAY + OUTPUT-KD")
print("=" * 60)
_, mae_matrix_replay_kd, _ = run_incremental_pipeline(
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
# ABLACIÓN DE HIPERPARÁMETROS (opcional)
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

    # 5.2 Predicción zero-shot (sin adaptación)
    print("\n--- 5.2 Predicción zero-shot (sin adaptación) ---")
    preds_zeroshot = predict_unlabeled(model_best, X_wisig, device)
    print(f"  Predicciones zero-shot:")
    print(f"    Media: {preds_zeroshot.mean():.2f} dB | Std: {preds_zeroshot.std():.2f} dB")
    print(f"    Rango: [{preds_zeroshot.min():.2f}, {preds_zeroshot.max():.2f}] dB")

    # 5.3 Test de degradación (zero-shot)
    print("\n--- 5.3 Test de degradación (zero-shot) ---")
    deg_results_zs = degradation_test(model_best, X_wisig[:10000], device)
    print_degradation_results(deg_results_zs)

    # 5.4 Generar pseudo-labels
    print("\n--- 5.4 Generando pseudo-labels para WiSig ---")
    X_pseudo, y_pseudo = generate_wisig_pseudo_labels(
        X_wisig, snr_levels=CONFIG["wisig_pseudo_snr_levels"],
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

    # Freezing: TODO el extractor + head_radio quedan congelados.
    # Solo entrenamos heads.wisig (~33k params). Si t5_train_input_norm=True,
    # también input_norm libre (afecta ligeramente a las features que verá
    # head_radio en T1-T4 después).
    trainable_keys = ["heads.wisig"]
    if CONFIG.get("t5_train_input_norm", False):
        trainable_keys.append("input_norm")
    for name, p in model_adapted.named_parameters():
        p.requires_grad = any(k in name for k in trainable_keys)
    n_train_p = sum(p.numel() for p in model_adapted.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in model_adapted.parameters())
    print(f"  Multi-head WiSig — entrenables: {n_train_p:,}/{n_total_p:,} "
          f"({100*n_train_p/n_total_p:.1f}%)")
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
    # FASE 6: EVALUACIÓN COMPLETA EN WiSig
    # ################################################################
    print("\n" + "█" * 60)
    print("  FASE 6: EVALUACIÓN CUANTITATIVA EN WiSig")
    print("█" * 60)

    # 6.1 Predicción post-adaptación
    print("\n--- 6.1 Predicción post-adaptación ---")
    preds_adapted = predict_unlabeled(model_adapted, X_wisig, device)
    print(f"  Zero-shot:       Media={preds_zeroshot.mean():.2f} dB, "
          f"Std={preds_zeroshot.std():.2f} dB")
    print(f"  Post-adaptación: Media={preds_adapted.mean():.2f} dB, "
          f"Std={preds_adapted.std():.2f} dB")

    # 6.2 Test de degradación post-adaptación
    print("\n--- 6.2 Test de degradación post-adaptación ---")
    deg_results_adapted = degradation_test(model_adapted, X_wisig[:10000], device)
    print_degradation_results(deg_results_adapted)

    # 6.3 Comparación de monotonicidad
    print(f"\n  Mejora en Spearman ρ: "
          f"{deg_results_zs['spearman_rho']:.4f} → {deg_results_adapted['spearman_rho']:.4f}")
    print(f"  Mejora en sensibilidad: "
          f"{deg_results_zs['mean_sensitivity']:.4f} → {deg_results_adapted['mean_sensitivity']:.4f}")

    # 6.3b CALIBRACIÓN LINEAL POST-HOC
    # Las pseudo-labels sesgan la escala del modelo en WiSig. Corregimos
    # slope/intercept con regresión sobre los puntos del degradation_test
    # (SNR controlado conocido → predicción observada). Esto no resuelve el
    # sesgo absoluto (WiSig ya trae ruido de canal desconocido), pero sí
    # ajusta compresión/expansión de la escala y offset sistemático.
    print("\n--- 6.3b Calibración lineal post-hoc ---")
    calib_preds  = []
    calib_truths = []
    for level, mean_pred in zip(deg_results_adapted["levels"],
                                 deg_results_adapted["means"]):
        if level == "clean":
            continue  # sin SNR ground truth asumible
        snr_true = float(level.replace("dB", ""))
        preds_at_level = deg_results_adapted["preds"][level]
        calib_preds.extend(preds_at_level.tolist())
        calib_truths.extend([snr_true] * len(preds_at_level))
    calib_preds  = np.array(calib_preds)
    calib_truths = np.array(calib_truths)

    preds_calibrated, slope_cal, intercept_cal = calibrate_linear(
        calib_preds, calib_truths, preds_adapted
    )
    print(f"  Pre-calibración  — Media: {preds_adapted.mean():.2f} dB, "
          f"Std: {preds_adapted.std():.2f} dB")
    print(f"  Post-calibración — Media: {preds_calibrated.mean():.2f} dB, "
          f"Std: {preds_calibrated.std():.2f} dB")

    # Calibrar también MAE sobre task5 test — si la calibración mejora MAE,
    # el modelo tenía sesgo de escala sistemático.
    test_ds_cal = IQDataset(task5["X_test"], task5["y_test"])
    test_loader_cal = DataLoader(test_ds_cal, batch_size=256, shuffle=False)
    _, _, _, preds_test_raw, y_test_raw = evaluate(
        model_adapted, test_loader_cal, nn.SmoothL1Loss(), device
    )
    preds_test_cal = slope_cal * preds_test_raw + intercept_cal
    mae_test_cal  = float(np.mean(np.abs(preds_test_cal - y_test_raw)))
    rmse_test_cal = float(np.sqrt(np.mean((preds_test_cal - y_test_raw) ** 2)))
    print(f"\n  T5 test (calibrado) — MAE: {mae_test_cal:.4f} dB | "
          f"RMSE: {rmse_test_cal:.4f} dB")
    print(f"  Reducción MAE: {mae_test_t5 - mae_test_cal:+.4f} dB")
    print(f"  AVISO: El MAE absoluto en WiSig depende de las pseudo-labels.")
    print(f"         Las métricas primarias son Spearman ρ y sensibilidad.")

    # 6.4 Análisis por receptor
    print("\n--- 6.4 Análisis por receptor ---")
    rx_analysis = analyze_by_metadata(preds_adapted, meta_wisig, 1, "receptor")

    # 6.5 Análisis por transmisor
    print("\n--- 6.5 Análisis por transmisor ---")
    tx_analysis = analyze_by_metadata(preds_adapted, meta_wisig, 0, "transmisor")

    # 6.6 Análisis por fecha
    print("\n--- 6.6 Análisis por fecha ---")
    date_analysis = analyze_by_metadata(preds_adapted, meta_wisig, 2, "fecha")

    # 6.7 Comparación con M2M4
    # WiSig es WiFi-OFDM. M2M4 NO aplica (κ_s≈2 → indeterminado).
    # Solo se reporta como referencia descriptiva, no como baseline válido.
    print("\n--- 6.7 Comparación con estimador clasico M2M4 ---")
    m2m4_comparison = compare_with_m2m4(
        preds_adapted, X_wisig, kappa_s=1.0,
        signal_type="WiSig WiFi-OFDM"
    )

    # Consolidar resultados Fase 5 y 6 en run_log
    run_log["phases"]["phase5_wisig"] = {
        "t5_mae_val":   float(best_mae_t5),
        "t5_mae_test":  float(mae_test_t5),
        "t5_rmse_test": float(rmse_test_t5),
        "radioml_pre_post": {str(k): v for k, v in radioml_pre_post.items()},
    }
    run_log["phases"]["phase6_eval"] = {
        "spearman_rho": {
            "zero_shot": float(deg_results_zs["spearman_rho"]),
            "adapted":   float(deg_results_adapted["spearman_rho"]),
        },
        "mean_sensitivity": {
            "zero_shot": float(deg_results_zs["mean_sensitivity"]),
            "adapted":   float(deg_results_adapted["mean_sensitivity"]),
        },
        "calibration": {
            "slope":     float(slope_cal),
            "intercept": float(intercept_cal),
            "mae_pre":   float(mae_test_t5),
            "mae_post":  float(mae_test_cal),
            "rmse_post": float(rmse_test_cal),
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

print(f"""
  ARQUITECTURA:
    Baseline CNN:   {sum(p.numel() for p in SNREstimatorCNN().parameters()):,} parámetros
    ResNet-SNR:     {sum(p.numel() for p in ResNetSNR().parameters()):,} parámetros

  RENDIMIENTO ESTÁTICO (techo):
    Baseline CNN (z-score):      {baseline_static_mae:.4f} dB MAE
    ResNet-SNR (InstanceNorm):   {resnet_static_mae:.4f} dB MAE

  RENDIMIENTO INCREMENTAL (forgetting medio):
    Sin replay:                  {metrics_norep['mean_forgetting']:.4f} dB
    Replay random:               {metrics_rep['mean_forgetting']:.4f} dB
    Replay + Output-KD:          {metrics_rkd['mean_forgetting']:.4f} dB
    Herding + Feature-KD + EWC:  {metrics_full['mean_forgetting']:.4f} dB

  MEJORAS IMPLEMENTADAS:
    ✓ ResNet-1D con skip connections y kernel adaptativos
    ✓ InstanceNorm interna (resuelve domain mismatch)
    ✓ Herding selection para Replay Buffer (tipo iCaRL)
    ✓ Feature-level Knowledge Distillation
    ✓ Elastic Weight Consolidation (EWC)
    ✓ Métricas CL estándar (AA, BWT, Forgetting)
    ✓ Tarea 5 de adaptación cross-domain con pseudo-labels
    ✓ Evaluación cuantitativa con M2M4 y métricas de calibración
    ✓ AdamW + Cosine Annealing (vs Adam + StepLR original)
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


# ################################################################

# ################################################################
# OPTIONAL: Task order ablation
# Set CONFIG["run_task_order_ablation"] = True to enable
# ################################################################
if CONFIG["run_task_order_ablation"]:
    print()
    print("=" * 60)
    print("  ABLATION: TASK ORDER SENSITIVITY")
    print("=" * 60)
    ablation_results = run_task_order_ablation(
        model_class=ResNetSNR,
        task_data=tasks,
        device=device,
        orders=None,
        seed=42,
        buffer_capacity=CONFIG["buffer_capacity"],
        lambda_kd=CONFIG["lambda_kd"],
        lambda_feat=CONFIG["lambda_feat"],
        lambda_ewc=CONFIG["lambda_ewc"],
        epochs=CONFIG["incremental_epochs"],
        lr=CONFIG["incremental_lr"],
        batch_size=CONFIG["batch_size"],
    )
    torch.save(ablation_results, "task_order_ablation.pt")
    print("  Results saved: task_order_ablation.pt")
    run_log["phases"]["ablation_task_order"] = {
        "mean_AA":  float(ablation_results.get("mean_AA", float("nan"))),
        "std_AA":   float(ablation_results.get("std_AA", float("nan"))),
        "mean_BWT": float(ablation_results.get("mean_BWT", float("nan"))),
        "std_BWT":  float(ablation_results.get("std_BWT", float("nan"))),
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
    json_path = os.path.join(CONFIG["output_dir"], "run_log.json")
    with open(json_path, "w") as f:
        json.dump(run_log, f, indent=2, default=_jsonify)
    print(f"\n  Log estructurado guardado: {json_path}")
