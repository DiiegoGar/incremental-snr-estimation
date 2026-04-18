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
    run_multiseed,
)
from evaluation import (
    predict_unlabeled, degradation_test, print_degradation_results,
    analyze_by_metadata, compare_with_m2m4, calibrate_linear
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
    "lambda_ewc": 500.0,

    # WiSig
    "wisig_pseudo_snr_levels": [-10, -5, 0, 5, 10, 15],
    "wisig_adaptation_epochs": 15,

    # Evaluación multi-seed (activar para el paper; desactivar para desarrollo rápido)
    "run_multiseed": False,
    "multiseed_seeds": [42, 123, 2024],
}

device = torch.device(CONFIG["device"])
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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


# ################################################################
# FASE 3: APRENDIZAJE INCREMENTAL — 4 ESTRATEGIAS
# ################################################################
print("\n" + "█" * 60)
print("  FASE 3: APRENDIZAJE INCREMENTAL (4 ESTRATEGIAS)")
print("█" * 60)

# Estrategia 1: Sin replay (baseline — muestra el olvido catastrófico)
print("\n" + "=" * 60)
print("  ESTRATEGIA 1: SIN REPLAY (OLVIDO CATASTRÓFICO)")
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

# Tabla comparativa
print(f"\n{'='*70}")
print(f"  COMPARACIÓN FINAL DE MAE (dB) TRAS LA ÚLTIMA TAREA")
print(f"{'='*70}")
print(f"  {'Tarea':<8} {'Sin Replay':<14} {'Replay':<14} {'R+KD':<14} {'R+FKD+EWC':<14}")
print(f"  {'-'*62}")
for tid in range(1, num_tasks + 1):
    norep = mae_matrix_norep[num_tasks].get(tid, float('nan'))
    rep   = mae_matrix_replay[num_tasks].get(tid, float('nan'))
    rkd   = mae_matrix_replay_kd[num_tasks].get(tid, float('nan'))
    full  = mae_matrix_full[num_tasks].get(tid, float('nan'))
    print(f"  T{tid:<6} {norep:>8.4f}      {rep:>8.4f}      {rkd:>8.4f}      {full:>8.4f}")

# Métricas CL para cada estrategia
metrics_norep = print_cl_metrics(mae_matrix_norep, num_tasks, "SIN REPLAY")
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
        X_wisig, snr_levels=CONFIG["wisig_pseudo_snr_levels"]
    )

    # 5.5 Crear Tarea 5 y adaptar incrementalmente
    print("\n--- 5.5 Adaptación incremental (Tarea 5) ---")
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

    # Guardar modelo pre-adaptación para KD
    old_model_t5 = copy.deepcopy(model_best)
    old_model_t5.eval()

    # Combinar con replay de tareas anteriores
    replay_buffer_t5 = ReplayBuffer(capacity=CONFIG["buffer_capacity"], selection="herding")
    # Llenar con datos de tareas anteriores (task_id para balance equitativo por tarea)
    for t in task_data_raw:
        replay_buffer_t5.add_examples(
            t["X_train"][:2500], t["y_train"][:2500],
            model=model_best, device=device,
            task_id=t["task_id"]
        )

    X_replay_t5, y_replay_t5 = replay_buffer_t5.sample_all()
    X_train_t5 = np.concatenate([task5["X_train"], X_replay_t5])
    y_train_t5 = np.concatenate([task5["y_train"], y_replay_t5])

    print(f"  Datos T5: {len(task5['X_train'])} WiSig + {len(X_replay_t5)} replay "
          f"= {len(X_train_t5)} total")

    # Entrenar Tarea 5
    model_adapted = copy.deepcopy(model_best)
    model_adapted, hist_t5, best_mae_t5, best_ep_t5 = train_task_incremental(
        model_adapted, old_model_t5,
        X_train_t5, y_train_t5, task5["X_val"], task5["y_val"],
        device, epochs=CONFIG["wisig_adaptation_epochs"],
        batch_size=CONFIG["batch_size"], lr=1e-4,
        lambda_kd=CONFIG["lambda_kd"], lambda_feat=CONFIG["lambda_feat"]
    )
    print(f"\n  Tarea 5 completada — MAE val: {best_mae_t5:.4f} dB (época {best_ep_t5})")

    # Evaluación sobre test set independiente (nunca visto durante entrenamiento)
    criterion_t5 = nn.SmoothL1Loss()
    test_ds_t5 = IQDataset(task5["X_test"], task5["y_test"])
    test_loader_t5 = DataLoader(test_ds_t5, batch_size=256, shuffle=False)
    _, mae_test_t5, rmse_test_t5, _, _ = evaluate(model_adapted, test_loader_t5, criterion_t5, device)
    print(f"  Tarea 5 — MAE test: {mae_test_t5:.4f} dB | RMSE test: {rmse_test_t5:.4f} dB")

    # 5.6 Verificar que no olvidó RadioML
    print("\n--- 5.6 Verificación anti-olvido tras Tarea 5 ---")
    criterion = nn.SmoothL1Loss()
    print(f"  {'Tarea':<10} {'Pre-T5 (dB)':<15} {'Post-T5 (dB)':<15} {'Δ (dB)':<10}")
    print(f"  {'-'*50}")
    for tid in range(1, num_tasks + 1):
        task = task_data_raw[tid - 1]
        eval_ds = IQDataset(task["X_test"], task["y_test"])
        eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)

        _, mae_pre, _, _, _ = evaluate(model_best, eval_loader, criterion, device)
        _, mae_post, _, _, _ = evaluate(model_adapted, eval_loader, criterion, device)
        delta = mae_post - mae_pre
        print(f"  T{tid:<8} {mae_pre:>8.4f}        {mae_post:>8.4f}        {delta:>+7.4f}")


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
    print("\n--- 6.7 Comparación con estimador clásico M2M4 ---")
    m2m4_comparison = compare_with_m2m4(preds_adapted, X_wisig)

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
        "config": CONFIG,
    }, "checkpoint_resnet_snr_adapted.pt")
    print("  Checkpoint adaptado: checkpoint_resnet_snr_adapted.pt")


# ################################################################
# EVALUACIÓN MULTI-SEED (activar en CONFIG para el paper)
# ################################################################
if CONFIG["run_multiseed"]:
    print("\n" + "█" * 60)
    print("  EVALUACIÓN MULTI-SEED — Estrategia completa (H+FKD+EWC)")
    print("█" * 60)

    def _fresh_task_data():
        """Recarga los datos con la seed activa en cada repetición."""
        td, _ = build_task_data(data_dict, mod_to_idx, normalize="none")
        return td

    multiseed_summary, multiseed_all = run_multiseed(
        model_class=ResNetSNR,
        task_data_fn=_fresh_task_data,
        device=device,
        seeds=CONFIG["multiseed_seeds"],
        buffer_capacity=CONFIG["buffer_capacity"],
        lambda_kd=CONFIG["lambda_kd"],
        lambda_feat=CONFIG["lambda_feat"],
        lambda_ewc=CONFIG["lambda_ewc"],
        use_ewc=True, use_herding=True,
        epochs=CONFIG["incremental_epochs"],
        lr=CONFIG["incremental_lr"],
        batch_size=CONFIG["batch_size"],
        verbose=True,
    )

    torch.save({
        "summary": multiseed_summary,
        "all_metrics": multiseed_all,
        "config": CONFIG,
    }, "multiseed_results.pt")
    print("\n  Resultados multi-seed guardados: multiseed_results.pt")
