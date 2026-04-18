"""
incremental_engine.py — Motor de aprendizaje incremental mejorado
=================================================================
Incluye:
  - ReplayBuffer con Herding Selection (MEJORA 4.3.1)
  - Feature-level Knowledge Distillation (MEJORA 4.3.2)
  - Elastic Weight Consolidation (MEJORA 4.3.3)
  - Métricas estándar de CL: Average Accuracy, BWT, FWT
  - Pipeline completo de entrenamiento incremental
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import IQDataset


# ============================================================
# 1. REPLAY BUFFER CON HERDING SELECTION (MEJORA 4.3.1)
# ============================================================
class ReplayBuffer:
    """
    Replay buffer con balance por tarea (estilo iCaRL) y dos modos de selección.

    El presupuesto de memoria (capacity) se reparte equitativamente entre todas
    las tareas vistas: ejemplares_por_tarea = capacity // num_tareas_vistas.
    Al registrar una nueva tarea, las antiguas se podan al nuevo presupuesto.
    Esto evita que las tareas con features más "centrales" dominen el buffer.

    Modos de selección:
      - herding: selección greedy vectorizada por cercanía al centroide (iCaRL)
      - random:  selección aleatoria uniforme
    """
    def __init__(self, capacity, selection="herding"):
        self.capacity  = capacity
        self.selection = selection
        # Almacenamiento por tarea: {task_id: {"X": ndarray, "y": ndarray}}
        self._tasks = {}

    def __len__(self):
        return sum(len(v["X"]) for v in self._tasks.values())

    @property
    def X(self):
        """Vista concatenada de todos los ejemplares (compatibilidad con código existente)."""
        if not self._tasks:
            return None
        return np.concatenate([v["X"] for v in self._tasks.values()], axis=0)

    @property
    def y(self):
        if not self._tasks:
            return None
        return np.concatenate([v["y"] for v in self._tasks.values()], axis=0)

    def add_examples(self, X_new, y_new, model=None, device=None, task_id=None):
        """
        Registra los ejemplares de una nueva tarea y redistribuye el presupuesto.

        Args:
            X_new, y_new: datos de la tarea nueva
            model, device: necesarios para herding selection
            task_id:       identificador de la tarea (int); si None se autoasigna
        """
        if len(X_new) == 0:
            return

        if task_id is None:
            task_id = len(self._tasks) + 1

        # Presupuesto por tarea tras añadir la nueva
        num_tasks   = len(self._tasks) + 1
        budget      = max(1, self.capacity // num_tasks)

        # Seleccionar ejemplares de la nueva tarea
        X_sel, y_sel = self._select(X_new, y_new, budget, model, device)
        self._tasks[task_id] = {"X": X_sel, "y": y_sel}

        # Reducir tareas antiguas al nuevo presupuesto si es necesario
        for tid in list(self._tasks.keys()):
            if tid == task_id:
                continue
            X_t, y_t = self._tasks[tid]["X"], self._tasks[tid]["y"]
            if len(X_t) > budget:
                X_t, y_t = self._select(X_t, y_t, budget, model, device)
                self._tasks[tid] = {"X": X_t, "y": y_t}

    def _select(self, X, y, n, model, device):
        """Selecciona n ejemplares de (X, y) según el modo configurado."""
        if len(X) <= n:
            return np.copy(X), np.copy(y)
        if self.selection == "herding" and model is not None and device is not None:
            idx = self._herding_select_from(X, n, model, device)
        else:
            idx = np.random.choice(len(X), n, replace=False)
        return X[idx], y[idx]

    def _herding_select_from(self, X, n, model, device):
        """
        Herding selection vectorizado sobre un array X arbitrario.
        Selecciona n índices cuya media acumulada de features se aproxima
        al centroide de X. Vectorizado: el bucle interno es broadcast NumPy.

        Inspirado en iCaRL (Rebuffi et al., 2017).
        """
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        features_list = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 256):
                batch = X_tensor[i:i+256].to(device)
                feat = model.extract_features(batch)
                features_list.append(feat.cpu().numpy())
        features = np.concatenate(features_list, axis=0)  # (N, D)

        centroid      = features.mean(axis=0)              # (D,)
        selected      = []
        selected_mask = np.zeros(len(features), dtype=bool)
        selected_sum  = np.zeros_like(centroid)

        for k in range(min(n, len(features))):
            target         = (k + 1) * centroid
            candidate_sums = selected_sum + features       # (N, D) broadcast
            dists          = np.linalg.norm(candidate_sums - target, axis=1)
            dists[selected_mask] = np.inf
            best_idx       = int(dists.argmin())
            selected.append(best_idx)
            selected_sum  += features[best_idx]
            selected_mask[best_idx] = True

        return np.array(selected)

    def sample_all(self):
        if self.X is None:
            return None, None
        return self.X, self.y

    def sample(self, n):
        """Muestrea n ejemplos aleatorios del buffer."""
        if self.X is None or n <= 0:
            return None, None
        idx = np.random.choice(len(self.X), min(n, len(self.X)), replace=False)
        return self.X[idx], self.y[idx]


# ============================================================
# 2. ELASTIC WEIGHT CONSOLIDATION (MEJORA 4.3.3)
# ============================================================
class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Implementación correcta multi-tarea: cada tarea registrada aporta su propio
    par (Fisher_t, params*_t). La penalización suma sobre todas las tareas vistas:

        L_EWC = λ · Σ_t  Σ_i  F_i(t) · (θ_i - θ*_i(t))²

    Esto difiere de acumular una Fisher global con un único vector de referencia,
    que es el error habitual: la Fisher acumulada penalizaría el alejamiento de
    los pesos de la última tarea, no de cada tarea por separado.
    """
    def __init__(self, model, lambda_ewc=1000.0):
        self.lambda_ewc = lambda_ewc
        # Historial por tarea: lista de dicts {param_name: tensor}
        self.task_fishers = []   # F_t para cada tarea t
        self.task_params  = []   # θ*_t para cada tarea t

    def register_task(self, model, data_loader, device, n_samples=2000):
        """
        Calcula la Fisher diagonal de la tarea actual y guarda los pesos
        óptimos de este momento. Debe llamarse DESPUÉS de entrenar cada tarea.
        """
        model.eval()
        fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                       if p.requires_grad}
        count = 0

        for X_batch, y_batch in data_loader:
            if count >= n_samples:
                break
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            model.zero_grad()
            preds = model(X_batch)
            loss = nn.MSELoss()(preds, y_batch)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_diag[n] += p.grad.data ** 2 * X_batch.size(0)
            count += X_batch.size(0)

        # Promediar Fisher de esta tarea (no acumular globalmente)
        for n in fisher_diag:
            fisher_diag[n] /= count

        # Guardar el par (Fisher_t, params*_t) de esta tarea
        self.task_fishers.append({n: f.clone() for n, f in fisher_diag.items()})
        self.task_params.append({n: p.data.clone() for n, p in model.named_parameters()
                                  if p.requires_grad})

    def penalty(self, model):
        """
        Penalización EWC sumando las contribuciones de todas las tareas vistas.
        Retorna 0.0 si todavía no se ha registrado ninguna tarea.
        """
        if not self.task_fishers:
            return torch.tensor(0.0)

        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)

        for fisher_t, params_t in zip(self.task_fishers, self.task_params):
            for n, p in model.named_parameters():
                if p.requires_grad and n in fisher_t:
                    f = fisher_t[n].to(device)
                    p_star = params_t[n].to(device)
                    loss += (f * (p - p_star) ** 2).sum()

        return self.lambda_ewc * loss


# ============================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entrenamiento estándar de una época."""
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)


def train_one_epoch_incremental(model, old_model, loader, criterion, optimizer,
                                 device, lambda_kd=0.5, lambda_feat=0.3,
                                 ewc_module=None):
    """
    Entrenamiento de una época con las 3 mejoras combinadas:
      1. Task loss (SmoothL1)
      2. Knowledge Distillation en output + features (MEJORA 4.3.2)
      3. EWC regularization (MEJORA 4.3.3)
    """
    model.train()
    running_loss = 0.0
    kd_criterion = nn.MSELoss()
    feat_criterion = nn.MSELoss()

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass actual
        preds = model(X_batch)
        loss_task = criterion(preds, y_batch)

        # Knowledge Distillation (output-level + feature-level)
        loss_kd = torch.tensor(0.0, device=device)
        if old_model is not None:
            with torch.no_grad():
                old_preds = old_model(X_batch)
                old_features = old_model.extract_features(X_batch)
            
            # Output-level KD
            loss_kd_output = kd_criterion(preds, old_preds)
            
            # Feature-level KD (MEJORA 4.3.2)
            new_features = model.extract_features(X_batch)
            loss_kd_feat = feat_criterion(new_features, old_features)
            
            loss_kd = lambda_kd * loss_kd_output + lambda_feat * loss_kd_feat

        # EWC regularization (MEJORA 4.3.3)
        loss_ewc = torch.tensor(0.0, device=device)
        if ewc_module is not None:
            loss_ewc = ewc_module.penalty(model)

        # Total loss
        loss = loss_task + loss_kd + loss_ewc
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Evaluación completa con todas las métricas."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    loss_mean = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    return loss_mean, mae, rmse, all_preds, all_targets


# ============================================================
# 4. PIPELINE INCREMENTAL COMPLETO
# ============================================================
def train_task_incremental(model, old_model, X_train, y_train, X_val, y_val,
                           device, ewc_module=None, epochs=20, batch_size=256,
                           lr=3e-4, lambda_kd=0.5, lambda_feat=0.3):
    """
    Entrena una tarea con el pipeline incremental completo:
    Replay + Feature-KD + EWC.
    """
    train_loader, val_loader = _make_loaders(X_train, y_train, X_val, y_val, batch_size)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float("inf")
    best_model_state = None
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch_incremental(
            model, old_model, train_loader, criterion, optimizer,
            device, lambda_kd=lambda_kd, lambda_feat=lambda_feat,
            ewc_module=ewc_module
        )
        val_loss, val_mae, val_rmse, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train: {train_loss:.4f} | Val MAE: {val_mae:.4f} dB | "
              f"Val RMSE: {val_rmse:.4f} dB", end="")
        if epoch + 1 == best_epoch:
            print(" ★", end="")
        print()

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_val_mae, best_epoch


def run_incremental_pipeline(model_class, task_data, device,
                             buffer_capacity=10000, lambda_kd=0.5,
                             lambda_feat=0.3, lambda_ewc=500.0,
                             use_ewc=True, use_herding=True,
                             epochs=20, lr=3e-4, batch_size=256,
                             model_kwargs=None):
    """
    Ejecuta el pipeline incremental completo sobre todas las tareas.
    
    Returns:
        model: modelo final entrenado
        mae_matrix: dict[after_task][eval_task] = MAE
        results: dict con historial detallado
    """
    if model_kwargs is None:
        model_kwargs = {}

    model = model_class(**model_kwargs).to(device)
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity,
        selection="herding" if use_herding else "random"
    )
    ewc_module = EWC(model, lambda_ewc=lambda_ewc) if use_ewc else None

    mae_matrix = {}
    all_results = {}

    for current_task_id, task in enumerate(task_data, 1):
        print(f"\n{'='*60}")
        print(f"TAREA {current_task_id}: {task['mods']}")
        print(f"{'='*60}")

        # 1. Congelar modelo anterior para KD
        old_model = None
        if current_task_id > 1:
            old_model = copy.deepcopy(model)
            old_model.to(device)
            old_model.eval()

        # 2. Combinar datos actuales + replay
        X_current = task["X_train"]
        y_current = task["y_train"]
        X_replay, y_replay = replay_buffer.sample_all()

        if X_replay is not None:
            X_train_combined = np.concatenate([X_current, X_replay], axis=0)
            y_train_combined = np.concatenate([y_current, y_replay], axis=0)
            print(f"  Datos: {len(X_current)} actuales + {len(X_replay)} replay "
                  f"= {len(X_train_combined)} total")
        else:
            X_train_combined = X_current
            y_train_combined = y_current
            print(f"  Datos: {len(X_current)} (sin replay)")

        # 3. Entrenar
        model, history, best_mae, best_epoch = train_task_incremental(
            model, old_model, X_train_combined, y_train_combined,
            task["X_val"], task["y_val"], device,
            ewc_module=ewc_module, epochs=epochs, batch_size=batch_size,
            lr=lr, lambda_kd=lambda_kd, lambda_feat=lambda_feat
        )
        print(f"  Mejor época: {best_epoch} | Mejor MAE: {best_mae:.4f} dB")

        # 4. Registrar EWC para esta tarea
        if ewc_module is not None:
            train_loader, _ = _make_loaders(X_current, y_current,
                                            task["X_val"], task["y_val"], batch_size)
            ewc_module.register_task(model, train_loader, device)

        # 5. Actualizar replay buffer
        replay_buffer.add_examples(X_current, y_current, model=model, device=device,
                                    task_id=current_task_id)
        buf_detail = ", ".join(f"T{tid}:{len(v["X"])}" for tid, v in replay_buffer._tasks.items())
        print(f"  Buffer size: {len(replay_buffer)} ({buf_detail})")

        # 6. Evaluar en todas las tareas vistas
        criterion = nn.SmoothL1Loss()
        mae_matrix[current_task_id] = {}
        seen_results = {}
        for eval_task_id in range(1, current_task_id + 1):
            eval_task = task_data[eval_task_id - 1]
            eval_ds = IQDataset(eval_task["X_test"], eval_task["y_test"])
            eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)
            _, mae, rmse, preds, targets = evaluate(model, eval_loader, criterion, device)
            mae_matrix[current_task_id][eval_task_id] = mae
            seen_results[eval_task_id] = {"mae": mae, "rmse": rmse}

        print(f"  Evaluación tras T{current_task_id}:")
        for tid, res in seen_results.items():
            print(f"    T{tid}: MAE={res['mae']:.4f} dB | RMSE={res['rmse']:.4f} dB")

        all_results[current_task_id] = {
            "history": history,
            "eval": seen_results,
            "best_mae": best_mae,
            "best_epoch": best_epoch,
        }

    return model, mae_matrix, all_results


# ============================================================
# 5. MÉTRICAS ESTÁNDAR DE CONTINUAL LEARNING (MEJORA 5.1)
# ============================================================
def compute_forgetting(mae_matrix, num_tasks):
    """
    Calcula el forgetting por tarea:
    F_j = MAE_final(T_j) - min_k(MAE_k(T_j)) para k donde T_j fue vista
    """
    forgetting = {}
    for task_id in range(1, num_tasks + 1):
        best_mae = float("inf")
        for after_task in range(task_id, num_tasks + 1):
            if after_task in mae_matrix and task_id in mae_matrix[after_task]:
                best_mae = min(best_mae, mae_matrix[after_task][task_id])
        final_mae = mae_matrix[num_tasks].get(task_id, float("inf"))
        forgetting[task_id] = final_mae - best_mae
    return forgetting


def compute_average_accuracy(mae_matrix, num_tasks):
    """
    Average Accuracy (AA): media del MAE en la diagonal y debajo
    tras la última tarea.
    """
    maes = []
    for task_id in range(1, num_tasks + 1):
        if task_id in mae_matrix[num_tasks]:
            maes.append(mae_matrix[num_tasks][task_id])
    return np.mean(maes) if maes else float("inf")


def compute_backward_transfer(mae_matrix, num_tasks):
    """
    Backward Transfer (BWT): media del cambio en rendimiento de tareas
    anteriores tras aprender nuevas.
    BWT negativo = olvido. BWT positivo = mejora retroactiva.
    """
    bwt_values = []
    for task_id in range(1, num_tasks):
        mae_just_learned = mae_matrix[task_id][task_id]
        mae_final = mae_matrix[num_tasks][task_id]
        bwt_values.append(mae_just_learned - mae_final)  # positivo = mejoró
    return np.mean(bwt_values) if bwt_values else 0.0


def print_cl_metrics(mae_matrix, num_tasks, label=""):
    """Imprime todas las métricas de CL."""
    forgetting = compute_forgetting(mae_matrix, num_tasks)
    aa = compute_average_accuracy(mae_matrix, num_tasks)
    bwt = compute_backward_transfer(mae_matrix, num_tasks)

    print(f"\n{'='*50}")
    print(f"  MÉTRICAS CL {label}")
    print(f"{'='*50}")
    print(f"  Average Accuracy (MAE medio final): {aa:.4f} dB")
    print(f"  Backward Transfer:  {bwt:+.4f} dB {'(mejora)' if bwt > 0 else '(olvido)'}")
    print(f"  Forgetting por tarea:")
    for tid, f in forgetting.items():
        print(f"    T{tid}: {f:+.4f} dB")
    mean_f = np.mean(list(forgetting.values()))
    print(f"  Forgetting medio: {mean_f:.4f} dB")
    return {"AA": aa, "BWT": bwt, "forgetting": forgetting, "mean_forgetting": mean_f}


# ============================================================
# 6. EVALUACIÓN MULTI-SEED
# ============================================================
def _set_seed(seed):
    """Fija todas las fuentes de aleatoriedad relevantes para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_multiseed(model_class, task_data_fn, device,
                  seeds=(42, 123, 2024),
                  buffer_capacity=10000, lambda_kd=0.5,
                  lambda_feat=0.3, lambda_ewc=500.0,
                  use_ewc=True, use_herding=True,
                  epochs=20, lr=3e-4, batch_size=256,
                  model_kwargs=None, verbose=False):
    """
    Repite el pipeline incremental completo con distintas semillas y agrega
    media ± desviación estándar de las métricas CL principales.

    Args:
        model_class:   clase del modelo (p.ej. ResNetSNR)
        task_data_fn:  callable sin argumentos que devuelve task_data fresco
                       para cada seed (debe respetar la seed fijada internamente)
        device:        torch.device
        seeds:         semillas a usar; por defecto (42, 123, 2024)
        verbose:       si True, imprime el detalle de cada seed

    Returns:
        summary: dict con media y std de AA, BWT, mean_forgetting por seed
        all_metrics: lista de dicts de métricas por seed (para análisis posterior)
    """
    all_metrics = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'='*60}")
        _set_seed(seed)

        task_data = task_data_fn()
        num_tasks = len(task_data)

        _, mae_matrix, _ = run_incremental_pipeline(
            model_class, task_data, device,
            buffer_capacity=buffer_capacity,
            lambda_kd=lambda_kd, lambda_feat=lambda_feat,
            lambda_ewc=lambda_ewc,
            use_ewc=use_ewc, use_herding=use_herding,
            epochs=epochs, lr=lr, batch_size=batch_size,
            model_kwargs=model_kwargs,
        )

        forgetting = compute_forgetting(mae_matrix, num_tasks)
        aa  = compute_average_accuracy(mae_matrix, num_tasks)
        bwt = compute_backward_transfer(mae_matrix, num_tasks)
        mf  = float(np.mean(list(forgetting.values())))

        metrics = {"seed": seed, "AA": aa, "BWT": bwt,
                   "mean_forgetting": mf, "forgetting": forgetting,
                   "mae_matrix": mae_matrix}
        all_metrics.append(metrics)

        if verbose:
            print(f"  Seed {seed} — AA: {aa:.4f} | BWT: {bwt:+.4f} | "
                  f"Forgetting: {mf:.4f} dB")

    # Agregar estadísticas
    aa_vals  = [m["AA"]  for m in all_metrics]
    bwt_vals = [m["BWT"] for m in all_metrics]
    mf_vals  = [m["mean_forgetting"] for m in all_metrics]

    summary = {
        "AA":              {"mean": float(np.mean(aa_vals)),  "std": float(np.std(aa_vals))},
        "BWT":             {"mean": float(np.mean(bwt_vals)), "std": float(np.std(bwt_vals))},
        "mean_forgetting": {"mean": float(np.mean(mf_vals)),  "std": float(np.std(mf_vals))},
        "n_seeds": len(seeds),
        "seeds": list(seeds),
    }

    print(f"\n{'█'*60}")
    print(f"  RESUMEN MULTI-SEED ({len(seeds)} seeds: {list(seeds)})")
    print(f"{'█'*60}")
    print(f"  Average Accuracy:  {summary['AA']['mean']:.4f} ± {summary['AA']['std']:.4f} dB")
    print(f"  Backward Transfer: {summary['BWT']['mean']:+.4f} ± {summary['BWT']['std']:.4f} dB")
    print(f"  Forgetting medio:  {summary['mean_forgetting']['mean']:.4f} "
          f"± {summary['mean_forgetting']['std']:.4f} dB")

    return summary, all_metrics


# ============================================================
# UTILIDADES INTERNAS
# ============================================================
def _set_seed_torch(seed):
    """Alias público para fijar seed desde main_pipeline."""
    _set_seed(seed)


def _make_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_ds = IQDataset(X_train, y_train)
    val_ds = IQDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader
