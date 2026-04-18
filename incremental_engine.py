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
    Replay buffer con dos modos de selección:
      - random: selección aleatoria (original)
      - herding: selección por cercanía al centroide de features
    
    MEJORA: Herding selecciona ejemplares cuya media de features
    es más representativa del cluster, mejorando la retención
    con el mismo presupuesto de memoria.
    """
    def __init__(self, capacity, selection="herding"):
        self.capacity = capacity
        self.selection = selection
        self.X = None
        self.y = None

    def __len__(self):
        return 0 if self.X is None else len(self.X)

    def add_examples(self, X_new, y_new, model=None, device=None):
        """
        Añade ejemplos al buffer.
        Si selection='herding' y model está disponible, selecciona
        los ejemplos más representativos por cercanía al centroide.
        """
        if len(X_new) == 0:
            return

        if self.X is None:
            self.X = np.copy(X_new)
            self.y = np.copy(y_new)
        else:
            self.X = np.concatenate([self.X, X_new], axis=0)
            self.y = np.concatenate([self.y, y_new], axis=0)

        # Recortar si excede capacidad
        if len(self.X) > self.capacity:
            if self.selection == "herding" and model is not None and device is not None:
                idx = self._herding_select(model, device)
            else:
                idx = np.random.choice(len(self.X), self.capacity, replace=False)
            self.X = self.X[idx]
            self.y = self.y[idx]

    def _herding_select(self, model, device):
        """
        Herding selection: selecciona los ejemplos cuya media acumulada
        de features se acerca más al centroide global.
        
        Inspirado en iCaRL (Rebuffi et al., 2017).
        """
        model.eval()
        # Extraer features de todos los ejemplos en el buffer
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        features_list = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 256):
                batch = X_tensor[i:i+256].to(device)
                feat = model.extract_features(batch)
                features_list.append(feat.cpu().numpy())
        features = np.concatenate(features_list, axis=0)

        # Centroide global
        centroid = features.mean(axis=0)

        # Selección greedy de herding
        selected = []
        selected_sum = np.zeros_like(centroid)
        remaining = set(range(len(features)))

        for k in range(min(self.capacity, len(features))):
            best_idx = -1
            best_dist = float('inf')
            target = (k + 1) * centroid  # suma ideal tras k+1 selecciones

            for idx in remaining:
                candidate_sum = selected_sum + features[idx]
                dist = np.linalg.norm(candidate_sum - target)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            selected.append(best_idx)
            selected_sum += features[best_idx]
            remaining.discard(best_idx)

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
    
    Calcula la Fisher Information Matrix diagonal tras cada tarea
    y penaliza cambios en parámetros importantes.
    
    MEJORA: complementa Replay + KD con regularización basada
    en importancia de parámetros. Los tres mecanismos son ortogonales.
    """
    def __init__(self, model, lambda_ewc=1000.0):
        self.lambda_ewc = lambda_ewc
        self.params = {}
        self.fisher = {}

    def register_task(self, model, data_loader, device, n_samples=2000):
        """
        Calcula la Fisher Information Matrix diagonal para la tarea actual.
        Debe llamarse DESPUÉS de entrenar cada tarea.
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

        # Promediar y acumular con Fisher de tareas anteriores
        for n in fisher_diag:
            fisher_diag[n] /= count
            if n in self.fisher:
                self.fisher[n] = self.fisher[n] + fisher_diag[n]
            else:
                self.fisher[n] = fisher_diag[n].clone()

        # Guardar parámetros óptimos actuales
        self.params = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}

    def penalty(self, model):
        """Calcula la penalización EWC."""
        if not self.params:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n].to(p.device) * (p - self.params[n].to(p.device)) ** 2).sum()
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
        replay_buffer.add_examples(X_current, y_current, model=model, device=device)
        print(f"  Buffer size: {len(replay_buffer)}")

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
# UTILIDADES INTERNAS
# ============================================================
def _make_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_ds = IQDataset(X_train, y_train)
    val_ds = IQDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader
