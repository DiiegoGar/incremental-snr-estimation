"""
data_utils.py — Carga de datos, normalización y adaptación cross-domain
========================================================================
Incluye:
  - Carga de RadioML 2016.10a con split estratificado
  - Carga y conversión de WiSig ManyRX
  - 3 estrategias de normalización (z-score, instance, power)
  - Generación de pseudo-labels para WiSig
  - División en tareas incrementales por modulación
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. DATASETS
# ============================================================
class IQDataset(Dataset):
    """Dataset genérico para señales IQ con target de SNR."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 2. CARGA DE RadioML 2016.10a
# ============================================================
def load_radioml(path, seed=42):
    """
    Carga RadioML 2016.10a y divide en train/val/test estratificado.
    
    Returns:
        data_dict: diccionario con X_train, y_snr_train, y_mod_train, etc.
        mods: lista de modulaciones
        snrs: lista de SNRs
        mod_to_idx: diccionario de modulación a índice
    """
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    mods = sorted(list(set(k[0] for k in data.keys())))
    snrs = sorted(list(set(k[1] for k in data.keys())))
    mod_to_idx = {mod: i for i, mod in enumerate(mods)}

    X_list, y_mod_list, y_snr_list = [], [], []
    for (mod, snr), samples in data.items():
        X_list.append(samples)
        y_mod_list.extend([mod_to_idx[mod]] * samples.shape[0])
        y_snr_list.extend([snr] * samples.shape[0])

    X = np.concatenate(X_list, axis=0)
    y_mod = np.array(y_mod_list)
    y_snr = np.array(y_snr_list, dtype=np.float32)

    # Split estratificado 70/15/15
    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []

    for mod in mods:
        for snr in snrs:
            idx = np.where((y_mod == mod_to_idx[mod]) & (y_snr == snr))[0]
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            idx_train.extend(idx[:n_train])
            idx_val.extend(idx[n_train:n_train + n_val])
            idx_test.extend(idx[n_train + n_val:])

    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    idx_test = np.array(idx_test)

    data_dict = {
        "X_train": X[idx_train], "y_snr_train": y_snr[idx_train], "y_mod_train": y_mod[idx_train],
        "X_val": X[idx_val],     "y_snr_val": y_snr[idx_val],     "y_mod_val": y_mod[idx_val],
        "X_test": X[idx_test],   "y_snr_test": y_snr[idx_test],   "y_mod_test": y_mod[idx_test],
    }

    print(f"RadioML cargado: {X.shape[0]} muestras totales")
    print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")
    print(f"  Modulaciones ({len(mods)}): {mods}")
    print(f"  SNRs ({len(snrs)}): {snrs}")

    return data_dict, mods, snrs, mod_to_idx


# ============================================================
# 3. NORMALIZACIÓN (3 estrategias)
# ============================================================
class NormalizationStrategy:
    """
    Encapsula las 3 estrategias de normalización.
    
    MEJORA CLAVE: el modelo ResNetSNR usa InstanceNorm interna,
    por lo que los datos se pasan SIN normalización externa.
    Para el baseline CNN, se usa z-score o power normalization.
    """

    @staticmethod
    def none(X_train, X_val, X_test):
        """Sin normalización externa (para ResNetSNR con InstanceNorm interna)."""
        print("Normalización: NINGUNA (InstanceNorm interna en el modelo)")
        return X_train.copy(), X_val.copy(), X_test.copy(), {}

    @staticmethod
    def zscore(X_train, X_val, X_test):
        """Z-score global basado en estadísticas de train."""
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = np.maximum(X_train.std(axis=(0, 2), keepdims=True), 1e-8)
        stats = {"mean": mean, "std": std}
        print(f"Normalización: Z-SCORE (mean_I={mean[0,0,0]:.6f}, std_I={std[0,0,0]:.6f})")
        return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, stats

    @staticmethod
    def power(X):
        """
        Power normalization per-sample.
        Normaliza cada muestra para que tenga potencia unitaria.
        
        MEJORA #2 alternativa: útil para baseline CNN cross-domain.
        """
        power = np.mean(X ** 2, axis=(1, 2), keepdims=True)
        return X / np.sqrt(power + 1e-8)

    @staticmethod
    def instance(X):
        """
        Instance normalization per-sample en NumPy.
        Equivalente a lo que hace nn.InstanceNorm1d.
        
        MEJORA #2 principal: domain-agnostic, funciona tanto
        para RadioML como WiSig sin parámetros globales.
        """
        # Normalizar cada canal (I/Q) de cada muestra independientemente
        mean = X.mean(axis=2, keepdims=True)   # (N, 2, 1)
        std = np.maximum(X.std(axis=2, keepdims=True), 1e-8)
        return (X - mean) / std


# ============================================================
# 4. TAREAS INCREMENTALES
# ============================================================
# Definición de grupos de modulación por tarea
TASK_MOD_GROUPS = [
    ["BPSK", "QPSK"],
    ["8PSK", "QAM16"],
    ["QAM64", "PAM4"],
    ["CPFSK", "GFSK", "WBFM", "AM-DSB"],
]


def build_task_data(data_dict, mod_to_idx, task_mod_groups=None, normalize="none"):
    """
    Construye los datos por tarea incremental.
    
    Args:
        data_dict: salida de load_radioml()
        mod_to_idx: mapeo modulación → índice
        task_mod_groups: lista de listas de modulaciones por tarea
        normalize: "none" | "zscore" | "power" | "instance"
    
    Returns:
        task_data: lista de diccionarios con datos por tarea
    """
    if task_mod_groups is None:
        task_mod_groups = TASK_MOD_GROUPS

    X_tr = data_dict["X_train"]
    X_va = data_dict["X_val"]
    X_te = data_dict["X_test"]

    # Aplicar normalización
    if normalize == "zscore":
        X_tr, X_va, X_te, stats = NormalizationStrategy.zscore(X_tr, X_va, X_te)
    elif normalize == "power":
        X_tr = NormalizationStrategy.power(X_tr)
        X_va = NormalizationStrategy.power(X_va)
        X_te = NormalizationStrategy.power(X_te)
        stats = {}
        print("Normalización: POWER (per-sample)")
    elif normalize == "instance":
        X_tr = NormalizationStrategy.instance(X_tr)
        X_va = NormalizationStrategy.instance(X_va)
        X_te = NormalizationStrategy.instance(X_te)
        stats = {}
        print("Normalización: INSTANCE (per-sample)")
    else:  # "none"
        stats = {}
        print("Normalización: NINGUNA (InstanceNorm interna en el modelo)")

    y_snr_tr = data_dict["y_snr_train"]
    y_snr_va = data_dict["y_snr_val"]
    y_snr_te = data_dict["y_snr_test"]
    y_mod_tr = data_dict["y_mod_train"]
    y_mod_va = data_dict["y_mod_val"]
    y_mod_te = data_dict["y_mod_test"]

    task_data = []
    for task_id, mod_group in enumerate(task_mod_groups, 1):
        mod_indices = [mod_to_idx[m] for m in mod_group]
        tr_mask = np.isin(y_mod_tr, mod_indices)
        va_mask = np.isin(y_mod_va, mod_indices)
        te_mask = np.isin(y_mod_te, mod_indices)

        task_info = {
            "task_id": task_id,
            "mods": mod_group,
            "X_train": X_tr[tr_mask].astype(np.float32),
            "y_train": y_snr_tr[tr_mask].astype(np.float32),
            "X_val": X_va[va_mask].astype(np.float32),
            "y_val": y_snr_va[va_mask].astype(np.float32),
            "X_test": X_te[te_mask].astype(np.float32),
            "y_test": y_snr_te[te_mask].astype(np.float32),
            "y_mod_test": y_mod_te[te_mask],
        }
        task_data.append(task_info)
        print(f"  Tarea {task_id}: {mod_group} — "
              f"Train {task_info['X_train'].shape[0]} | "
              f"Val {task_info['X_val'].shape[0]} | "
              f"Test {task_info['X_test'].shape[0]}")

    return task_data, stats


# ============================================================
# 5. CARGA Y CONVERSIÓN DE WiSig ManyRX
# ============================================================
def load_wisig_manyrx(pkl_path, use_equalized_idx=0, target_length=128):
    """
    Carga WiSig ManyRX y convierte a formato IQ (2, target_length).
    
    Cada señal original es (256, 2). Se divide en ventanas de target_length
    para ser compatible con el modelo entrenado en RadioML.
    
    Returns:
        X: array (N, 2, target_length) float32
        meta: lista de tuplas (tx, rx, date, eq, sample, window)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    raw = data["data"]
    X_all, meta_all = [], []
    n_tx = len(raw)

    for tx_idx in range(n_tx):
        n_rx = len(raw[tx_idx])
        for rx_idx in range(n_rx):
            block = raw[tx_idx][rx_idx]
            for date_idx in range(len(block)):
                eq_block = block[date_idx]
                samples = np.array(eq_block[use_equalized_idx])  # (200, 256, 2)

                for sample_idx in range(samples.shape[0]):
                    sig = samples[sample_idx]  # (256, 2)
                    sig_iq = sig.T  # (2, 256)

                    # Dividir en ventanas de target_length
                    n_windows = sig_iq.shape[1] // target_length
                    for w in range(n_windows):
                        start = w * target_length
                        end = start + target_length
                        window = sig_iq[:, start:end]
                        X_all.append(window)
                        meta_all.append((tx_idx, rx_idx, date_idx,
                                         use_equalized_idx, sample_idx, w))

    X = np.array(X_all, dtype=np.float32)
    print(f"WiSig ManyRX cargado: {X.shape[0]} ventanas de shape {X.shape[1:]}")
    print(f"  Transmisores: {n_tx} | Receptores: {n_rx}")
    return X, meta_all, data


# ============================================================
# 6. PSEUDO-LABELING PARA WiSig (MEJORA #4)
# ============================================================
def generate_wisig_pseudo_labels(X_wisig, snr_levels=None, seed=42):
    """
    Genera pseudo-labels para WiSig mediante degradación controlada con AWGN.
    
    MEJORA CLAVE: permite usar WiSig como Tarea 5 de adaptación incremental.
    
    La idea: si degradamos una señal limpia con X dB de ruido, sabemos que
    el SNR resultante es aproximadamente X dB (relativo a la potencia original).
    
    Args:
        X_wisig: array (N, 2, L) señales WiSig normalizadas por potencia
        snr_levels: lista de SNR a generar, ej. [-10, -5, 0, 5, 10, 15]
        seed: semilla de aleatoriedad
    
    Returns:
        X_pseudo: array con señales degradadas
        y_pseudo: array con pseudo-SNR labels
    """
    if snr_levels is None:
        snr_levels = [-10, -5, 0, 5, 10, 15]

    rng = np.random.default_rng(seed)
    X_pseudo_list, y_pseudo_list = [], []

    # Seleccionar subconjunto representativo (para eficiencia)
    n_per_level = min(5000, len(X_wisig))
    indices = rng.choice(len(X_wisig), n_per_level, replace=False)
    X_subset = X_wisig[indices]

    for snr_db in snr_levels:
        signal_power = np.mean(X_subset ** 2, axis=(1, 2), keepdims=True)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = rng.standard_normal(X_subset.shape).astype(np.float32)
        noise = noise * np.sqrt(noise_power + 1e-8)

        X_noisy = (X_subset + noise).astype(np.float32)
        X_pseudo_list.append(X_noisy)
        y_pseudo_list.append(np.full(len(X_noisy), snr_db, dtype=np.float32))

    X_pseudo = np.concatenate(X_pseudo_list, axis=0)
    y_pseudo = np.concatenate(y_pseudo_list, axis=0)

    # Shuffle
    perm = rng.permutation(len(X_pseudo))
    X_pseudo = X_pseudo[perm]
    y_pseudo = y_pseudo[perm]

    print(f"Pseudo-labels generadas: {len(X_pseudo)} muestras")
    print(f"  SNR levels: {snr_levels}")
    print(f"  Muestras por nivel: {n_per_level}")

    return X_pseudo, y_pseudo


def add_awgn(X, snr_db, seed=42):
    """Añade ruido AWGN a un conjunto de señales con un SNR dado."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(X ** 2, axis=(1, 2), keepdims=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.standard_normal(X.shape).astype(np.float32)
    noise = noise * np.sqrt(noise_power + 1e-8)
    return (X + noise).astype(np.float32)


# ============================================================
# 7. DATALOADERS RÁPIDOS
# ============================================================
def make_loaders(X_train, y_train, X_val, y_val, batch_size=256):
    """Crea DataLoaders de train y val."""
    train_ds = IQDataset(X_train, y_train)
    val_ds = IQDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader
