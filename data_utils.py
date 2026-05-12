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


class IQDatasetMixed(Dataset):
    """
    Dataset que combina muestras de la tarea actual con muestras de replay,
    etiquetando cada muestra con un flag is_replay.

    El flag permite aplicar KD solo sobre muestras antiguas (replay), evitando
    que el teacher OOD interfiera en el aprendizaje de tareas nuevas.
    """
    def __init__(self, X_task, y_task, X_replay=None, y_replay=None):
        if X_replay is not None and len(X_replay) > 0:
            X_all = np.concatenate([X_task, X_replay], axis=0)
            y_all = np.concatenate([y_task, y_replay], axis=0)
            flags = np.concatenate([
                np.zeros(len(X_task), dtype=np.float32),
                np.ones(len(X_replay), dtype=np.float32),
            ])
        else:
            X_all = X_task
            y_all = y_task
            flags = np.zeros(len(X_task), dtype=np.float32)

        self.X = torch.tensor(X_all, dtype=torch.float32)
        self.y = torch.tensor(y_all, dtype=torch.float32)
        self.is_replay = torch.tensor(flags, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.is_replay[idx]


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
def generate_wisig_pseudo_labels(X_wisig, snr_levels=None, seed=42,
                                  n_per_level=5000):
    """
    Genera pseudo-labels para WiSig mediante degradación controlada con AWGN.

    ATENCIÓN SEMÁNTICA: las etiquetas devueltas son el SNR del AWGN AÑADIDO
    sobre la señal previamente normalizada a potencia unitaria, NO el SNR
    físico real de la trama WiFi. El modelo entrenado con estas etiquetas
    aprende a estimar "intensidad de AWGN sintético", no SNR físico real;
    interpretar el MAE resultante como SNR físico es incorrecto. El alias
    semántico `generate_wisig_snr_awgn_labels` deja esto explícito en el
    código que llama.

    Metodología:
      1. Normalizar cada señal WiSig a potencia unitaria (P=1).
         Esto alinea la definición de SNR con RadioML, donde las muestras
         también tienen potencia normalizada y SNR = nivel de AWGN añadido.
      2. Añadir AWGN con potencia 1/SNR_target_linear.
         Con P_señal_norm = 1, el SNR resultante es exactamente SNR_target.

    Sin la normalización previa, X_wisig contiene señal + ruido de canal, y
    signal_power mediría P_señal + P_ruido_canal, haciendo que el label asignado
    no coincida con el SNR total real de la señal degradada.

    Args:
        X_wisig:    array (N, 2, L) señales WiSig en bruto
        snr_levels: lista de SNR objetivo en dB, ej. [-10, -5, 0, 5, 10, 15]
        seed:       semilla de aleatoriedad

    Returns:
        X_pseudo: array (N_total, 2, L) señales degradadas normalizadas
        y_pseudo: array (N_total,) etiquetas SNR_AWGN_añadido en dB
    """
    if snr_levels is None:
        snr_levels = [-10, -5, 0, 5, 10, 15]

    rng = np.random.default_rng(seed)
    X_pseudo_list, y_pseudo_list = [], []

    # Seleccionar subconjunto representativo
    n_per_level = min(n_per_level, len(X_wisig))
    indices = rng.choice(len(X_wisig), n_per_level, replace=False)
    X_subset = X_wisig[indices].copy()

    # Paso 1 — normalizar a potencia unitaria por muestra.
    # Tras esto P_señal_norm ≈ 1.0, igual que en RadioML.
    rms = np.sqrt(np.mean(X_subset ** 2, axis=(1, 2), keepdims=True) + 1e-8)
    X_subset = X_subset / rms

    for snr_db in snr_levels:
        # Paso 2 — añadir AWGN con potencia exacta 1/SNR_linear.
        # Con P_señal=1 → SNR_total = 1 / noise_power = SNR_target  ✓
        snr_linear  = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise = rng.standard_normal(X_subset.shape).astype(np.float32)
        noise = noise * np.sqrt(noise_power)

        X_noisy = (X_subset + noise).astype(np.float32)
        X_pseudo_list.append(X_noisy)
        y_pseudo_list.append(np.full(len(X_noisy), snr_db, dtype=np.float32))

    X_pseudo = np.concatenate(X_pseudo_list, axis=0)
    y_pseudo = np.concatenate(y_pseudo_list, axis=0)

    perm = rng.permutation(len(X_pseudo))
    X_pseudo = X_pseudo[perm]
    y_pseudo = y_pseudo[perm]

    print(f"Pseudo-labels generadas: {len(X_pseudo)} muestras")
    print(f"  SNR levels: {snr_levels}")
    print(f"  Muestras por nivel: {n_per_level}")
    print(f"  Normalización: potencia unitaria por muestra antes de añadir AWGN")

    return X_pseudo, y_pseudo


# Alias semántico: deja explícito que la etiqueta es "SNR de AWGN añadido",
# no SNR físico real de la trama WiFi. Útil al usar este pipeline en scripts
# nuevos o en la memoria escrita.
def generate_wisig_snr_awgn_labels(X_wisig, snr_levels=None, seed=42,
                                    n_per_level=5000):
    """Alias de generate_wisig_pseudo_labels con nomenclatura explícita.

    Returns:
        X_pseudo:           señales con AWGN sintético añadido
        y_snr_awgn_added_db: etiquetas del SNR del AWGN añadido (en dB)
    """
    return generate_wisig_pseudo_labels(
        X_wisig, snr_levels=snr_levels, seed=seed, n_per_level=n_per_level
    )


def split_wisig_adapt_eval(X_wisig, meta_wisig=None, adapt_frac=0.5, seed=42):
    """
    Divide WiSig en dos conjuntos estrictamente separados:
      - X_adapt: para generación de pseudo-labels y entrenamiento (Tarea 5)
      - X_eval:  held-out, NUNCA visto durante entrenamiento → solo métricas físicas

    Esto elimina el bucle circular donde las mismas muestras usadas para
    generar pseudo-labels se usan también para evaluar el modelo.

    Returns:
        X_adapt, meta_adapt, X_eval, meta_eval
    """
    rng = np.random.default_rng(seed)
    n = len(X_wisig)
    idx = rng.permutation(n)
    n_adapt = int(adapt_frac * n)
    idx_adapt = idx[:n_adapt]
    idx_eval  = idx[n_adapt:]

    X_adapt = X_wisig[idx_adapt]
    X_eval  = X_wisig[idx_eval]
    meta_adapt = [meta_wisig[i] for i in idx_adapt] if meta_wisig is not None else None
    meta_eval  = [meta_wisig[i] for i in idx_eval]  if meta_wisig is not None else None

    print(f"WiSig partido: {n_adapt} adapt | {len(idx_eval)} eval (held-out, nunca visto en T5)")
    return X_adapt, meta_adapt, X_eval, meta_eval


def add_awgn(X, snr_db, seed=42, normalize_first=True):
    """
    Añade AWGN a un conjunto de señales para un SNR objetivo en dB.

    Args:
        normalize_first: si True (por defecto), normaliza cada señal a
                         potencia unitaria antes de añadir el ruido, garantizando
                         que el SNR resultante coincide exactamente con snr_db.
                         Usar False solo si X ya tiene potencia unitaria garantizada.
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy().astype(np.float32)

    if normalize_first:
        rms = np.sqrt(np.mean(X_out ** 2, axis=(1, 2), keepdims=True) + 1e-8)
        X_out = X_out / rms

    snr_linear  = 10 ** (snr_db / 10.0)
    noise_power = 1.0 / snr_linear          # P_señal = 1 tras normalización
    noise = rng.standard_normal(X_out.shape).astype(np.float32)
    noise = noise * np.sqrt(noise_power)
    return (X_out + noise).astype(np.float32)


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
