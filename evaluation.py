"""
evaluation.py — Evaluación cross-domain y métricas de calibración
================================================================
Incluye:
  - Predicción sobre WiSig sin labels
  - Test de degradación controlada con métricas de monotonicidad
  - Análisis por receptor, transmisor y fecha
  - Calibración lineal post-hoc
  - Validación con estimador clásico M2M4
  - Spearman rank correlation
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats


# ============================================================
# 1. PREDICCIÓN SOBRE DATOS SIN LABELS
# ============================================================
def predict_unlabeled(model, X, device, batch_size=256):
    """
    Genera predicciones de SNR sobre datos sin etiqueta.
    
    Args:
        model: modelo entrenado
        X: array (N, 2, L) señales IQ
        device: torch device
    
    Returns:
        preds: array (N,) predicciones de SNR en dB
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dummy_y = torch.zeros(len(X_tensor))
    loader = DataLoader(TensorDataset(X_tensor, dummy_y),
                        batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            y_hat = model(X_batch)
            preds.append(y_hat.squeeze().cpu().numpy())

    return np.concatenate(preds)


# ============================================================
# 2. TEST DE DEGRADACIÓN CONTROLADA (MEJORA 5.2)
# ============================================================
def degradation_test(model, X, device, snr_levels=None, seed=42):
    """
    Evalúa la coherencia del modelo añadiendo ruido AWGN progresivo.
    
    Si el modelo es coherente:
      - Más ruido → SNR estimado más bajo
      - La curva debe ser monótona decreciente
      - Idealmente ΔSNR_estimado ≈ ΔSNR_ruido
    
    Returns:
        results: dict con predicciones, medias, stds, y métricas de calibración
    """
    if snr_levels is None:
        snr_levels = [None, 15, 10, 5, 0, -5]  # None = sin ruido adicional

    rng = np.random.default_rng(seed)
    results = {"levels": [], "means": [], "stds": [], "preds": {}}

    for snr_db in snr_levels:
        label = "clean" if snr_db is None else f"{snr_db}dB"

        if snr_db is None:
            X_input = X
        else:
            signal_power = np.mean(X ** 2, axis=(1, 2), keepdims=True)
            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise = rng.standard_normal(X.shape).astype(np.float32)
            noise = noise * np.sqrt(noise_power + 1e-8)
            X_input = (X + noise).astype(np.float32)

        preds = predict_unlabeled(model, X_input, device)
        results["levels"].append(label)
        results["means"].append(preds.mean())
        results["stds"].append(preds.std())
        results["preds"][label] = preds

    # Métricas de calibración
    numeric_levels = [0] + [s for s in snr_levels if s is not None]
    mean_preds = results["means"]

    # Monotonicidad: Spearman entre nivel de ruido y SNR estimado
    # Más ruido (menor snr_db) → menor SNR estimado → correlación positiva
    # entre snr_db y mean prediction
    snr_values = [100] + [s for s in snr_levels if s is not None]  # 100 for clean
    rho, p_value = stats.spearmanr(snr_values, mean_preds)
    results["spearman_rho"] = rho
    results["spearman_p"] = p_value

    # Sensibilidad: ΔSNR estimado / ΔSNR ruido
    if len(mean_preds) >= 2:
        clean_pred = mean_preds[0]
        sensitivities = []
        for i, snr_db in enumerate(snr_levels):
            if snr_db is not None and i > 0:
                delta_noise = 0 - snr_db  # degradación desde "clean"
                delta_pred = clean_pred - mean_preds[i]
                if abs(delta_noise) > 0:
                    sensitivities.append(delta_pred / delta_noise)
        results["mean_sensitivity"] = np.mean(sensitivities) if sensitivities else 0.0
    else:
        results["mean_sensitivity"] = 0.0

    return results


def print_degradation_results(results):
    """Imprime los resultados del test de degradación."""
    print(f"\n{'='*55}")
    print(f"  TEST DE DEGRADACIÓN CONTROLADA")
    print(f"{'='*55}")
    print(f"  {'Nivel':<10} {'SNR Estimado (dB)':<20} {'Std (dB)':<12}")
    print(f"  {'-'*42}")
    for level, mean, std in zip(results["levels"], results["means"], results["stds"]):
        print(f"  {level:<10} {mean:>8.2f}              {std:>6.2f}")
    print(f"\n  Spearman ρ: {results['spearman_rho']:.4f} "
          f"(p={results['spearman_p']:.2e})")
    print(f"  Sensibilidad media (ΔSNR_pred/ΔSNR_ruido): "
          f"{results['mean_sensitivity']:.3f}")
    print(f"  {'(Ideal ≈ 1.0)':>45}")


# ============================================================
# 3. ANÁLISIS POR RECEPTOR / TRANSMISOR / FECHA
# ============================================================
def analyze_by_metadata(preds, meta, field_idx, field_name):
    """
    Analiza predicciones agrupadas por un campo de metadata.
    
    Args:
        preds: array de predicciones
        meta: lista de tuplas de metadata
        field_idx: índice del campo en la tupla (0=tx, 1=rx, 2=date)
        field_name: nombre del campo para el print
    
    Returns:
        analysis: dict {field_value: {"mean": ..., "std": ..., "count": ...}}
    """
    groups = {}
    for pred, m in zip(preds, meta):
        key = m[field_idx]
        if key not in groups:
            groups[key] = []
        groups[key].append(pred)

    analysis = {}
    print(f"\n  Análisis por {field_name}:")
    print(f"  {'ID':<6} {'Media (dB)':<14} {'Std (dB)':<12} {'N':<8}")
    print(f"  {'-'*40}")
    for key in sorted(groups.keys()):
        vals = np.array(groups[key])
        analysis[key] = {"mean": vals.mean(), "std": vals.std(), "count": len(vals)}
        print(f"  {key:<6} {vals.mean():>8.2f}       {vals.std():>6.2f}       {len(vals)}")

    # Consistencia inter-grupo: std de las medias
    means = [v["mean"] for v in analysis.values()]
    print(f"\n  Consistencia inter-{field_name}: std de medias = {np.std(means):.3f} dB")

    return analysis


# ============================================================
# 4. CALIBRACIÓN LINEAL POST-HOC (MEJORA 4.2 Nivel 3)
# ============================================================
def calibrate_linear(preds_calibration, snr_true_calibration, preds_target):
    """
    Calibración lineal: ajusta offset y escala de predicciones.
    
    Requiere un pequeño conjunto con SNR conocido (ej. degradación controlada).
    
    Args:
        preds_calibration: predicciones del modelo en datos con SNR conocido
        snr_true_calibration: SNR real de esos datos
        preds_target: predicciones a calibrar
    
    Returns:
        preds_calibrated: predicciones corregidas
        slope: pendiente del ajuste
        intercept: offset del ajuste
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        preds_calibration, snr_true_calibration
    )
    preds_calibrated = slope * preds_target + intercept

    print(f"\n  Calibración lineal:")
    print(f"    SNR_calibrado = {slope:.4f} * SNR_raw + {intercept:.4f}")
    print(f"    R² = {r_value**2:.4f}")
    print(f"    Error estándar: {std_err:.4f}")

    return preds_calibrated, slope, intercept


# ============================================================
# 5. ESTIMADOR CLÁSICO M2M4 (MEJORA 5.2)
# ============================================================
def m2m4_estimator(X):
    """
    Estimador de SNR basado en momentos M2 y M4 (Pauluzzi & Beaulieu, 2000).
    
    Asume señal PSK/QAM con distribución de envolvente conocida.
    SNR_est = M2² / (M4 - M2²)   (simplificado para señales de envolvente constante)
    
    Args:
        X: array (N, 2, L) señales IQ
    
    Returns:
        snr_estimates: array (N,) en dB
    """
    # Calcular envolvente compleja: |s|² = I² + Q²
    power = X[:, 0, :] ** 2 + X[:, 1, :] ** 2  # (N, L)
    M2 = np.mean(power, axis=1)       # segundo momento
    M4 = np.mean(power ** 2, axis=1)  # cuarto momento

    # SNR estimado (fórmula para señales de envolvente constante tipo PSK)
    # Para modulaciones reales es una aproximación
    kurtosis = M4 / (M2 ** 2 + 1e-12)
    # Para AWGN puro, kurtosis ≈ 2; para señal pura, kurtosis ≈ 1
    # SNR ≈ 1 / (kurtosis - 1) en lineal
    snr_linear = np.clip(1.0 / (kurtosis - 1.0 + 1e-12), 1e-6, 1e6)
    snr_db = 10 * np.log10(snr_linear)

    return snr_db


def compare_with_m2m4(model_preds, X, meta=None):
    """
    Compara predicciones del modelo DL con el estimador M2M4.
    
    Returns:
        correlation: Pearson r entre ambos estimadores
    """
    m2m4_preds = m2m4_estimator(X)

    # Filtrar valores extremos
    mask = np.isfinite(m2m4_preds) & (np.abs(m2m4_preds) < 50)
    m2m4_filtered = m2m4_preds[mask]
    model_filtered = model_preds[mask]

    r, p = stats.pearsonr(model_filtered, m2m4_filtered)
    rho, _ = stats.spearmanr(model_filtered, m2m4_filtered)

    print(f"\n  Comparación DL vs M2M4:")
    print(f"    Pearson r:  {r:.4f} (p={p:.2e})")
    print(f"    Spearman ρ: {rho:.4f}")
    print(f"    DL  — Media: {model_preds.mean():.2f} dB, Std: {model_preds.std():.2f} dB")
    print(f"    M2M4— Media: {m2m4_filtered.mean():.2f} dB, Std: {m2m4_filtered.std():.2f} dB")

    return {"pearson_r": r, "spearman_rho": rho,
            "dl_preds": model_preds, "m2m4_preds": m2m4_preds}
