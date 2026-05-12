"""
model.py — Arquitecturas para estimación de SNR
================================================
Incluye:
  - SNREstimatorCNN: modelo baseline original (para comparación)
  - ResidualBlock1D: bloque residual con BatchNorm
  - ResNetSNR: modelo mejorado con skip connections + InstanceNorm de entrada
"""

import torch
import torch.nn as nn


# ============================================================
# 1. MODELO BASELINE ORIGINAL (sin cambios, para comparación)
# ============================================================
class SNREstimatorCNN(nn.Module):
    """CNN baseline del notebook original."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)

    def extract_features(self, x):
        """Extrae features intermedias (para Feature-level KD)."""
        return self.features(x).squeeze(-1)


# ============================================================
# 2. MODELO MEJORADO: ResNet-1D-SNR
# ============================================================
def _make_norm(norm, num_channels, num_groups=8):
    """
    Fábrica de capas de normalización.

    BatchNorm acumula running_mean/var durante el entrenamiento, lo que
    contamina las estadísticas entre tareas en CL. GroupNorm no tiene
    estado acumulado: normaliza por grupos de canales dentro de cada muestra,
    siendo independiente del historial de tareas anteriores.
    """
    if norm == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=True)
    return nn.BatchNorm1d(num_channels)


class ResidualBlock1D(nn.Module):
    """Bloque residual 1D con skip connection y normalización configurable."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 norm="group", num_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm1 = _make_norm(norm, out_channels, num_groups)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.norm2 = _make_norm(norm, out_channels, num_groups)
        self.relu  = nn.ReLU(inplace=True)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                _make_norm(norm, out_channels, num_groups),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetSNR(nn.Module):
    """
    ResNet-1D para estimación de SNR.

    Normalización:
      - InstanceNorm1d en la entrada (domain-agnostic, sin running stats)
      - GroupNorm(num_groups=8) en bloques residuales y stem, en lugar de
        BatchNorm. BatchNorm acumula running_mean/var que se contaminan entre
        tareas en CL; GroupNorm normaliza por grupos de canales dentro de cada
        muestra sin estado acumulado.

    Multi-head:
      - self.heads es un nn.ModuleDict con un head por dominio.
      - El head por defecto se llama "radio" y reemplaza al antiguo self.regressor.
      - set_active_head(name) cambia el head usado en forward; add_head(name)
        crea un head nuevo (típicamente para un dominio cross-domain como WiSig).
      - La propiedad self.regressor sigue funcionando como alias de read-only
        del head "radio" para no romper código que lo lea (no se registra en
        state_dict, así que no hay duplicación).
    """
    def __init__(self, use_instance_norm=True, dropout=0.3,
                 norm="group", num_groups=8, feature_dim=256, head_hidden=128,
                 use_physical_features=False):
        super().__init__()
        self.use_instance_norm = use_instance_norm
        self.use_physical_features = use_physical_features
        # Si las features físicas están activas, se concatenan 4 floats
        # (log_M2, kurt, M2, M4) al final del vector de features. Los heads
        # se dimensionan a partir de este total.
        self._phys_dim   = 4 if use_physical_features else 0
        self.feature_dim = feature_dim
        self.total_feat  = feature_dim + self._phys_dim
        self._head_hidden = head_hidden
        self._head_dropout = dropout

        if use_instance_norm:
            self.input_norm = nn.InstanceNorm1d(2, affine=True)

        # GroupNorm en el stem (num_groups debe dividir a 64 → 8 grupos de 8)
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            _make_norm(norm, 64, num_groups),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.layer1 = ResidualBlock1D(64,  64,  kernel_size=5, stride=1, norm=norm, num_groups=num_groups)
        self.layer2 = ResidualBlock1D(64,  128, kernel_size=5, stride=2, norm=norm, num_groups=num_groups)
        self.layer3 = ResidualBlock1D(128, 256, kernel_size=3, stride=2, norm=norm, num_groups=num_groups)
        self.layer4 = ResidualBlock1D(256, 256, kernel_size=3, stride=2, norm=norm, num_groups=num_groups)

        # Global Average Pooling + ModuleDict de heads (al menos uno: "radio")
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleDict({"radio": self._build_head(self.total_feat)})
        self.active_head = "radio"

        # Inicialización Kaiming
        self._init_weights()

    def _build_head(self, in_dim):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, self._head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self._head_dropout),
            nn.Linear(self._head_hidden, 1),
        )

    @staticmethod
    def _physical_features(x):
        """
        4 features físicas INVARIANTES A ESCALA, calculadas sobre la señal
        sin normalizar. Invariantes también a rotación de fase y permutación
        temporal.

        Motivación: en este pipeline RadioML tiene potencia media ~3e-5 y
        WiSig ~5e-4 (≈15× distinta). Features absolutas (log_M2, M2, M4)
        introducirían un shift cross-domain de >20 σ — exactamente el
        problema que InstanceNorm intenta resolver. Estas 4 features
        invariantes a escala muestran shifts < 1 σ entre ambos dominios
        (medido empíricamente sobre 5500 RadioML vs 2400 WiSig):

            - kurt    = M4 / M2²                    (forma envelope:
                                                     PSK≈1, OFDM≈2, QAM 1.3-1.6)
            - peak2avg = max(|x|²) / M2              (peak-to-avg power)
            - cv      = std(|x|²) / M2               (coef. variación envelope)
            - outlier_frac = mean(|x|² > 2·M2)       (fracción de outliers)
        """
        EPS = 1e-12
        p   = (x[:, 0] ** 2 + x[:, 1] ** 2)              # (B, L)
        M2  = p.mean(dim=1, keepdim=True).clamp_min(EPS) # (B, 1) protegido
        M4  = (p ** 2).mean(dim=1, keepdim=True)
        p_max = p.max(dim=1, keepdim=True).values
        p_std = (p - M2).pow(2).mean(dim=1, keepdim=True).sqrt()
        outlier = (p > 2.0 * M2).float().mean(dim=1, keepdim=True)
        kurt     = M4    / (M2 ** 2)
        peak2avg = p_max / M2
        cv       = p_std / M2
        return torch.cat([kurt, peak2avg, cv, outlier], dim=1)  # (B, 4)

    @property
    def regressor(self):
        # Alias read-only para código legacy. No es un atributo registrado:
        # no aparece en state_dict ni se itera en parameters() dos veces.
        return self.heads["radio"]

    def add_head(self, name, hidden=None, dropout=None):
        """Crea un head nuevo (típicamente para un dominio cross-domain).

        Se hereda automáticamente device y dtype del modelo padre para que
        funcione tanto si se llama antes como después de model.to(device).
        """
        if name in self.heads:
            raise ValueError(f"Head '{name}' ya existe")
        if hidden is None:
            hidden = self._head_hidden
        if dropout is None:
            dropout = self._head_dropout
        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.total_feat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Inicialización Kaiming explícita para el head nuevo
        nn.init.kaiming_normal_(head[1].weight)
        nn.init.constant_(head[1].bias, 0)
        nn.init.kaiming_normal_(head[4].weight)
        nn.init.constant_(head[4].bias, 0)
        # Heredar device/dtype del modelo padre. Si el modelo todavía no
        # tiene parámetros (caso extremo), se queda en su default.
        try:
            ref = next(self.parameters())
            head = head.to(device=ref.device, dtype=ref.dtype)
        except StopIteration:
            pass
        self.heads[name] = head

    def set_active_head(self, name):
        """name=None ó "radio" → head original; cualquier otro nombre debe existir en self.heads."""
        target = "radio" if name is None else name
        if target not in self.heads:
            raise KeyError(f"Head '{target}' no existe. Disponibles: {list(self.heads.keys())}")
        self.active_head = target

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        """
        Extrae feature vector de total_feat dim (para Feature-level KD).
        Si use_physical_features=True, concatena 4 features físicas
        calculadas sobre la señal sin normalizar al final del vector.
        """
        if self.use_physical_features:
            phys = self._physical_features(x)  # (B, 4), antes de InstanceNorm
        if self.use_instance_norm:
            x = self.input_norm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(-1)  # (B, feature_dim)
        if self.use_physical_features:
            x = torch.cat([x, phys], dim=1)   # (B, feature_dim + 4)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        return self.heads[self.active_head](features).squeeze(1)


# ============================================================
# 3. UTILIDADES
# ============================================================
def count_parameters(model):
    """Cuenta parámetros entrenables."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def model_summary(model, input_shape=(2, 128)):
    """Imprime resumen del modelo con conteo de parámetros."""
    print(f"{'='*60}")
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    print(f"{'='*60}")
    print(model)
    # Test forward pass
    x = torch.randn(1, *input_shape)
    with torch.no_grad():
        out = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output value: {out.item():.4f}")
    return model


if __name__ == "__main__":
    print("=== BASELINE ===")
    model_summary(SNREstimatorCNN())
    print("\n=== RESNET-SNR ===")
    model_summary(ResNetSNR())
