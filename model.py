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

    Arquitectura: 4 bloques residuales [64, 128, 256, 256] (~350K parámetros).
    """
    def __init__(self, use_instance_norm=True, dropout=0.3,
                 norm="group", num_groups=8):
        super().__init__()
        self.use_instance_norm = use_instance_norm

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

        # Global Average Pooling + Head de regresión
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Inicialización Kaiming
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        """Extrae feature vector de 256-dim (para Feature-level KD)."""
        if self.use_instance_norm:
            x = self.input_norm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(-1)  # (B, 256)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        return self.regressor(features).squeeze(1)


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
