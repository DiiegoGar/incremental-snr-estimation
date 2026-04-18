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
class ResidualBlock1D(nn.Module):
    """Bloque residual 1D con skip connection y BatchNorm."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection con proyección si cambian las dimensiones
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetSNR(nn.Module):
    """
    ResNet-1D para estimación de SNR.
    
    MEJORA #1: InstanceNorm de entrada → elimina dependencia de estadísticas
               globales del dataset de entrenamiento. Resuelve el domain mismatch
               entre RadioML y WiSig.
    
    MEJORA #3: Skip connections → gradientes más estables, features más ricas.
               4 bloques residuales [64, 128, 256, 256].
    
    Parámetros: ~350K (vs ~45K del baseline).
    """
    def __init__(self, use_instance_norm=True, dropout=0.3):
        super().__init__()
        self.use_instance_norm = use_instance_norm

        # MEJORA #1: Instance Normalization por muestra
        # Normaliza cada muestra IQ individualmente → domain-agnostic
        if use_instance_norm:
            self.input_norm = nn.InstanceNorm1d(2, affine=True)

        # Stem: primera convolución amplia para capturar patrones espectrales
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 128 → 64
        )

        # 4 bloques residuales con reducción progresiva
        self.layer1 = ResidualBlock1D(64, 64, kernel_size=5, stride=1)     # 64
        self.layer2 = ResidualBlock1D(64, 128, kernel_size=5, stride=2)    # 64 → 32
        self.layer3 = ResidualBlock1D(128, 256, kernel_size=3, stride=2)   # 32 → 16
        self.layer4 = ResidualBlock1D(256, 256, kernel_size=3, stride=2)   # 16 → 8

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
