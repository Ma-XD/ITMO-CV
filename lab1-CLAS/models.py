from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from config import (
    ALL_MODELS,
    MODEL_CUSTOM,
    MODEL_MOBILENETV3,
    MODEL_RESNET18,
    NUM_CLASSES,
    PRETRAINED_MODELS,
)


class BasicBlock(nn.Module):
    """Residual-блок ResNet-18/34: y = ReLU(BN(Conv) + BN(Conv(BN(Conv(x))))) + shortcut.

    Shortcut спасает от vanishing gradient: градиент течёт через сложение
    в обход свёрток. Без этого глубокие сети не сходятся.
    """

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # bias=False перед BN: BN сам вычитает среднее, bias избыточен.
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # downsample нужен, когда shortcut и основной путь не совпадают по
        # форме (stride>1 уменьшил H/W, или сменилось число каналов).
        if stride != 1 or in_ch != out_ch:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class CustomResNet18(nn.Module):
    """ResNet-18 с нуля. Архитектура:

        stem: Conv7x7(s=2) → BN → ReLU → MaxPool3x3(s=2)   →  64ch, /4
        layer1: BasicBlock × 2, stride=1                    →  64ch, /4
        layer2: BasicBlock × 2, stride=2 (первый блок)      → 128ch, /8
        layer3: BasicBlock × 2, stride=2 (первый блок)      → 256ch, /16
        layer4: BasicBlock × 2, stride=2 (первый блок)      → 512ch, /32
        head:  AdaptiveAvgPool(1) → Linear(512, num_classes)

    Вход 224x224 → выход после layer4 — 7x7. ~11.7M параметров.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(in_ch=64, out_ch=64, n_blocks=2, stride=1)
        self.layer2 = self._make_stage(in_ch=64, out_ch=128, n_blocks=2, stride=2)
        self.layer3 = self._make_stage(in_ch=128, out_ch=256, n_blocks=2, stride=2)
        self.layer4 = self._make_stage(in_ch=256, out_ch=512, n_blocks=2, stride=2)
        # AdaptiveAvgPool(1): любой H x W → 1 x 1 (среднее по пространству).
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks: list[nn.Module] = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        # Kaiming для Conv: std=sqrt(2/fan_out), сохраняет дисперсию через ReLU.
        # BN: γ=1, β=0 — стартует как identity. Без этого ResNet from-scratch
        # часто не сходится.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# ---- Pretrained-обёртки (torchvision) -----------------------------------
# Transfer learning: загружаем модель с ImageNet-весами, заменяем последний
# Linear (1000 классов ImageNet → NUM_CLASSES наших классов), дообучаем.


def _build_resnet18_pretrained(num_classes: int) -> nn.Module:
    # weights=...IMAGENET1K_V1 — чекпоинт ImageNet (top-1 ~69.7%),
    # скачивается один раз в ~/.cache/torch/.
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # У ResNet head — это .fc = Linear(512, 1000). Заменяем на NUM_CLASSES.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenetv3_small_pretrained(num_classes: int) -> nn.Module:
    # MobileNetV3-Small — лёгкая (~2.5M params), архитектурно сильно отличается
    # от ResNet (depthwise-separable conv, h-swish, SE). Хорошая контрольная
    # модель для сравнения "разные семейства".
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # У MobileNetV3 head — .classifier (Sequential из 4 слоёв), последний
    # Linear(1024, 1000). Заменяем только его, остальное (Hardswish, Dropout) — как есть.
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name == MODEL_CUSTOM:
        return CustomResNet18(num_classes=num_classes)
    if name == MODEL_RESNET18:
        return _build_resnet18_pretrained(num_classes)
    if name == MODEL_MOBILENETV3:
        return _build_mobilenetv3_small_pretrained(num_classes)
    raise ValueError(f"Unknown model: {name!r}. Available: {ALL_MODELS}")


def _head_parameters(model: nn.Module, name: str) -> Iterable[nn.Parameter]:
    if name == MODEL_RESNET18:
        return model.fc.parameters()  # type: ignore[union-attr]
    if name == MODEL_MOBILENETV3:
        return model.classifier[-1].parameters()  # type: ignore[index]
    raise ValueError(f"head parameters undefined for {name!r}")


def get_param_groups(
    model: nn.Module,
    name: str,
    *,
    lr_head: float,
    weight_decay: float,
    lr_backbone: float | None = None,
) -> list[dict]:
    """Param groups для optimizer (разные LR для разных частей).

    Pretrained: 2 группы — head (новый, с нуля) с большим LR, backbone
    (ImageNet) с меньшим, чтобы не разрушить фичи. lr_backbone обязателен.
    Custom: 1 группа со всеми параметрами; lr_backbone не используется.
    """
    if name not in PRETRAINED_MODELS:
        return [{"params": list(model.parameters()), "lr": lr_head, "weight_decay": weight_decay}]

    if lr_backbone is None:
        raise ValueError(f"lr_backbone обязателен для pretrained модели {name!r}")

    head_params = list(_head_parameters(model, name))
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
    ]
