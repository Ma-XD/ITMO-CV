from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

from env_config import LAB_DIR, get_save_dir, is_colab  # noqa: E402


PROJECT_DIR: Path = LAB_DIR

# В Colab данные распаковываются в /content/data/dvm (быстрый локальный SSD VM),
# чтобы DataLoader не читал тысячи мелких jpg напрямую с Drive.
if is_colab:
    DATA_DIR: Path = Path("/content/data/dvm")
else:
    DATA_DIR = PROJECT_DIR / "data" / "dvm"

CONFIRMED_FRONTS_DIR: Path = DATA_DIR / "confirmed_fronts"
INDEX_PATH: Path = DATA_DIR / "index.csv"

SAVE_DIR: Path = get_save_dir()
CHECKPOINT_DIR: Path = SAVE_DIR / "checkpoints"
LOG_DIR: Path = SAVE_DIR / "logs"
FIGURE_DIR: Path = SAVE_DIR / "figures"

for _d in (CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


EXCLUDE_COLORS: set[str] = {"unlisted", "multicolour"}
TARGET_COLORS: list[str] = ["black", "grey", "white", "blue", "silver", "red"]
NUM_CLASSES: int = len(TARGET_COLORS)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(TARGET_COLORS)}


SEED: int = 42
IMG_SIZE: int = 224

BATCH_SIZE: int = 64
NUM_WORKERS: int = 2

VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15

MODEL_CUSTOM: str = "custom_resnet18"
MODEL_RESNET18: str = "resnet18"
MODEL_MOBILENETV3: str = "mobilenetv3_small"
ALL_MODELS: list[str] = [MODEL_CUSTOM, MODEL_RESNET18, MODEL_MOBILENETV3]
PRETRAINED_MODELS: set[str] = {MODEL_RESNET18, MODEL_MOBILENETV3}

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
