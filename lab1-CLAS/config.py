from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from env_config import get_save_dir, is_colab  # noqa: E402


PROJECT_NAME: str = "lab1-CLAS"
PROJECT_DIR: Path = _REPO_ROOT / PROJECT_NAME

# В Colab данные распаковываются в /content/data/dvm (быстрый локальный SSD VM),
# чтобы DataLoader не читал тысячи мелких jpg напрямую с Drive.
if is_colab:
    DATA_DIR: Path = Path("/content/data/dvm")
else:
    DATA_DIR = PROJECT_DIR / "data" / "dvm"

CONFIRMED_FRONTS_DIR: Path = DATA_DIR / "confirmed_fronts"
INDEX_PATH: Path = DATA_DIR / "index.csv"

SAVE_DIR: Path = get_save_dir(PROJECT_NAME)
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

EPOCHS_SCRATCH: int = 25
EPOCHS_FINETUNE: int = 12

LR_SCRATCH: float = 1e-3
LR_FINETUNE: float = 1e-4
WEIGHT_DECAY: float = 1e-4

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
