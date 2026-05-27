from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

from env_config import LAB_DIR, get_save_dir, is_colab  # noqa: E402


PROJECT_DIR: Path = LAB_DIR

if is_colab:
    DATA_DIR: Path = Path("/content/data")
else:
    DATA_DIR = PROJECT_DIR / "data"

# SVHN — наша конверсия в YOLO-формат (на Drive хранится как svhn_yolo.zip).
SVHN_DIR: Path = DATA_DIR / "svhn_yolo"
SVHN_YAML: Path = SVHN_DIR / "data.yaml"

# Toronto numberdetection — Roboflow YOLOv8 zip, распаковывается как есть.
TORONTO_DIR: Path = DATA_DIR / "numberdetection"
TORONTO_YAML: Path = TORONTO_DIR / "data.yaml"

# Папка с пользовательскими уличными фотографиями (на Drive — street_photos/).
STREET_PHOTOS_DIR: Path = DATA_DIR / "street_photos"


SAVE_DIR: Path = get_save_dir()
CHECKPOINT_DIR: Path = SAVE_DIR / "checkpoints"
LOG_DIR: Path = SAVE_DIR / "logs"
FIGURE_DIR: Path = SAVE_DIR / "figures"

for _d in (CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


NUM_CLASSES: int = 10
CLASS_NAMES: list[str] = [str(i) for i in range(NUM_CLASSES)]
