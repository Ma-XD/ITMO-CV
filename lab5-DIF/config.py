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

# Директории для датасета DreamBooth
if is_colab:
    # В Colab берем фото с Google Диска
    INSTANCE_DIR: Path = DRIVE_ROOT / "ITMO-CV" / PROJECT_NAME / "data" / "instance_images"
else:
    # Локально берем из папки проекта
    INSTANCE_DIR: Path = DATA_DIR / "instance_images"

CLASS_DIR: Path = DATA_DIR / "class_images"

SAVE_DIR: Path = get_save_dir()
CHECKPOINT_DIR: Path = SAVE_DIR / "checkpoints"
LOG_DIR: Path = SAVE_DIR / "logs"
FIGURE_DIR: Path = SAVE_DIR / "figures"

for _d in (INSTANCE_DIR, CLASS_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Константы для обучения
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INSTANCE_PROMPT = "a photo of ohwx man"
CLASS_PROMPT = "a photo of a man"
RESOLUTION = 512
