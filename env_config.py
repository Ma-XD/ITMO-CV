"""
Автоматическое определение окружения и настройка путей.
Используется всеми подпроектами в репо.

Поддерживает два сценария:
  • Google Colab (есть `/content`, GPU через CUDA, Google Drive для сохранений)
  • Локальная разработка на macOS (CPU / Apple MPS, сохранения в репо)

Модуль спроектирован так, чтобы его можно было импортировать до установки
зависимостей (torch, torchvision) — в этом случае device-функции выдают
предупреждение вместо ошибки.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Определение окружения
# --------------------------------------------------------------------------- #

def _detect_colab() -> bool:
    """Вернёт True, если код выполняется в Google Colab."""
    # В Colab всегда смонтирован каталог `/content`, а также обычно есть
    # модуль google.colab. Проверяем оба признака для надёжности.
    if not Path("/content").exists():
        return False
    try:
        import google.colab  # type: ignore # noqa: F401
        return True
    except ImportError:
        # На некоторых машинах `/content` может существовать случайно —
        # дополнительно сверяемся с переменной окружения Colab.
        return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


is_colab: bool = _detect_colab()
is_local: bool = not is_colab


# --------------------------------------------------------------------------- #
# Поиск корня репозитория
# --------------------------------------------------------------------------- #

def _find_repo_root_local() -> Path:
    """Ищет корень репозитория, поднимаясь от текущего файла до каталога с `.git`."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    # Фолбэк: каталог рядом с этим файлом.
    return here.parent


def _get_repo_name() -> str:
    """Пытается определить имя репозитория из git remote; иначе берёт имя папки."""
    root = _find_repo_root_local()
    try:
        url = subprocess.check_output(
            ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if url:
            # Поддерживаем и SSH (git@github.com:user/repo.git), и HTTPS.
            name = url.rstrip("/").split("/")[-1]
            if name.endswith(".git"):
                name = name[:-4]
            return name
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return root.name


REPO_NAME: str = _get_repo_name()

if is_colab:
    REPO_ROOT: Path = Path("/content") / REPO_NAME
else:
    REPO_ROOT = _find_repo_root_local()


# --------------------------------------------------------------------------- #
# Google Drive
# --------------------------------------------------------------------------- #

DRIVE_ROOT: Optional[Path] = Path("/content/drive/MyDrive") if is_colab else None


# --------------------------------------------------------------------------- #
# Устройство (CUDA / MPS / CPU)
# --------------------------------------------------------------------------- #

def _resolve_device():
    """Возвращает torch.device под текущее окружение."""
    if not _TORCH_AVAILABLE:
        return None
    if is_colab and torch.cuda.is_available():
        return torch.device("cuda")
    if is_local and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()


# --------------------------------------------------------------------------- #
# Публичные функции
# --------------------------------------------------------------------------- #

def get_project_dir(project_name: str) -> Path:
    """Возвращает путь к подпроекту внутри репозитория.

    Args:
        project_name: имя подкаталога (например, ``"lab1-CLAS"``).

    Returns:
        Абсолютный ``Path`` к каталогу проекта.
    """
    return REPO_ROOT / project_name


def get_save_dir(project_name: str) -> Path:
    """Возвращает каталог для артефактов обучения (чекпоинты, логи, графики).

    • В Colab — внутри Google Drive, чтобы пережить сброс сессии.
    • Локально — внутри подпроекта в репозитории.

    Каталог создаётся автоматически, если его ещё нет.
    """
    if is_colab:
        if DRIVE_ROOT is None:
            raise RuntimeError(
                "❌ Google Drive не смонтирован: "
                "вызови drive.mount('/content/drive') или запусти colab_setup.py"
            )
        save_dir = DRIVE_ROOT / project_name / "outputs"
    else:
        save_dir = REPO_ROOT / project_name / "outputs"

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_device():
    """Возвращает ``torch.device`` и печатает краткую информацию.

    Returns:
        ``torch.device`` или ``None``, если torch недоступен.
    """
    if not _TORCH_AVAILABLE:
        print("⚠️  torch не установлен — device недоступен")
        return None

    device = DEVICE
    if device is None:
        print("⚠️  Не удалось определить device")
        return None

    if device.type == "cuda":
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"
        print(f"🖥️  Device: cuda ({name})")
    elif device.type == "mps":
        print("🖥️  Device: mps")
    else:
        print("🖥️  Device: cpu")
    return device


def print_env() -> None:
    """Печатает сводку окружения: платформу, device, ключевые пути и версии."""
    env_label = "Google Colab" if is_colab else "Local (macOS)"
    print("=" * 60)
    print(f"🌐 Environment: {env_label}")
    print(f"📦 Repo name:   {REPO_NAME}")
    print(f"📂 REPO_ROOT:   {REPO_ROOT}")
    print(f"📂 DRIVE_ROOT:  {DRIVE_ROOT if DRIVE_ROOT else '—'}")

    get_device()

    if _TORCH_AVAILABLE:
        print(f"📦 torch:       {torch.__version__}")
        try:
            import torchvision  # type: ignore
            print(f"📦 torchvision: {torchvision.__version__}")
        except ImportError:
            print("📦 torchvision: ❌ не установлен")
    else:
        print("📦 torch:       ❌ не установлен")
        print("📦 torchvision: ❌ не установлен")
    print("=" * 60)


if __name__ == "__main__":
    print_env()
