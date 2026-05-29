from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _detect_colab() -> bool:
    if not Path("/content").exists():
        return False
    try:
        import google.colab  # type: ignore # noqa: F401
        return True
    except ImportError:
        return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


is_colab: bool = _detect_colab()
is_local: bool = not is_colab


LAB_DIR: Path = Path(__file__).resolve().parent
PROJECT_NAME: str = LAB_DIR.name


DRIVE_ROOT: Optional[Path] = Path("/content/drive/MyDrive") if is_colab else None


def _resolve_device():
    if not _TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()


def get_save_dir() -> Path:
    if is_colab:
        if DRIVE_ROOT is None:
            raise RuntimeError(
                "❌ Google Drive не смонтирован: вызови drive.mount('/content/drive')"
            )
        save_dir = DRIVE_ROOT / "ITMO-CV" / PROJECT_NAME / "outputs"
    else:
        save_dir = LAB_DIR / "outputs"

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_device():
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
    import platform
    env_label = "Google Colab" if is_colab else f"Local ({platform.system()})"
    print("=" * 60)
    print(f"🌐 Environment: {env_label}")
    print(f"📦 Project:     {PROJECT_NAME}")
    print(f"📂 LAB_DIR:     {LAB_DIR}")
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
