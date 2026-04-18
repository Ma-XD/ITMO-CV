"""
Одноразовая настройка окружения в Google Colab.
Запуск: !python colab_setup.py

Выполняет:
  1. Проверку, что мы действительно в Colab
  2. Диагностику GPU
  3. Проверку, что Google Drive уже смонтирован
     (сам `drive.mount()` должен быть вызван в ячейке ноутбука — он требует
     IPython-ядро и не работает из subprocess-Python)
  4. Установку зависимостей из requirements.txt
  5. Импорт-тест ключевых пакетов
  6. Создание каталога outputs на Google Drive
  7. Итоговый статус ✅ / ❌
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# Критичные пакеты, без которых дальше работать нельзя.
CRITICAL_PACKAGES: list[str] = [
    "torch",
    "torchvision",
    "sklearn",      # пакет scikit-learn импортируется как sklearn
    "matplotlib",
]


def _step(title: str) -> None:
    print(f"\n⏳ {title}")


def ensure_colab() -> None:
    """Прерывает выполнение, если запущено вне Colab."""
    _step("Step 1/6 — проверка окружения")
    if not Path("/content").exists():
        print("❌ Этот скрипт предназначен для Google Colab.")
        print("   Локально он не требуется — просто используй env_config.py.")
        sys.exit(1)
    try:
        import google.colab  # type: ignore # noqa: F401
    except ImportError:
        print("❌ Модуль google.colab недоступен — это не Colab-окружение.")
        sys.exit(1)
    print("✅ Colab обнаружен")


def check_gpu() -> None:
    """Печатает информацию о GPU (или предупреждает, если его нет)."""
    _step("Step 2/6 — проверка GPU")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True,
        ).strip()
        if out:
            print(f"🖥️  GPU: {out}")
            print("✅ GPU доступен")
            return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    print("⚠️  GPU не найден. В Colab: Runtime → Change runtime type → GPU.")


def ensure_drive_mounted() -> None:
    """Проверяет, что Google Drive смонтирован в /content/drive.

    Сам ``drive.mount()`` здесь не вызывается: модуль ``google.colab.drive``
    требует работающее IPython-ядро ноутбука, а ``!python colab_setup.py``
    запускается как отдельный subprocess без IPython. Поэтому монтирование
    должно выполняться в ячейке ноутбука **до** запуска этого скрипта.
    """
    _step("Step 3/6 — проверка Google Drive")
    if Path("/content/drive/MyDrive").exists():
        print("✅ Google Drive подключён: /content/drive/MyDrive")
        return
    print("❌ Google Drive не смонтирован.")
    print("   Выполни в ячейке ноутбука до запуска colab_setup.py:")
    print("     from google.colab import drive")
    print("     drive.mount('/content/drive')")
    sys.exit(1)


def install_requirements() -> None:
    """Устанавливает зависимости из requirements.txt (если файл существует)."""
    _step("Step 4/6 — установка зависимостей")
    if not REQUIREMENTS.exists():
        print(f"⚠️  {REQUIREMENTS} не найден — пропускаю установку")
        return
    print(f"📦 pip install -r {REQUIREMENTS.name}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)],
        check=False,
    )
    if result.returncode != 0:
        print("❌ pip install завершился с ошибкой")
        sys.exit(result.returncode)
    print("✅ Зависимости установлены")


def verify_imports() -> None:
    """Проверяет, что все критичные пакеты импортируются."""
    _step("Step 5/6 — проверка импортов")
    failed: list[str] = []
    for pkg in CRITICAL_PACKAGES:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError as e:
            print(f"  ❌ {pkg}: {e}")
            failed.append(pkg)
    if failed:
        print(f"❌ Не импортируются: {', '.join(failed)}")
        sys.exit(1)


def ensure_outputs_dir() -> None:
    """Создаёт корневой каталог outputs на Google Drive."""
    _step("Step 6/6 — подготовка каталога outputs")
    outputs_root = Path("/content/drive/MyDrive") / "ITMO-CV-outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    print(f"📂 Каталог для результатов: {outputs_root}")
    print("   Конкретный подкаталог создаётся через env_config.get_save_dir(project_name)")


def main() -> None:
    ensure_colab()
    check_gpu()
    ensure_drive_mounted()
    install_requirements()
    verify_imports()
    ensure_outputs_dir()
    print("\n" + "=" * 60)
    print("✅ Ready — окружение настроено, можно запускать обучение")
    print("=" * 60)


if __name__ == "__main__":
    main()
