from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_config import REPO_ROOT, is_colab  # noqa: E402


def _download_with_gdown(folder_id: str, dest: Path) -> int:
    try:
        import gdown  # type: ignore
    except ImportError:
        print("❌ gdown не установлен. Поставь: pip install gdown")
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"⏳ Скачиваю {url} → {dest}")
    try:
        gdown.download_folder(url=url, output=str(dest), quiet=False, use_cookies=False)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Ошибка скачивания: {e}")
        return 1
    print(f"✅ Готово: {dest}")
    return 0


def _print_instructions(project: str | None) -> None:
    print("ℹ️  Как забрать результаты обучения:")
    print()
    print("  1. Открой Google Drive (тот же аккаунт, который был в Colab).")
    print("  2. Перейди в MyDrive → ITMO-CV-outputs"
          + (f" → {project}" if project else ""))
    print("  3. Скачай папку outputs → размести её рядом с проектом локально.")
    print()
    if project:
        local_dir = REPO_ROOT / project / "outputs"
        print(f"📂 Локальный путь: {local_dir}")
    else:
        print(f"📂 Корень репозитория: {REPO_ROOT}")
    print()
    print("Автоматизировать можно так:")
    print("  python scripts/sync_results.py --folder-id <DRIVE_FOLDER_ID> "
          "[--project <name>]")
    print("  (папка должна быть доступна по ссылке, иначе gdown не сможет её открыть).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Синхронизация результатов с Google Drive")
    parser.add_argument("--folder-id", type=str, default=None,
                        help="ID папки Google Drive с результатами")
    parser.add_argument("--project", type=str, default=None,
                        help="Имя подпроекта (например, lab1-CLAS)")
    args = parser.parse_args()

    if is_colab:
        print("⚠️  Скрипт рассчитан на локальный запуск. В Colab данные уже на Drive.")
        return 0

    if args.folder_id:
        dest = REPO_ROOT / (args.project or "") / "outputs"
        return _download_with_gdown(args.folder_id, dest)

    _print_instructions(args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
