from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from data import (  # noqa: E402
    count_class_distribution,
    parse_digit_struct,
    split_train_val,
    write_data_yaml,
    write_yolo_labels,
)


# Одноразовая prep-утилита: вызывается локально с распакованными SVHN-tarball'ами,
# результат (svhn_yolo.zip) кладётся на Drive. В Colab эта программа не запускается.
LAB_DIR: Path = _SCRIPTS_DIR.parent
DATA_DIR: Path = LAB_DIR / "data"
SVHN_TRAIN_TAR: Path = DATA_DIR / "train.tar.gz"
SVHN_TEST_TAR: Path = DATA_DIR / "test.tar.gz"
SVHN_RAW_DIR: Path = DATA_DIR / "svhn_raw"
SVHN_YOLO_DIR: Path = DATA_DIR / "svhn_yolo"
SVHN_YAML: Path = SVHN_YOLO_DIR / "data.yaml"

NUM_CLASSES: int = 10
CLASS_NAMES: list[str] = [str(i) for i in range(NUM_CLASSES)]
VAL_FRACTION: float = 0.10
SEED: int = 42


def _extract_tar(tar_path: Path, dest_dir: Path, expected_subdir: str) -> Path:
    """Распаковывает tar (если нужно) и возвращает путь к подпапке внутри dest_dir."""
    target = dest_dir / expected_subdir
    if target.exists() and (target / "digitStruct.mat").exists():
        print(f"✅ Уже распакован: {target}")
        return target

    if not tar_path.exists():
        raise FileNotFoundError(f"Не найден архив: {tar_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Распаковываю {tar_path.name} → {dest_dir} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)

    if not target.exists():
        raise RuntimeError(
            f"После распаковки не найдена ожидаемая подпапка: {target}. "
            f"Проверь содержимое {dest_dir}."
        )
    return target


def _build_split(
    split_name: str,
    annotations,
    images_src_dir: Path,
    yolo_root: Path,
) -> None:
    images_dst = yolo_root / "images" / split_name
    labels_dst = yolo_root / "labels" / split_name
    n_imgs, n_skipped_boxes = write_yolo_labels(
        annotations,
        images_src_dir=images_src_dir,
        images_dst_dir=images_dst,
        labels_dst_dir=labels_dst,
    )
    counts = count_class_distribution(annotations, num_classes=NUM_CLASSES)
    total_boxes = sum(counts.values())
    print(
        f"  [{split_name:>5}] images={n_imgs:>6}  boxes={total_boxes:>6}  "
        f"skipped_boxes={n_skipped_boxes}"
    )
    dist = "  ".join(f"{i}:{counts[i]}" for i in range(NUM_CLASSES))
    print(f"          class dist: {dist}")


def build(
    *,
    train_tar: Path,
    test_tar: Path,
    raw_dir: Path,
    yolo_root: Path,
    yaml_path: Path,
    val_fraction: float,
    seed: int,
    force: bool,
) -> None:
    if force and yolo_root.exists():
        import shutil
        print(f"🗑  --force: удаляю {yolo_root}")
        shutil.rmtree(yolo_root)

    if yolo_root.exists() and yaml_path.exists():
        print(f"✅ YOLO-датасет уже собран: {yolo_root} (используй --force для пересборки)")
        return

    train_src = _extract_tar(train_tar, raw_dir, "train")
    test_src = _extract_tar(test_tar, raw_dir, "test")

    print("\n🔍 Парсю digitStruct.mat (train)...")
    train_full = parse_digit_struct(train_src / "digitStruct.mat")
    print(f"   найдено {len(train_full)} train-изображений")

    print("\n🔍 Парсю digitStruct.mat (test)...")
    test_anns = parse_digit_struct(test_src / "digitStruct.mat")
    print(f"   найдено {len(test_anns)} test-изображений")

    train_anns, val_anns = split_train_val(train_full, val_fraction=val_fraction, seed=seed)
    print(f"\n✂️  Split train: {len(train_anns)} train + {len(val_anns)} val "
          f"(val_fraction={val_fraction:.2f})")

    print("\n💾 Запись YOLO-разметки...")
    _build_split("train", train_anns, train_src, yolo_root)
    _build_split("val", val_anns, train_src, yolo_root)
    _build_split("test", test_anns, test_src, yolo_root)

    write_data_yaml(yaml_path, names=CLASS_NAMES)
    print(f"\n📄 data.yaml → {yaml_path}")
    print(f"\n✅ Готово: {yolo_root}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SVHN tarballs → YOLO-формат (одноразовый local prep)")
    ap.add_argument("--train-tar", type=Path, default=SVHN_TRAIN_TAR)
    ap.add_argument("--test-tar", type=Path, default=SVHN_TEST_TAR)
    ap.add_argument("--raw-dir", type=Path, default=SVHN_RAW_DIR)
    ap.add_argument("--yolo-root", type=Path, default=SVHN_YOLO_DIR)
    ap.add_argument("--yaml-path", type=Path, default=SVHN_YAML)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="пересобрать, удалив yolo_root")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    build(
        train_tar=args.train_tar,
        test_tar=args.test_tar,
        raw_dir=args.raw_dir,
        yolo_root=args.yolo_root,
        yaml_path=args.yaml_path,
        val_fraction=args.val_fraction,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
