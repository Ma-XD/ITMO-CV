from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import yaml
from PIL import Image
from tqdm.auto import tqdm


@dataclass
class DigitBBox:
    label: int   # YOLO class id (0..9)
    top: float
    left: float
    height: float
    width: float


@dataclass
class ImageAnnotation:
    filename: str
    boxes: list[DigitBBox]


def svhn_label_to_yolo_class(svhn_label: int) -> int:
    """SVHN использует 1..10, где 10 = '0'. Переводим в естественную нумерацию 0..9 (digit value = class id)."""
    if svhn_label == 10:
        return 0
    if 1 <= svhn_label <= 9:
        return svhn_label
    raise ValueError(f"Unexpected SVHN label: {svhn_label!r}")


def _read_str(f: h5py.File, ref) -> str:
    arr = f[ref][()]
    return "".join(chr(int(c[0])) for c in arr)


def _read_field(f: h5py.File, bbox_ref, key: str) -> list[float]:
    ds = f[bbox_ref][key]
    if ds.shape[0] == 1:
        return [float(ds[0][0])]
    return [float(f[ds[i][0]][()][0][0]) for i in range(ds.shape[0])]


def parse_digit_struct(mat_path: Path) -> list[ImageAnnotation]:
    """Парсит digitStruct.mat (HDF5 v7.3) → список ImageAnnotation, по одной записи на изображение."""
    annotations: list[ImageAnnotation] = []
    with h5py.File(mat_path, "r") as f:
        names_ds = f["/digitStruct/name"]
        bbox_ds = f["/digitStruct/bbox"]
        n = names_ds.shape[0]

        for i in tqdm(range(n), desc=f"parse {mat_path.parent.name}"):
            name = _read_str(f, names_ds[i][0])
            bref = bbox_ds[i][0]

            labels_raw = _read_field(f, bref, "label")
            tops = _read_field(f, bref, "top")
            lefts = _read_field(f, bref, "left")
            heights = _read_field(f, bref, "height")
            widths = _read_field(f, bref, "width")

            boxes = [
                DigitBBox(
                    label=svhn_label_to_yolo_class(int(lbl)),
                    top=top, left=left, height=h, width=w,
                )
                for lbl, top, left, h, w in zip(labels_raw, tops, lefts, heights, widths)
            ]
            annotations.append(ImageAnnotation(filename=name, boxes=boxes))

    return annotations


def bbox_to_yolo(
    *, top: float, left: float, height: float, width: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float] | None:
    """(top,left,h,w) в пикселях → (cx,cy,w,h) нормализованные. None если бокс вырожден после клипа."""
    # SVHN иногда содержит координаты, выходящие за границы изображения — клипуем.
    x1 = max(0.0, min(left, img_w))
    y1 = max(0.0, min(top, img_h))
    x2 = max(0.0, min(left + width, img_w))
    y2 = max(0.0, min(top + height, img_h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    return cx, cy, bw / img_w, bh / img_h


def write_yolo_labels(
    annotations: list[ImageAnnotation],
    images_src_dir: Path,
    images_dst_dir: Path,
    labels_dst_dir: Path,
) -> tuple[int, int]:
    """Копирует/линкует изображения и пишет .txt с YOLO-разметкой. Возвращает (n_images_written, n_skipped_boxes)."""
    images_dst_dir.mkdir(parents=True, exist_ok=True)
    labels_dst_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped_boxes = 0

    for ann in tqdm(annotations, desc=f"write {images_dst_dir.parent.name}/{images_dst_dir.name}"):
        src = images_src_dir / ann.filename
        if not src.exists():
            raise FileNotFoundError(src)

        with Image.open(src) as im:
            img_w, img_h = im.size

        lines: list[str] = []
        for box in ann.boxes:
            yolo = bbox_to_yolo(
                top=box.top, left=box.left, height=box.height, width=box.width,
                img_w=img_w, img_h=img_h,
            )
            if yolo is None:
                n_skipped_boxes += 1
                continue
            cx, cy, w, h = yolo
            lines.append(f"{box.label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Пустой .txt — допустимо в YOLO (=изображение без объектов), но в SVHN такого
        # быть не должно. Если все боксы выкинуты — пропускаем картинку целиком.
        if not lines:
            continue

        dst_img = images_dst_dir / ann.filename
        if not dst_img.exists():
            try:
                dst_img.symlink_to(src.resolve())
            except OSError:
                # На некоторых ФС (NTFS, FAT) symlink не разрешён — копируем.
                import shutil
                shutil.copy2(src, dst_img)

        label_path = labels_dst_dir / (Path(ann.filename).stem + ".txt")
        label_path.write_text("\n".join(lines) + "\n")
        n_written += 1

    return n_written, n_skipped_boxes


def split_train_val(
    annotations: list[ImageAnnotation],
    val_fraction: float,
    seed: int,
) -> tuple[list[ImageAnnotation], list[ImageAnnotation]]:
    rng = random.Random(seed)
    indices = list(range(len(annotations)))
    rng.shuffle(indices)
    n_val = int(round(len(annotations) * val_fraction))
    val_idx = set(indices[:n_val])
    train_anns = [a for i, a in enumerate(annotations) if i not in val_idx]
    val_anns = [a for i, a in enumerate(annotations) if i in val_idx]
    return train_anns, val_anns


def write_data_yaml(yaml_path: Path, names: list[str]) -> None:
    """ultralytics-совместимый data.yaml. `path: .` означает «папка, где лежит этот yaml»,
    поэтому zip с этим файлом портируемый: распаковывается куда угодно — пути остаются валидными."""
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": list(names),
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def count_class_distribution(annotations: list[ImageAnnotation], num_classes: int) -> dict[int, int]:
    counts = {i: 0 for i in range(num_classes)}
    for ann in annotations:
        for box in ann.boxes:
            counts[box.label] = counts.get(box.label, 0) + 1
    return counts
