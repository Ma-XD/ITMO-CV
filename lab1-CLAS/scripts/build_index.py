from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_LAB_DIR = Path(__file__).resolve().parent.parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from config import (  # noqa: E402
    CLASS_TO_IDX,
    CONFIRMED_FRONTS_DIR,
    EXCLUDE_COLORS,
    INDEX_PATH,
    TARGET_COLORS,
)


_FILENAME_SEP: str = "$$"
_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def _parse_filename(path: Path) -> Optional[dict]:
    parts = path.stem.split(_FILENAME_SEP)
    if len(parts) < 6:
        return None
    maker, model, year, color = parts[0], parts[1], parts[2], parts[3]
    if not year.isdigit() or not color:
        return None
    return {
        "path": str(path),
        "maker": maker,
        "model": model,
        "year": int(year),
        "color": color.strip().lower(),
    }


def build_index(
    data_dir: Path,
    index_path: Path,
    *,
    target_colors: list[str],
    exclude_colors: set[str],
    force: bool = False,
) -> pd.DataFrame:
    if index_path.exists() and not force:
        print(f"✅ Индекс уже построен: {index_path} — пропускаю (--force чтобы пересобрать)")
        return pd.read_csv(index_path)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Не найден каталог с изображениями: {data_dir}\n"
            "Распакуй dvm_confirmed_fronts.zip в DATA_DIR из config.py"
        )

    paths = [p for ext in _IMAGE_EXTS for p in data_dir.rglob(f"*{ext}")]
    if not paths:
        raise RuntimeError(f"В {data_dir} не найдено изображений с расширениями {_IMAGE_EXTS}")

    rows: list[dict] = []
    skipped_bad_name = 0
    for p in tqdm(paths, desc="parse", unit="img"):
        entry = _parse_filename(p)
        if entry is None:
            skipped_bad_name += 1
            continue
        rows.append(entry)

    df = pd.DataFrame(rows)
    n_total = len(df)

    excluded_mask = df["color"].isin(exclude_colors)
    n_excluded = int(excluded_mask.sum())
    df = df[~excluded_mask].copy()

    target_set = set(target_colors)
    target_mask = df["color"].isin(target_set)
    n_off_target = int((~target_mask).sum())
    df = df[target_mask].copy()

    df["label"] = df["color"].map(CLASS_TO_IDX).astype("int64")
    df = df[["path", "maker", "model", "year", "color", "label"]].sort_values("path").reset_index(drop=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(index_path, index=False)

    print(
        f"\n✅ Индекс построен: {index_path}\n"
        f"   parsed:    {n_total} (из {len(paths)}; пропущено по имени: {skipped_bad_name})\n"
        f"   excluded:  {n_excluded} ({sorted(exclude_colors)})\n"
        f"   off-target: {n_off_target} (не из top-{len(target_colors)})\n"
        f"   kept:      {len(df)}\n"
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"🎨 Распределение классов ({len(TARGET_COLORS)}):")
    counts = df["color"].value_counts().reindex(TARGET_COLORS).fillna(0).astype(int)
    total = len(df)
    for color in TARGET_COLORS:
        n = int(counts[color])
        pct = n / total * 100 if total else 0
        print(f"  [{CLASS_TO_IDX[color]}] {color:<10} {n:>7}  ({pct:5.2f}%)")
    print(f"  {'TOTAL':<14} {total:>7}")
    print("=" * 60)
    print(f"📅 Годы: {df['year'].min()}–{df['year'].max()} ({df['year'].nunique()} уникальных)")
    print(f"🚗 Марок: {df['maker'].nunique()}, моделей: {df['model'].nunique()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Сборка index.csv для lab1-CLAS из confirmed_fronts/")
    ap.add_argument("--data-dir", type=Path, default=CONFIRMED_FRONTS_DIR,
                    help="каталог с изображениями (по умолчанию из config.py)")
    ap.add_argument("--index", type=Path, default=INDEX_PATH,
                    help="куда писать index.csv (по умолчанию из config.py)")
    ap.add_argument("--force", action="store_true",
                    help="пересобрать, даже если индекс уже существует")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = build_index(
        data_dir=args.data_dir,
        index_path=args.index,
        target_colors=TARGET_COLORS,
        exclude_colors=EXCLUDE_COLORS,
        force=args.force,
    )
    print_summary(df)


if __name__ == "__main__":
    main()
