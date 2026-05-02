from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    INDEX_PATH,
    NUM_WORKERS,
    SEED,
    TARGET_COLORS,
    TEST_FRACTION,
    VAL_FRACTION,
)


def load_index(index_path: Path = INDEX_PATH) -> pd.DataFrame:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Не найден {index_path}. Сначала запусти:\n"
            f"  python lab1-CLAS/scripts/build_index.py"
        )
    return pd.read_csv(index_path)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    classes: list[str]
    class_to_idx: dict[str, int]


def make_splits(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> Splits:
    train_val, test = train_test_split(
        df, test_size=test_fraction, stratify=df["label"], random_state=seed,
    )
    val_rel = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val["label"], random_state=seed,
    )
    classes = list(TARGET_COLORS)
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        classes=classes,
        class_to_idx={c: i for i, c in enumerate(classes)},
    )


def build_transforms(img_size: int = IMG_SIZE) -> tuple[transforms.Compose, transforms.Compose]:
    resize_for_center_crop = int(round(img_size * 256 / 224))
    # Намеренно НЕ jitter-им hue/saturation: таргет — цвет, перекрашивание ломает label.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(resize_for_center_crop),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class DVMColorDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        assert {"path", "label"}.issubset(frame.columns)
        self.paths: list[str] = frame["path"].tolist()
        self.labels: list[int] = frame["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def _make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    series = pd.Series(labels)
    class_counts = series.value_counts().to_dict()
    weights = series.map(lambda y: 1.0 / class_counts[y]).astype("float64").tolist()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_dataloaders(
    splits: Splits,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    img_size: int = IMG_SIZE,
    use_weighted_sampler: bool = True,
) -> dict[str, DataLoader]:
    train_tf, eval_tf = build_transforms(img_size=img_size)

    train_ds = DVMColorDataset(splits.train, transform=train_tf)
    val_ds = DVMColorDataset(splits.val, transform=eval_tf)
    test_ds = DVMColorDataset(splits.test, transform=eval_tf)

    if use_weighted_sampler:
        sampler = _make_weighted_sampler(splits.train["label"].tolist())
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
