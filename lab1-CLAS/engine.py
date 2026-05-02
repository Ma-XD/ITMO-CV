from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CHECKPOINT_DIR, LOG_DIR, TARGET_COLORS


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    desc: str = "train",
) -> dict:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += bs
        pbar.set_postfix(loss=f"{total_loss / total_seen:.4f}",
                         acc=f"{total_correct / total_seen:.4f}")

    return {"loss": total_loss / total_seen, "acc": total_correct / total_seen}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    desc: str = "val",
) -> dict:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_seen += bs
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"loss": total_loss / total_seen, "acc": acc, "f1_macro": float(f1)}


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    *,
    model_name: str,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    log_dir: Path = LOG_DIR,
) -> dict:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if device is None:
        device = _device_of(model)
    else:
        model.to(device)
    criterion = criterion.to(device)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1_macro": [],
        "lr_head": [], "lr_backbone": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            desc=f"epoch {epoch}/{epochs} train",
        )
        val_metrics = validate(
            model, loaders["val"], criterion, device,
            desc=f"epoch {epoch}/{epochs} val",
        )
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        # LR обеих групп: head всегда первым в get_param_groups, backbone — вторым (если есть).
        lrs = [g["lr"] for g in optimizer.param_groups]
        history["lr_head"].append(lrs[0])
        history["lr_backbone"].append(lrs[1] if len(lrs) > 1 else lrs[0])

        dt = time.time() - t0
        print(
            f"[{model_name}] epoch {epoch:>2}/{epochs}  "
            f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}  |  "
            f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
            f"f1={val_metrics['f1_macro']:.4f}  |  {dt:.1f}s"
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"{model_name}_last.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "history": history,
        "model_name": model_name,
    }, ckpt_path)
    print(f"💾 checkpoint: {ckpt_path}")

    log_path = log_dir / f"{model_name}_history.json"
    with log_path.open("w") as f:
        json.dump({"model_name": model_name, "epochs": epochs, "history": history}, f, indent=2)
    print(f"📝 history:    {log_path}")

    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    *,
    class_names: list[str] = TARGET_COLORS,
    desc: str = "test",
) -> dict:
    if device is None:
        device = _device_of(model)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float((y_true_arr == y_pred_arr).mean())
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    f1_per_class = f1_score(
        y_true_arr, y_pred_arr,
        labels=list(range(len(class_names))),
        average=None, zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": {name: float(v) for name, v in zip(class_names, f1_per_class)},
        "confusion_matrix": cm,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "class_names": list(class_names),
    }


def load_checkpoint(model: nn.Module, model_name: str,
                    checkpoint_dir: Path = CHECKPOINT_DIR,
                    device: Optional[torch.device] = None) -> dict:
    """Восстанавливает веса в model in-place. Возвращает meta-dict без state_dict."""
    ckpt_path = checkpoint_dir / f"{model_name}_last.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")
    map_loc = device if device is not None else "cpu"
    blob = torch.load(ckpt_path, map_location=map_loc)
    model.load_state_dict(blob["model_state_dict"])
    return {k: v for k, v in blob.items() if k != "model_state_dict"}
