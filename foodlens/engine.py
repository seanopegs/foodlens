"""Training, evaluation, timing and result storage."""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _keep_frozen_bn_in_eval(model: nn.Module) -> None:
    """BatchNorm layers inside frozen blocks keep their ImageNet running statistics."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    training = optimizer is not None
    model.train(training)
    if training:
        _keep_frozen_bn_in_eval(model)
    use_amp = device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
    return total_loss / seen, correct / seen


def save_checkpoint(path: str, state_dict: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": state_dict, **meta}, path)


def fit(model, train_loader, val_loader, device, lr: float, max_epochs: int = 15,
        patience: int = 3, ckpt_path: str | None = None, ckpt_meta: dict | None = None) -> dict:
    """Adam + cross-entropy, early stopping on validation loss.

    The weights of the best epoch (lowest validation loss) are restored at the end
    and, if ``ckpt_path`` is given, saved together with ``ckpt_meta``.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history, best_state = [], None
    best_loss, best_epoch, waited = float("inf"), 0, 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer, scaler,
                                     desc=f"epoch {epoch} train")
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device, desc=f"epoch {epoch} val")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "elapsed_min": (time.perf_counter() - start) / 60})
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_loss:
            best_loss, best_epoch, waited = va_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                save_checkpoint(ckpt_path, best_state, {**(ckpt_meta or {}), "epoch": epoch})
        else:
            waited += 1
            if waited >= patience:
                print(f"early stopping: validation loss did not improve for {patience} epochs")
                break

    train_time_s = time.perf_counter() - start
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "epochs_run": len(history),
            "train_time_s": train_time_s}


@torch.inference_mode()
def predict(model, loader, device):
    """Return (y_true, probabilities) for a whole loader."""
    model.eval()
    labels, probs = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 top_k_accuracy_score)

    y_pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    top5 = top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(probs.shape[1]))
    return {"test_acc": float(accuracy_score(y_true, y_pred)), "test_top5_acc": float(top5),
            "macro_precision": float(precision), "macro_recall": float(recall),
            "macro_f1": float(f1)}


@torch.inference_mode()
def measure_inference_ms(model, device, img_size: int = 224, runs: int = 30, warmup: int = 5) -> float:
    """Average latency for one image (batch size 1), in milliseconds."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def save_run(run_dir: str, config: dict, history: list, metrics: dict,
             y_true: np.ndarray, probs: np.ndarray) -> None:
    """Write config, per-epoch history, test metrics and test predictions of one run."""
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    np.savez_compressed(os.path.join(run_dir, "predictions.npz"), y_true=y_true,
                        y_pred=probs.argmax(axis=1), confidence=probs.max(axis=1), top5=top5)


def load_runs(runs_dir: str) -> tuple[pd.DataFrame, dict]:
    """Return (metrics table indexed by run name, {run name: history DataFrame})."""
    rows, histories = {}, {}
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            metrics_path = os.path.join(runs_dir, name, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    rows[name] = json.load(f)
                histories[name] = pd.read_csv(os.path.join(runs_dir, name, "history.csv"))
    return pd.DataFrame.from_dict(rows, orient="index"), histories


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
                   k: int = 10) -> pd.DataFrame:
    """The k most frequent (true, predicted) pairs among the mistakes."""
    wrong = y_true != y_pred
    pairs = pd.DataFrame({"true": y_true[wrong], "predicted": y_pred[wrong]})
    counts = pairs.value_counts().head(k).reset_index(name="count")
    per_class = np.bincount(y_true, minlength=len(class_names))
    counts["share of true class"] = counts["count"] / per_class[counts["true"]]
    counts["true"] = [class_names[i] for i in counts["true"]]
    counts["predicted"] = [class_names[i] for i in counts["predicted"]]
    return counts
