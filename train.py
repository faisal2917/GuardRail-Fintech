"""
Train the GuardRail fraud model locally with PyTorch.

Dataset: Kaggle "Credit Card Fraud Detection" (ULB) — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Prepare it first with data_pipeline.py, which writes data/processed/{train,val,test}.csv.

Usage:
    python data_pipeline.py all             # 1) download from Kaggle + clean + split
    python train.py                         # 2) train on data/processed/
    python train.py --synthetic             # no download needed: synthetic data with the same schema (demo only)
    python train.py --device cpu --epochs 40

Outputs (written to ./artifacts, consumed by app.py):
    fraud_model.pt      network weights + scaler stats + decision threshold
    metadata.json       test metrics, curves, training history, global feature importance
    background.npy      standardised reference sample used by SHAP
    presets.json        one legitimate and one fraudulent test transaction for the UI presets
    sample_batch.csv    held-out transactions for trying the Batch CSV Scanner
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import DataPreparer
from fraud_model import FEATURES, TARGET, FraudNet, explain

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the GuardRail PyTorch fraud model.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed", help="Folder with train/val/test.csv from data_pipeline.py.")
    p.add_argument("--synthetic", action="store_true", help="Generate a synthetic dataset instead of reading --data-dir.")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pos-weight", type=float, default=None, help="BCE positive-class weight. Default: sqrt(neg/pos).")
    p.add_argument("--patience", type=int, default=6, help="Early-stopping patience (epochs) on validation PR-AUC.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_synthetic_dataset(n: int = 120_000, fraud_rate: float = 0.004, seed: int = 42) -> pd.DataFrame:
    """Synthetic data with the Kaggle schema. Fraud is shifted on the components that dominate the real dataset."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    stds = np.linspace(1.9, 0.3, 28)
    V = rng.normal(0, stds, size=(n, 28))

    shifts = {14: -7.5, 12: -6.0, 10: -5.0, 17: -6.5, 4: 4.0, 11: 3.5, 3: -6.0, 16: -4.0, 7: -5.0, 2: 3.0}
    fraud_idx = np.flatnonzero(y)
    strength = np.where(rng.random(len(fraud_idx)) < 0.25, 0.25, 1.0)  # a quarter of frauds are "stealthy"
    for comp, shift in shifts.items():
        V[fraud_idx, comp - 1] += shift * strength * rng.uniform(0.5, 1.3, len(fraud_idx))

    amount = rng.lognormal(3.0, 1.4, n)
    small = rng.random(len(fraud_idx)) < 0.45
    amount[fraud_idx] = np.where(small, rng.uniform(0.5, 5, len(fraud_idx)), rng.lognormal(4.8, 1.1, len(fraud_idx)))

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)).round())
    df["Amount"] = np.clip(amount, 0, 25_000).round(2)
    df[TARGET] = y
    return df


def load_splits(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], str]:
    if args.synthetic:
        return DataPreparer(seed=args.seed).split(make_synthetic_dataset(seed=args.seed)), "synthetic"

    paths = {name: args.data_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        raise SystemExit(
            f"Prepared splits not found in {args.data_dir}\n"
            "Run `python data_pipeline.py all` first (downloads from Kaggle, cleans and splits), "
            "or use --synthetic for a demo model."
        )
    splits = {name: pd.read_csv(p) for name, p in paths.items()}
    for name, part in splits.items():
        missing = [c for c in FEATURES + [TARGET] if c not in part.columns]
        if missing:
            raise SystemExit(f"{paths[name]} is missing columns: {missing}")
    return splits, "creditcard.csv (Kaggle)"


@torch.no_grad()
def predict_logits(model: nn.Module, Z: np.ndarray, device: torch.device, batch: int = 16384) -> np.ndarray:
    model.eval()
    t = torch.from_numpy(Z)
    return torch.cat([model(t[i : i + batch].to(device)).squeeze(1).cpu() for i in range(0, len(t), batch)]).numpy()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def downsample(xs: np.ndarray, ys: np.ndarray, n: int = 250) -> dict[str, list[float]]:
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return {"x": xs[idx].round(5).tolist(), "y": ys[idx].round(5).tolist()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    splits, source = load_splits(args)
    df = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    X = df[FEATURES].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    n_train, n_val = len(splits["train"]), len(splits["val"])
    idx_train = np.arange(n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, len(df))
    print(f"Loaded {len(df):,} rows from {source} · fraud rate {y.mean():.4%} · device {device}")

    mean = X[idx_train].mean(0)
    scale = X[idx_train].std(0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale

    neg, pos = (y[idx_train] == 0).sum(), (y[idx_train] == 1).sum()
    pos_weight = args.pos_weight if args.pos_weight is not None else math.sqrt(neg / max(pos, 1))
    print(f"Train {len(idx_train):,} · Val {len(idx_val):,} · Test {len(idx_test):,} · pos_weight {pos_weight:.1f}")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Z[idx_train]), torch.from_numpy(y[idx_train])),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # avoids a size-1 final batch, which BatchNorm cannot train on
    )
    model = FraudNet(len(FEATURES), tuple(args.hidden), args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history: list[dict] = []
    best_ap, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / (len(loader) * args.batch_size)

        val_logits = predict_logits(model, Z[idx_val], device)
        val_loss = float(criterion(torch.from_numpy(val_logits).to(device), torch.from_numpy(y[idx_val]).to(device)).item())
        val_ap = average_precision_score(y[idx_val], val_logits)
        val_auc = roc_auc_score(y[idx_val], val_logits)
        scheduler.step(val_ap)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_pr_auc": val_ap, "val_roc_auc": val_auc})
        print(f"Epoch {epoch:02d} · loss {train_loss:.4f} · val_loss {val_loss:.4f} · val PR-AUC {val_ap:.4f} · ROC-AUC {val_auc:.4f} · {time.time() - t0:.1f}s")

        if val_ap > best_ap:
            best_ap, best_state, stale = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping (no PR-AUC gain for {args.patience} epochs).")
                break

    model.load_state_dict(best_state)
    model = model.cpu().eval()
    cpu = torch.device("cpu")

    # Decision threshold: maximise F1 on the validation set
    val_p = sigmoid(predict_logits(model, Z[idx_val], cpu))
    prec, rec, thr = precision_recall_curve(y[idx_val], val_p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    threshold = float(thr[np.argmax(f1)])

    # Held-out test evaluation
    y_test = y[idx_test]
    test_p = sigmoid(predict_logits(model, Z[idx_test], cpu))
    pred = (test_p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    amounts = X[idx_test, FEATURES.index("Amount")]
    pr_p, pr_r, _ = precision_recall_curve(y_test, test_p)
    fpr, tpr, _ = roc_curve(y_test, test_p)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_p)),
        "pr_auc": float(average_precision_score(y_test, test_p)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fraud_amount_caught": float(amounts[(pred == 1) & (y_test == 1)].sum()),
        "fraud_amount_missed": float(amounts[(pred == 0) & (y_test == 1)].sum()),
    }
    print(
        f"\nTest · ROC-AUC {metrics['roc_auc']:.4f} · PR-AUC {metrics['pr_auc']:.4f} · "
        f"P {metrics['precision']:.3f} · R {metrics['recall']:.3f} · F1 {metrics['f1']:.3f} @ threshold {threshold:.3f}"
    )

    # SHAP reference sample + global importance
    rng = np.random.default_rng(args.seed)
    background = Z[rng.choice(idx_train, size=min(200, len(idx_train)), replace=False)]
    test_fraud, test_legit = idx_test[y_test == 1], idx_test[y_test == 0]
    explain_idx = np.concatenate([
        rng.choice(test_fraud, size=min(150, len(test_fraud)), replace=False),
        rng.choice(test_legit, size=min(150, len(test_legit)), replace=False),
    ])
    sv, method = explain(model, background, Z[explain_idx])
    importance = dict(sorted(zip(FEATURES, np.abs(sv).mean(0).round(5).tolist()), key=lambda kv: -kv[1]))

    # UI presets: a confidently-legit and a confidently-fraudulent held-out transaction
    legit_pos = np.flatnonzero(y_test == 0)
    fraud_pos = np.flatnonzero(y_test == 1)
    legit_pick = legit_pos[np.argsort(test_p[legit_pos])[: max(1, len(legit_pos) // 10)]]
    legit_row = legit_pick[np.argmin(np.abs(amounts[legit_pick] - np.median(amounts)))]
    fraud_pick = fraud_pos[test_p[fraud_pos] >= threshold] if (test_p[fraud_pos] >= threshold).any() else fraud_pos
    big = fraud_pick[amounts[fraud_pick] >= 100]
    fraud_row = (big if len(big) else fraud_pick)[np.argmax(test_p[big if len(big) else fraud_pick])]
    presets = {
        "legitimate": dict(zip(FEATURES, X[idx_test[legit_row]].round(6).tolist())),
        "fraud": dict(zip(FEATURES, X[idx_test[fraud_row]].round(6).tolist())),
    }

    # Sample batch for the CSV scanner (held-out data, includes ground truth)
    n_fraud = min(40, len(test_fraud))
    sample_idx = np.concatenate([rng.choice(test_fraud, n_fraud, replace=False), rng.choice(test_legit, 1000 - n_fraud, replace=False)])
    df.iloc[rng.permutation(sample_idx)][FEATURES + [TARGET]].to_csv(args.out / "sample_batch.csv", index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": FEATURES,
            "hidden": list(args.hidden),
            "dropout": args.dropout,
            "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(),
            "threshold": threshold,
        },
        args.out / "fraud_model.pt",
    )
    np.save(args.out / "background.npy", background.astype(np.float32))
    (args.out / "presets.json").write_text(json.dumps(presets, indent=2))
    (args.out / "metadata.json").write_text(
        json.dumps(
            {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_source": source,
                "n_rows": len(df),
                "fraud_rate": float(y.mean()),
                "split": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
                "device": str(device),
                "torch_version": torch.__version__,
                "params": {
                    "epochs_run": len(history),
                    "best_val_pr_auc": best_ap,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "pos_weight": pos_weight,
                },
                "threshold": threshold,
                "test_metrics": metrics,
                "curves": {"pr": downsample(pr_r, pr_p), "roc": downsample(fpr, tpr)},
                "history": history,
                "importance": {"method": method, "values": importance},
            },
            indent=2,
        )
    )
    print(f"Artifacts written to {args.out}")


if __name__ == "__main__":
    main()
