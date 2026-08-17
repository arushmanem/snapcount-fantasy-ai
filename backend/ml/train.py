# ml/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Model
# ----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.15):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ----------------------------
# Utils
# ----------------------------
def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_feature_cols(schemas_dir: Path, pos: str, fmt: str) -> List[str]:
    p = schemas_dir / f"{pos}_{fmt}_features.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing schema: {p}")
    with open(p, "r") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError(f"Schema malformed: {p}")
    return cols


def load_parquet(datasets_dir: Path, pos: str, fmt: str, split: str) -> pd.DataFrame:
    p = datasets_dir / f"{pos}_{fmt}_{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset: {p}")
    return pd.read_parquet(p)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def to_json(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def fit(X: np.ndarray, eps: float = 1e-6) -> "Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < eps, 1.0, std)  # avoid divide-by-zero
        return Scaler(mean=mean, std=std)


def df_to_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing feature columns: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = df[feature_cols].astype(np.float32).to_numpy()
    y = df["y"].astype(np.float32).to_numpy()

    # Defensive: replace NaNs/Infs (shouldn't happen if build step is solid)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        mse = torch.mean((pred - yb) ** 2).item()
        mae = torch.mean(torch.abs(pred - yb)).item()
        bs = xb.shape[0]
        total_mse += mse * bs
        total_mae += mae * bs
        n += bs
    if n == 0:
        return {"mse": float("inf"), "mae": float("inf")}
    return {"mse": total_mse / n, "mae": total_mae / n}


def train_one(
    pos: str,
    fmt: str,
    datasets_dir: Path,
    schemas_dir: Path,
    models_dir: Path,
    *,
    hidden: List[int],
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
) -> None:
    feature_cols = load_feature_cols(schemas_dir, pos, fmt)

    df_train = load_parquet(datasets_dir, pos, fmt, "train")
    df_val = load_parquet(datasets_dir, pos, fmt, "val")

    X_train, y_train = df_to_xy(df_train, feature_cols)
    X_val, y_val = df_to_xy(df_val, feature_cols)

    scaler = Scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    train_loader = make_loader(X_train_s, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val_s, y_val, batch_size=batch_size, shuffle=False)

    model = MLPRegressor(in_dim=X_train_s.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        metrics = evaluate(model, val_loader, device)
        val_mae = metrics["mae"]

        print(f"[{pos}-{fmt}] epoch {ep:03d}/{epochs}  val_mae={val_mae:.4f}  val_mse={metrics['mse']:.4f}")

        if val_mae + 1e-6 < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{pos}-{fmt}] Early stopping (no val MAE improvement in {patience} epochs).")
                break

    if best_state is None:
        raise RuntimeError(f"[{pos}-{fmt}] Training produced no best state (unexpected).")

    # Save model checkpoint
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"{pos}_{fmt}_model.pt"
    torch.save(
        {
            "pos": pos,
            "format": fmt,
            "feature_dim": len(feature_cols),
            "hidden": hidden,
            "dropout": dropout,
            "state_dict": best_state,
        },
        ckpt_path,
    )
    print(f"[{pos}-{fmt}] Saved model -> {ckpt_path}")

    # Save scaler (must match feature order)
    scaler_path = schemas_dir / f"{pos}_{fmt}_scaler.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler.to_json(), f, indent=2)
    print(f"[{pos}-{fmt}] Saved scaler -> {scaler_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_hidden(s: str) -> List[int]:
    # e.g. "128,64" -> [128,64]
    s = s.strip()
    if not s:
        return [128, 64]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Train per-position, per-format PyTorch models.")
    ap.add_argument("--positions", nargs="+", default=["QB", "RB", "WR", "TE"])
    ap.add_argument("--formats", nargs="+", default=["ppr", "half_ppr", "standard"])
    ap.add_argument("--datasets_dir", type=str, default="artifacts/datasets")
    ap.add_argument("--schemas_dir", type=str, default="artifacts/schemas")
    ap.add_argument("--models_dir", type=str, default="artifacts/models")

    ap.add_argument("--hidden", type=str, default="128,64", help='Comma-separated hidden sizes, e.g. "256,128,64"')
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA/MPS is available.")

    args = ap.parse_args()

    device = pick_device(force_cpu=args.cpu)
    hidden = parse_hidden(args.hidden)

    datasets_dir = Path(args.datasets_dir)
    schemas_dir = Path(args.schemas_dir)
    models_dir = Path(args.models_dir)

    print(f"Device: {device}")
    print(f"Hidden: {hidden}, dropout={args.dropout}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"Datasets: {datasets_dir}")
    print(f"Schemas:  {schemas_dir}")
    print(f"Models:   {models_dir}")

    for pos in args.positions:
        for fmt in args.formats:
            train_one(
                pos=pos,
                fmt=fmt,
                datasets_dir=datasets_dir,
                schemas_dir=schemas_dir,
                models_dir=models_dir,
                hidden=hidden,
                dropout=args.dropout,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                weight_decay=args.weight_decay,
                device=device,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
