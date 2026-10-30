"""
Shared model definition for GuardRail Fintech.

Both `train.py` (training) and `app.py` (inference) import from here so the
network architecture and the preprocessing logic can never drift apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

FEATURES: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


class FraudNet(nn.Module):
    """Feed-forward network for tabular fraud detection. Outputs a raw logit."""

    def __init__(self, n_features: int = len(FEATURES), hidden: tuple[int, ...] = (128, 64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape (N, 1) logits


class FraudScorer:
    """Bundles the trained network with its scaler stats and decision threshold."""

    def __init__(self, model: FraudNet, mean: np.ndarray, scale: np.ndarray, threshold: float, features: list[str]):
        self.model = model.eval()
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.threshold = float(threshold)
        self.features = features

    @classmethod
    def load(cls, path: str | Path) -> "FraudScorer":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = FraudNet(n_features=len(ckpt["features"]), hidden=tuple(ckpt["hidden"]), dropout=ckpt["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model=model,
            mean=np.asarray(ckpt["scaler_mean"], dtype=np.float32),
            scale=np.asarray(ckpt["scaler_scale"], dtype=np.float32),
            threshold=ckpt["threshold"],
            features=list(ckpt["features"]),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise raw features. The result is each feature's z-score vs. the training baseline."""
        return (np.asarray(X, dtype=np.float32) - self.mean) / self.scale

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        Z = torch.from_numpy(self.transform(X))
        out = [torch.sigmoid(self.model(Z[i : i + batch_size])).squeeze(1) for i in range(0, len(Z), batch_size)]
        return torch.cat(out).numpy() if out else np.empty(0, dtype=np.float32)


def explain(model: FraudNet, background_z: np.ndarray, Z: np.ndarray, nsamples: int = 200) -> tuple[np.ndarray, str]:
    """
    Per-feature attributions in log-odds space for already-standardised rows `Z`.

    Uses SHAP's GradientExplainer (expected gradients) and falls back to plain
    Integrated Gradients if `shap` is unavailable or fails on this platform.
    Returns (attributions of shape (N, F), method name).
    """
    model.eval()
    bg = torch.from_numpy(np.asarray(background_z, dtype=np.float32))
    x = torch.from_numpy(np.asarray(Z, dtype=np.float32))
    try:
        import shap

        sv = shap.GradientExplainer(model, bg).shap_values(x, nsamples=nsamples, rseed=0)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        return sv, "SHAP · GradientExplainer"
    except Exception:
        steps = 64
        base = bg.mean(0, keepdim=True)
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1)
        path = (base + alphas * (x - base)).requires_grad_(True)
        out = model(path.reshape(-1, x.shape[1])).sum()
        grads = torch.autograd.grad(out, path)[0]
        return ((x - base) * grads.mean(0)).detach().numpy(), "Integrated Gradients"
