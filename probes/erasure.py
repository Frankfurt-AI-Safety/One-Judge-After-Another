"""
Concept erasure + probe-recoverability helpers (reasoning low/high-complexity test).

Implements the TaCo discriminator: erase a binary concept from activations, then ask whether a
**non-linear** probe can still recover it.
- ``diffmean_erase`` — our method: project out the difference-of-means direction (linear).
- ``leace_erase``    — LEACE (Belrose et al. 2023, ``concept-erasure``): provably-optimal *affine*
  erasure such that **no linear** classifier can predict the concept.
- ``probe_recoverability`` — fit a **linear** (LogisticRegression) and a **non-linear** (MLPClassifier)
  probe on a train split and score a held-out eval split. After LEACE the linear probe must drop to
  chance; if the MLP still recovers the concept ⇒ **high-complexity/entangled**, else **low-complexity**.

`X` are CPU float32 activation tensors (from ``get_embeddings``); `y` are 0/1 labels.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from probes.probe import project_to_null_space


def _as_tensor(y) -> torch.Tensor:
    return y if isinstance(y, torch.Tensor) else torch.as_tensor(np.asarray(y))


def diffmean_erase(X: torch.Tensor, y) -> torch.Tensor:
    """Difference-of-means direction (normalized) for concept ``y`` — project it out with
    :func:`project_to_null_space` to erase the concept linearly (our baseline method)."""
    y = _as_tensor(y).bool()
    direction = X[y].mean(0) - X[~y].mean(0)
    return direction / (direction.norm() + 1e-8)


def leace_erase(X: torch.Tensor, y):
    """Fit a LEACE eraser on (X, y). Returns a callable eraser; apply to any tensor of the same dim."""
    from concept_erasure import LeaceEraser

    return LeaceEraser.fit(X.float(), _as_tensor(y))


def probe_recoverability(
    X_tr: torch.Tensor, y_tr, X_ev: torch.Tensor, y_ev, seed: int = 0,
) -> Dict[str, float]:
    """Held-out recoverability of concept ``y`` from activations: linear vs non-linear probe accuracy,
    with the majority-class ``chance`` baseline. Features standardized on the train split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    Xtr = X_tr.float().cpu().numpy()
    Xev = X_ev.float().cpu().numpy()
    ytr = _as_tensor(y_tr).long().cpu().numpy()
    yev = _as_tensor(y_ev).long().cpu().numpy()

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xev = scaler.transform(Xtr), scaler.transform(Xev)

    lin = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                        random_state=seed).fit(Xtr, ytr)
    chance = float(max(np.mean(yev == 0), np.mean(yev == 1)))
    return {
        "linear_acc": float(lin.score(Xev, yev)),
        "mlp_acc": float(mlp.score(Xev, yev)),
        "chance": chance,
        "n_train": int(len(ytr)), "n_eval": int(len(yev)),
    }


def apply_eraser(eraser, X: torch.Tensor) -> torch.Tensor:
    """Apply a fitted LEACE eraser to activations X."""
    return eraser(X.float())


def apply_diffmean(direction: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Erase the difference-of-means direction from X (full projection, alpha=1)."""
    return project_to_null_space(X.float(), direction, alpha=1.0)
