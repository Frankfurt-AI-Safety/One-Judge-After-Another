"""
Concept erasure + probe-recoverability helpers (reasoning low/high-complexity test, Bias-in-Bios scrub check).

Implements the TaCo discriminator: erase a binary concept from activations, then ask whether a
**non-linear** probe can still recover it.
- ``diffmean_direction`` — our method: the difference-of-means direction, to be projected out (linear).
- ``leace_erase``        — LEACE (Belrose et al. 2023, ``concept-erasure``): the optimal *affine* erasure after
  which no linear classifier predicts the concept better than a constant — exactly on the states it was
  fitted on, approximately on held-out ones (the fit is a covariance estimate, rank-deficient when the
  dimension exceeds the number of training states).
- ``probe_recoverability`` — fit a **linear** (LogisticRegression) and a **non-linear** (MLPClassifier)
  probe on a train split and score a held-out eval split, with 95% cluster-bootstrap intervals. After LEACE
  the linear probe should sit at chance; if the MLP still recovers the concept ⇒ **high-complexity/entangled**,
  else **low-complexity**. That is read from the interval of ``mlp_above_chance``, one-sided: recoverable only
  if the interval lies clearly above 0. The baseline is the majority-class share (>= 0.5), so a probe at chance
  sits at or below it.

`X` are CPU float32 activation tensors (from ``get_embeddings``); `y` are 0/1 labels (bool, int or float),
and both classes must occur in a training split.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

import numpy as np
import torch

from probes.probe import project_to_null_space


def _as_tensor(y) -> torch.Tensor:
    return y if isinstance(y, torch.Tensor) else torch.as_tensor(np.asarray(y))


def _labels(y, *, train: bool) -> torch.Tensor:
    """``y`` as a 0/1 long tensor. Any other coding (e.g. 1/2) raises: it would silently mean another concept,
    and an empty class makes the difference-of-means direction NaN, which the projection then drops, so the
    "erased" states would be the raw ones. A training split needs both classes."""
    t = _as_tensor(y)
    values = set(torch.unique(t).tolist())
    if not values <= {0, 1}:
        raise ValueError(f"labels must be 0/1, got the values {sorted(values)}")
    if train and len(values) < 2:
        raise ValueError(f"a training split needs both classes, got only {sorted(values)}")
    return t.long()


def diffmean_direction(X: torch.Tensor, y) -> torch.Tensor:
    """Difference-of-means direction (normalized) for concept ``y`` — project it out with
    :func:`apply_diffmean` to erase the concept linearly (our baseline method)."""
    y = _labels(y, train=True).bool()
    direction = X[y].mean(0) - X[~y].mean(0)
    return direction / (direction.norm() + 1e-8)


def leace_erase(X: torch.Tensor, y):
    """Fit a LEACE eraser on (X, y). Returns a callable eraser; apply to any tensor of the same dim. ``y`` is one 0/1
    label per row, or several concepts' labels (a list of label sequences), erased jointly: then no linear function of
    the states predicts any of them (a single concept's erasure leaves the others, from which a non-linear probe can
    rebuild it where the concepts combine, e.g. correctness = valence XOR premise)."""
    from concept_erasure import LeaceEraser

    if isinstance(y, (list, tuple)) and y and isinstance(y[0], (list, tuple, torch.Tensor)):
        z = torch.stack([_labels(c, train=True).float() for c in y], dim=1)
        return LeaceEraser.fit(X.float(), z)
    return LeaceEraser.fit(X.float(), _labels(y, train=True))


def probe_recoverability(
    X_tr: torch.Tensor, y_tr, X_ev: torch.Tensor, y_ev, seed: int = 0,
    groups_ev: Optional[Sequence[Hashable]] = None, n_boot: Optional[int] = None,
    reference_ok: Optional[Sequence[bool]] = None, return_items: bool = False, mlp_seeds: int = 1,
) -> Dict[str, Any]:
    """Held-out recoverability of concept ``y`` from activations: linear vs non-linear probe accuracy,
    with the majority-class ``chance`` baseline. Features standardized on the train split.

    ``intervals`` holds each accuracy, ``chance`` and ``mlp_above_chance`` / ``linear_above_chance``
    (accuracy − chance, computed on the same replicate) with its 95% percentile interval over resamples of
    whole eval clusters (`scoring.intervals.cluster_bootstrap`). ``groups_ev`` names each eval state's
    cluster: states that share an item (e.g. an applicant's four reasoning cells) are correlated, and
    resampling them one by one would give too narrow an interval. Without it every state is its own cluster.

    ``reference_ok`` (one flag per eval state: did a reference predictor, e.g. a no-model rule, get it right?)
    adds ``reference_acc`` and the paired ``linear_minus_reference`` / ``mlp_minus_reference`` on the same states
    and replicates: what the probe recovers beyond the reference.

    ``return_items`` adds ``items`` (per eval state: linear correct, MLP correct, label, reference correct), which
    `recoverability_summary` pools over several fits (the reasoning erasure test's paraphrase folds). ``mlp_seeds`` > 1
    fits that many MLPs (seeds ``seed``, ``seed + 1``, …; an odd number) and takes their majority vote per state: one
    early-stopped fit can miss a non-linear structure another finds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    Xtr = X_tr.float().cpu().numpy()
    Xev = X_ev.float().cpu().numpy()
    ytr = _labels(y_tr, train=True).cpu().numpy()
    yev = _labels(y_ev, train=False).cpu().numpy()
    if groups_ev is not None and len(groups_ev) != len(yev):
        raise ValueError(f"{len(groups_ev)} groups for {len(yev)} eval states")
    if reference_ok is not None and len(reference_ok) != len(yev):
        raise ValueError(f"{len(reference_ok)} reference flags for {len(yev)} eval states")

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xev = scaler.transform(Xtr), scaler.transform(Xev)

    lin = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
    if mlp_seeds < 1 or mlp_seeds % 2 == 0:
        raise ValueError(f"mlp_seeds must be odd and positive (a majority vote), got {mlp_seeds}")
    votes = sum(MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                              random_state=seed + k).fit(Xtr, ytr).predict(Xev) for k in range(mlp_seeds))
    lin_ok = lin.predict(Xev) == yev
    mlp_ok = (votes * 2 > mlp_seeds).astype(int) == yev

    # items: (linear correct, MLP correct, label, reference correct) per eval state
    ref = [False] * len(yev) if reference_ok is None else [bool(x) for x in reference_ok]
    items = list(zip(lin_ok.tolist(), mlp_ok.tolist(), yev.tolist(), ref))
    keys = list(range(len(items))) if groups_ev is None else list(groups_ev)
    out = {**recoverability_summary(items, keys, seed=seed, n_boot=n_boot, reference=reference_ok is not None),
           "n_train": int(len(ytr)), "n_eval": int(len(yev))}
    if return_items:
        out["items"] = items
    return out


def recoverability_summary(items: Sequence[Tuple[bool, bool, int, bool]], keys: Sequence[Hashable], seed: int = 0,
                           n_boot: Optional[int] = None, reference: bool = False) -> Dict[str, Any]:
    """Accuracies, ``chance`` (the majority-class share) and their intervals over resamples of whole ``keys``
    clusters, from per-state ``items`` (linear correct, MLP correct, label, reference correct) — of one fit, or of
    several pooled (then a key names the same unit, e.g. an applicant, in every fit)."""
    from scoring.intervals import DEFAULT_N_BOOT, cluster_bootstrap, clusters_of

    if len(keys) != len(items):
        raise ValueError(f"{len(keys)} cluster keys for {len(items)} items")

    def acc(k: int):
        return lambda s: float(np.mean([it[k] for it in s]))

    def chance(s) -> float:
        share = float(np.mean([it[2] for it in s]))
        return max(share, 1.0 - share)

    stats = {"linear_acc": acc(0), "mlp_acc": acc(1), "chance": chance,
             "linear_above_chance": lambda s: acc(0)(s) - chance(s),
             "mlp_above_chance": lambda s: acc(1)(s) - chance(s)}
    if reference:
        stats.update({"reference_acc": acc(3), "linear_minus_reference": lambda s: acc(0)(s) - acc(3)(s),
                      "mlp_minus_reference": lambda s: acc(1)(s) - acc(3)(s)})
    intervals = cluster_bootstrap(clusters_of(items, keys), stats,
                                  n_boot=DEFAULT_N_BOOT if n_boot is None else n_boot, seed=seed)
    return {"linear_acc": acc(0)(items), "mlp_acc": acc(1)(items), "chance": chance(items), "intervals": intervals}


def bag_of_words(train_texts: Sequence[str], eval_texts: Sequence[str]):
    """Binary bag-of-words vectors (float32 tensors) of ``train_texts`` and ``eval_texts``, the vocabulary taken from
    the training texts: a representation with nothing but word identity. Run through the same erasure and probes as
    the model's states, it is the **lexical control**: what surface features alone recover. A model's
    non-linear recovery after LEACE says more than the text's surface only where it exceeds this control."""
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(binary=True, lowercase=True, token_pattern=r"[A-Za-z']+").fit(train_texts)
    as_tensor = lambda texts: torch.tensor(vec.transform(texts).toarray(), dtype=torch.float32)
    return as_tensor(train_texts), as_tensor(eval_texts)


def apply_eraser(eraser, X: torch.Tensor) -> torch.Tensor:
    """Apply a fitted LEACE eraser to activations X."""
    return eraser(X.float())


def apply_diffmean(direction: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Erase the difference-of-means direction from X (full projection, alpha=1)."""
    return project_to_null_space(X.float(), direction, alpha=1.0)
