"""
Tier-1 structural validation gate — enforces single-axis variance on a matched pair.

Three deterministic checks (no model/semantic deps):
1. **Exact single-axis diff** — stripping each marker clause from its side yields *identical*
   remainders. This is the strong guarantee that ONLY the demographic marker differs (financial
   content is byte-identical). Each clause must occur exactly once.
2. **Length parity** — |Δchars| and |Δtokens| within bounds, so the marker isn't a gross length cue
   (the brittleness confound). Bounds are lenient enough to admit legitimate marker-length variation
   (e.g. a career-gap proxy is intrinsically a little longer) while catching template-level drift.
3. **Readability parity** — |ΔFlesch reading-ease| within bound (via ``textstat``; skipped with a
   note if textstat is unavailable, e.g. on a minimal CI image).

Over-generate ~3× upstream and discard pairs that fail; keep the survivors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pairs.markers import GeneratedPair

try:
    import textstat as _textstat
except Exception:  # pragma: no cover - optional dep
    _textstat = None


@dataclass
class Thresholds:
    max_char_delta: int = 12
    max_token_delta: int = 3
    max_flesch_delta: float = 8.0


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def _strip_once(text: str, clause: str) -> Tuple[str, int]:
    """Return (text with the first occurrence of clause removed, occurrence count)."""
    return text.replace(clause, "", 1), text.count(clause)


def validate_pair(pair: GeneratedPair, thr: Optional[Thresholds] = None) -> ValidationResult:
    thr = thr or Thresholds()
    reasons: List[str] = []

    # 1. exact single-axis diff
    rem_a, n_a = _strip_once(pair.text_a, pair.clause_a)
    rem_b, n_b = _strip_once(pair.text_b, pair.clause_b)
    if n_a != 1 or n_b != 1:
        reasons.append(f"marker clause not found exactly once (a={n_a}, b={n_b})")
    if rem_a != rem_b:
        reasons.append("non-marker content differs between the pair (not single-axis)")

    # 2. length parity
    char_delta = abs(len(pair.text_a) - len(pair.text_b))
    token_delta = abs(len(pair.text_a.split()) - len(pair.text_b.split()))
    if char_delta > thr.max_char_delta:
        reasons.append(f"char-length delta {char_delta} > {thr.max_char_delta}")
    if token_delta > thr.max_token_delta:
        reasons.append(f"token-length delta {token_delta} > {thr.max_token_delta}")

    # 3. readability parity
    flesch_delta = float("nan")
    if _textstat is not None:
        try:
            flesch_delta = abs(
                _textstat.flesch_reading_ease(pair.text_a)
                - _textstat.flesch_reading_ease(pair.text_b)
            )
            if flesch_delta > thr.max_flesch_delta:
                reasons.append(f"Flesch delta {flesch_delta:.1f} > {thr.max_flesch_delta}")
        except Exception:  # pragma: no cover - defensive
            pass

    return ValidationResult(
        ok=not reasons,
        reasons=reasons,
        metrics={"char_delta": char_delta, "token_delta": token_delta, "flesch_delta": flesch_delta},
    )


def validate_pairs(
    pairs: List[GeneratedPair], thr: Optional[Thresholds] = None
) -> Tuple[List[GeneratedPair], List[Tuple[GeneratedPair, ValidationResult]], Dict[str, object]]:
    """Validate a batch. Returns (passed, failed_with_results, report)."""
    thr = thr or Thresholds()
    passed: List[GeneratedPair] = []
    failed: List[Tuple[GeneratedPair, ValidationResult]] = []
    reason_counts: Dict[str, int] = {}
    for p in pairs:
        res = validate_pair(p, thr)
        if res.ok:
            passed.append(p)
        else:
            failed.append((p, res))
            for r in res.reasons:
                key = r.split(" (")[0].split(" >")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
    report = {
        "n_total": len(pairs),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "discard_rate": round(len(failed) / max(len(pairs), 1), 4),
        "failure_reasons": reason_counts,
        "textstat_available": _textstat is not None,
        "thresholds": {"max_char_delta": thr.max_char_delta, "max_token_delta": thr.max_token_delta,
                       "max_flesch_delta": thr.max_flesch_delta},
    }
    return passed, failed, report
