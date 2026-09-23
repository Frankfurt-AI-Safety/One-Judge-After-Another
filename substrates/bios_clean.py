"""
Plausibility filters for Bias-in-Bios records in the hiring factorial (`pairs/factorial.py`).

Every biography is rendered at all levels of sex × age × family status, so it must fit each level:
a clause saying the applicant is 30 must not sit next to a degree earned in 2005 or "25 years of
experience". Unlike credit, the fields are free text, so the rules are text patterns and err on the
side of dropping (about 37% of scrubbed bios on a 20k sample; ~60k of 94k remain).

- ``mentions_year_before_2014``: a 30-year-old in 2026 was born in 1996, so any event year up to 2013
  (before age 18) is implausible, and a 1900s/2000s date is usually a degree or first job.
- ``mentions_more_than_8_years``: "N years" with N > 8 (most often experience) exceeds a career that
  started after a degree at about 22.
- ``mentions_many_years_in_words``: "fifteen years", "two decades", ...
- ``mentions_retirement``: incompatible with "currently in continuous employment".

The family-status levels need no rule: the explicit clause ("currently on parental leave") fits a
continuing career, and the proxy (a parent-association role) is a volunteering activity.

**Profession cap and role assignment (2026-09-23).** Two consequences of these rules for the role-match
label `qualified`, both found by the 2026-09-23 audit:

- The rules keep professions at very different rates (dentist 29%, physician 86%), after the loader has
  already balanced the target roles — so they re-opened the role-name leak the loader closed (role-only
  accuracy 50.6% before the rules, 52.4% after, 53.4% in a 4,000-bio sample; P(qualified | dentist) 0.25).
  `qualified` and the target roles are therefore **re-assigned here, on the records actually used**:
  after the rules, the profession cap and the `n` sample. Half of every profession is qualified (exactly,
  not by an independent coin, whose noise alone let a role-name reader reach 0.529 on 4,000 bios), and
  unqualified bios are screened for a permutation of the unqualified professions, so every role name is
  qualified in half its uses.
- Professors are 42% of the filtered pool, so the task collapsed to "is this an academic?": a rule that
  only asks whether bio and header agree on "professor" scored 92.4% on `qualified`. **No profession may
  exceed `DEFAULT_PROFESSION_CAP` of the pool** (the first bios of a capped profession in the loader's seeded
  order are kept).
- Some "mismatches" were near-synonyms (1,235 teacher<->professor): **`NEAR_SYNONYM_ROLES` pairs are never
  assigned as a mismatch**, since a holder of one title may be qualified for the other.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from substrates.bios_ingest import (
    DEFAULT_BIOS_PATH, RealCVRecord, _coin, load_bias_in_bios, with_article,
)
from substrates.rules import Rule, apply_rules

_YEAR_RE = re.compile(r"\b(?:19\d\d|200\d|201[0-3])\b")
_YEARS_RE = re.compile(r"\b(\d{1,2})\+?\s*(?:-\s*)?(?:plus\s+)?years?\b", re.IGNORECASE)
_WORD_YEARS_RE = re.compile(
    r"\b(?:nine|ten|eleven|twelve|fifteen|twenty|thirty|forty)\b[\w\s-]{0,12}\byears?\b|\bdecades?\b",
    re.IGNORECASE,
)
_RETIRED_RE = re.compile(r"\bretire(?:d|ment)\b", re.IGNORECASE)

FACTORIAL_RULES: Tuple[Rule, ...] = (
    ("mentions_year_before_2014", lambda r: bool(_YEAR_RE.search(r.bio_text))),
    ("mentions_more_than_8_years",
     lambda r: any(int(m.group(1)) > 8 for m in _YEARS_RE.finditer(r.bio_text))),
    ("mentions_many_years_in_words", lambda r: bool(_WORD_YEARS_RE.search(r.bio_text))),
    ("mentions_retirement", lambda r: bool(_RETIRED_RE.search(r.bio_text))),
)


# Pairs of professions where a holder of one could plausibly be qualified for the other, so neither is
# ever assigned as the other's mismatched target role. Conservative: shared job title or near-identical
# work. (teacher / yoga_teacher: the role word itself appears in yoga-teacher bios.)
NEAR_SYNONYM_ROLES: FrozenSet[FrozenSet[str]] = frozenset({
    frozenset({"professor", "teacher"}),
    frozenset({"physician", "surgeon"}),
    frozenset({"attorney", "paralegal"}),
    frozenset({"teacher", "yoga_teacher"}),
})

# Max share of the pool any single profession may take.
DEFAULT_PROFESSION_CAP = 0.10


def cap_professions(records: Sequence[RealCVRecord],
                    cap: float) -> Tuple[List[RealCVRecord], Dict[str, object]]:
    """Keep at most `m` bios per profession, with the largest `m` such that no profession exceeds
    `cap` of the CAPPED pool (removing professors shrinks the pool, so the limit is solved for, not
    read off the uncapped counts). The first bios of each capped profession in input order are kept."""
    counts = Counter(r.profession for r in records)
    if not records or max(counts.values()) <= cap * len(records):
        return list(records), {"cap": cap, "limit": None, "capped": {}, "n_in": len(records),
                               "n_out": len(records)}
    sizes = sorted(counts.values(), reverse=True)
    limit = 0
    for k in range(1, len(sizes) + 1):          # try capping the k largest professions at `m`
        rest = sum(sizes[k:])
        m = int(cap * rest / (1 - cap * k)) if cap * k < 1 else sizes[0]
        if k == len(sizes) or m >= sizes[k]:     # the next profession is already under the limit
            limit = min(m, sizes[k - 1])
            break
    seen: Counter = Counter()
    kept: List[RealCVRecord] = []
    for r in records:
        if seen[r.profession] < limit:
            kept.append(r)
        seen[r.profession] += 1
    capped = {p: {"from": c, "to": min(c, limit)} for p, c in counts.items() if c > limit}
    return kept, {"cap": cap, "limit": limit, "capped": capped, "n_in": len(records),
                  "n_out": len(kept)}


def _allowed(profession: str, target: str, forbidden: FrozenSet[FrozenSet[str]]) -> bool:
    return profession != target and frozenset({profession, target}) not in forbidden


def _assign_mismatched_roles(professions: Sequence[str], rng: random.Random,
                             forbidden: FrozenSet[FrozenSet[str]]) -> List[str]:
    """A target role for each unqualified bio: the group's own professions, shuffled so that no bio
    gets its own profession or a near-synonym of it. Like `bios_ingest._assign_mismatched_roles`
    (whose idea this extends), the targets are a permutation of the group's professions, so the role
    name alone says nothing about `qualified`."""
    n = len(professions)
    targets = list(professions)
    rng.shuffle(targets)
    for k in range(n):
        if _allowed(professions[k], targets[k], forbidden):
            continue
        start = rng.randrange(n)
        for step in range(n):
            j = (start + step) % n
            if (_allowed(professions[k], targets[j], forbidden)
                    and _allowed(professions[j], targets[k], forbidden)):
                targets[k], targets[j] = targets[j], targets[k]
                break
        else:
            raise ValueError(
                f"cannot give an unqualified {professions[k]!r} bio a target role that is neither its "
                f"own profession nor a near-synonym; one profession dominates the group (lower the cap)")
    return targets


def _row(record: RealCVRecord) -> int:
    return int(record.source_record_id.rsplit("-", 1)[1])


def _balanced_heads(records: Sequence[RealCVRecord], seed: int) -> List[bool]:
    """`qualified` for each record, EXACTLY half within every profession (odd counts: the loader's coin
    of the middle bio decides the extra one). Within a profession the bios are ranked by the same
    (seed, row) digest the loader's coin uses, so the choice is deterministic and independent of the
    text. An independent fair coin per bio, as the loader draws, leaves small professions lopsided
    (4 of 6 qualified), which a role-name reader can exploit on a sample: role-only accuracy 0.529 and
    P(qualified | role) 0.33-0.75 on 4,000 bios, from coin noise alone."""
    def rank(row: int) -> bytes:
        return hashlib.sha256(f"{seed}|{row}".encode("utf-8")).digest()

    by_prof: Dict[str, List[int]] = {}
    for k, r in enumerate(records):
        by_prof.setdefault(r.profession, []).append(k)
    heads = [False] * len(records)
    for members in by_prof.values():
        members.sort(key=lambda k: rank(_row(records[k])))
        n_heads = len(members) // 2 + (len(members) % 2 and _coin(seed, _row(records[members[len(members) // 2]])))
        for k in members[:n_heads]:
            heads[k] = True
    return heads


def assign_roles(records: Sequence[RealCVRecord], seed: int,
                 forbidden: FrozenSet[FrozenSet[str]] = NEAR_SYNONYM_ROLES) -> List[RealCVRecord]:
    """Re-assign `qualified` and the target roles on exactly these records: half of every profession
    is qualified (`_balanced_heads`); unqualified bios get a derangement of the unqualified group's
    professions that avoids `forbidden` near-synonym pairs. Because the unqualified targets are a
    permutation of the unqualified professions, every role is then qualified in half its uses."""
    heads = _balanced_heads(records, seed)
    tails = [k for k, h in enumerate(heads) if not h]
    mismatched = _assign_mismatched_roles([records[k].profession for k in tails], random.Random(seed),
                                          forbidden)
    target_of = dict(zip(tails, mismatched))
    out = []
    for k, r in enumerate(records):
        target = r.profession if heads[k] else target_of[k]
        out.append(dataclasses.replace(r, target_role=target, role=with_article(target),
                                       qualified=heads[k]))
    return out


def role_leak_report(records: Sequence[RealCVRecord]) -> Dict[str, object]:
    """How well the rendered target role alone predicts `qualified` (0.5 = no leak)."""
    by_role: Dict[str, List[int]] = {}
    for r in records:
        by_role.setdefault(r.target_role, [0, 0])[int(r.qualified)] += 1
    n = max(len(records), 1)
    p_q = {role: round(v[1] / sum(v), 3) for role, v in by_role.items()}
    return {"role_only_accuracy": round(sum(max(v) for v in by_role.values()) / n, 4),
            "p_qualified_by_role_min": min(p_q.values(), default=None),
            "p_qualified_by_role_max": max(p_q.values(), default=None)}


def load_factorial_bios(
    path: str | Path = DEFAULT_BIOS_PATH,
    *,
    n: Optional[int] = None,
    report: Optional[Dict[str, object]] = None,
    profession_cap: Optional[float] = DEFAULT_PROFESSION_CAP,
    **kwargs,
) -> List[RealCVRecord]:
    """Scrubbed biographies that fit every level of the hiring factorial, with a clean role label.

    Order: the loader (scrub, coin) -> `FACTORIAL_RULES` -> the profession cap -> the `n` sample ->
    role re-assignment on exactly the records returned (see the module docstring). `kwargs` go to
    `load_bias_in_bios`, including its `seed`, which also fixes the coin and the role shuffle. If
    `report` is a dict it is filled with the loader's report plus ``factorial_rules``,
    ``profession_cap`` and ``role_leak`` (measured on the returned records).
    """
    seed = kwargs.get("seed", 42)
    records = load_bias_in_bios(path, report=report, **kwargs)
    kept, rules_report = apply_rules(records, FACTORIAL_RULES)
    cap_report: Dict[str, object] = {"cap": None}
    if profession_cap:
        kept, cap_report = cap_professions(kept, profession_cap)
        # The cap keeps the FIRST bios of a capped profession, which sit early in the loader's order,
        # so reshuffle: otherwise a prefix (the `n` sample) is still dominated by the capped profession.
        random.Random(f"{seed}|profession_cap").shuffle(kept)
    kept = assign_roles(kept[:n] if n else kept, seed)
    if report is not None:
        report["factorial_rules"] = rules_report
        report["profession_cap"] = cap_report
        report["role_leak"] = role_leak_report(kept)
    return kept
