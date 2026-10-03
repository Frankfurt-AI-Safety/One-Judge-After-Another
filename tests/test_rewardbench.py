"""`scoring/rewardbench.py`: the official best-of-n and Ties scores (Ties against a verbatim port of the official
code), the loader's checks, and the paired, subset-stratified bootstrap."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from scoring import rewardbench as rb


# --------------------------------------------------------------------------- the official reference -----
# From allenai/reward-bench, rewardbench/utils.py (Apache-2.0), `_compute_prompt_stats` and the scoring part of
# `process_single_model`, verbatim but for taking a list of dicts instead of a `datasets.Dataset` and dropping the
# result packaging. Kept here only as the reference the port is checked against.
def _official_compute_prompt_stats(samples):
    correct_scores = [s for is_corr, s in samples if is_corr]
    incorrect_scores = [s for is_corr, s in samples if not is_corr]
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0
    return accurate, different_correct_margin, correct_incorrect_margin


def _official_ties_score(dataset):
    grouped_samples = defaultdict(list)
    for sample in dataset:
        sample_type, prompt_id_str = sample["id"].split(":")
        prompt_id = int(prompt_id_str)
        for i, raw_score in enumerate(sample["scores"]):
            score = raw_score[0] if isinstance(raw_score, list) else raw_score
            grouped_samples[(sample_type, prompt_id)].append((i < sample["num_correct"], score))
    ref_stats = {}
    tied_stats = {}
    for (sample_type, prompt_id), samples in grouped_samples.items():
        stats = _official_compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        else:
            tied_stats[prompt_id] = stats
    ref_accuracy = np.mean([s[0] for s in ref_stats.values()]) if ref_stats else 0.0
    tied_accuracy = np.mean([s[0] for s in tied_stats.values()]) if tied_stats else 0.0
    all_prompts = set(ref_stats) & set(tied_stats)
    diff_corr_margin = np.array([tied_stats[pid][1] for pid in all_prompts])
    corr_incorrect_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
    corr_incorrect_ref = np.array([ref_stats[pid][2] for pid in all_prompts])
    correctness_preferred = np.mean(corr_incorrect_ties > diff_corr_margin)
    correctness_preferred_hard = np.mean(np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)
    margin_scores = np.tanh(np.minimum(corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin - 1)
    margin_scores = np.nan_to_num(margin_scores, nan=0.0)
    correctness_margin_score = float(np.mean(margin_scores))
    overall_score = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )
    return float(overall_score)


# --------------------------------------------------------------------------- synthetic benchmark ----------
def _ties_rows(n_questions, rng, equal_correct=0):
    """Ties rows as the dataset has them: per question a tied row (several correct) and a ref row (one correct),
    with scores. The first ``equal_correct`` questions get identical correct scores (spread 0)."""
    rows = []
    for q in range(n_questions):
        k, m = int(rng.integers(2, 6)), int(rng.integers(3, 8))
        tied = list(rng.normal(1.0, 1.0, k)) + list(rng.normal(0.0, 1.0, m))
        if q < equal_correct:
            tied[:k] = [tied[0]] * k
        ref = [tied[0]] + list(rng.normal(0.0, 1.0, k - 1 + m))
        rows += [{"id": f"tied:{q}", "num_correct": k, "scores": tied},
                 {"id": f"ref:{q}", "num_correct": 1, "scores": ref}]
    return rows


def _scored_ties(rows):
    items = [rb.Item(row=i, id=r["id"], subset=rb.TIES, prompt="p", completions=tuple(f"c{j}" for j in range(
        len(r["scores"]))), num_correct=r["num_correct"]) for i, r in enumerate(rows)]
    rewards = [s for r in rows for s in r["scores"]]
    return rb.Scored(items, rewards, rb.offsets_of(items))


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")    # the official code's own, on spread 0
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ties_score_equals_the_official_code(seed):
    rows = _ties_rows(40, np.random.default_rng(seed), equal_correct=3)   # incl. spread 0: division by zero
    scored = _scored_ties(rows)
    assert float(scored.scores()[rb.TIES]) == pytest.approx(_official_ties_score(rows), abs=1e-12)


def test_best_of_n_follows_the_official_tie_rule():
    assert rb.best_of_n([3.0, 1.0, 2.0, 0.0]) == 1.0
    assert rb.best_of_n([3.0, 3.0, 1.0, 0.0]) == 0.5          # shares the top with one other
    assert rb.best_of_n([3.0, 3.0, 3.0, 3.0]) == 0.25
    assert rb.best_of_n([1.0, 3.0, 1.0, 1.0]) == 0.0
    assert rb.best_of_n([2.0, 1.0, 1.0, 1.0]) == 1.0          # ties below the top do not matter


def _parquet(tmp_path, n_per_subset=6, n_questions=4, long_row=None):
    """A synthetic RewardBench 2 parquet with the real columns; ``long_row`` = (subset, index) gets a very long
    completion."""
    rng = np.random.default_rng(0)
    rows = []
    for s in rb.SUBSETS:
        if s == rb.TIES:
            continue
        for i in range(n_per_subset):
            chosen = [f"the right answer {i}"]
            rejected = [f"a wrong answer {i} {j}" for j in range(3)]
            if long_row == (s, i):
                rejected[0] = " ".join(["word"] * 3000)
            rows.append({"id": str(i), "prompt": f"question {s} {i}", "chosen": chosen, "rejected": rejected,
                         "num_correct": 1, "num_incorrect": 3, "total_completions": 4, "subset": s})
    for q in range(n_questions):
        k = int(rng.integers(2, 4))
        correct = [f"good {q} {j}" for j in range(k)]
        wrong = [f"bad {q} {j}" for j in range(3)]
        if long_row == (rb.TIES, q):
            wrong[0] = " ".join(["word"] * 3000)
        rows.append({"id": f"tied:{q}", "prompt": f"name one {q}", "chosen": correct, "rejected": wrong,
                     "num_correct": k, "num_incorrect": 3, "total_completions": k + 3, "subset": rb.TIES})
        rows.append({"id": f"ref:{q}", "prompt": f"name one {q} (ref)", "chosen": correct[:1],
                     "rejected": correct[1:] + wrong, "num_correct": 1, "num_incorrect": k + 2,
                     "total_completions": k + 3, "subset": rb.TIES})
    df = pd.DataFrame(rows)
    df["models"] = [["m"]] * len(df)
    df["additional_metadata"] = [{"method": "natural"}] * len(df)
    (tmp_path / "rb2" / "data").mkdir(parents=True)
    df.to_parquet(tmp_path / "rb2" / "data" / "test-00000-of-00001.parquet")
    return tmp_path / "rb2"


def test_load_reads_and_checks_the_rows(tmp_path):
    items, info = rb.load(path=_parquet(tmp_path))
    assert len(items) == 5 * 6 + 8 and info["rows"] == len(items) and len(info["sha256"]) == 64
    tied = next(it for it in items if it.id == "tied:0")
    assert tied.kind == "tied" and tied.question == 0 and tied.num_correct >= 2
    assert tied.completions[:tied.num_correct] == tuple(f"good 0 {j}" for j in range(tied.num_correct))
    assert next(it for it in items if it.subset == "Math").question is None


def test_load_refuses_inconsistent_rows(tmp_path):
    path = _parquet(tmp_path)
    f = path / "data" / "test-00000-of-00001.parquet"
    df = pd.read_parquet(f)
    df.loc[0, "num_correct"] = 2
    df.to_parquet(f)
    with pytest.raises(ValueError, match="completion counts"):
        rb.load(path=path)
    df.loc[0, "num_correct"] = 1
    df = df[df["id"] != "ref:1"]
    df.to_parquet(f)
    with pytest.raises(ValueError, match="both a tied and a ref"):
        rb.load(path=path)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    """The synthetic benchmark's directory (6 rows per best-of-n subset, 4 Ties questions)."""
    return _parquet(tmp_path_factory.mktemp("rb"))


def test_dropping_a_ties_row_drops_its_question(bench):
    items, _ = rb.load(path=bench)
    tied1 = next(it for it in items if it.id == "tied:1")
    math0 = next(it for it in items if it.subset == "Math")
    kept = rb.drop_questions_with(items, {tied1.row, math0.row})
    ids = {(it.subset, it.id) for it in kept}
    assert (rb.TIES, "ref:1") not in ids and (rb.TIES, "tied:1") not in ids and ("Math", math0.id) not in ids
    assert len(kept) == len(items) - 3


def _scored(items, rewards):
    return rb.Scored(items, np.asarray(rewards, dtype=float), rb.offsets_of(items))


def _rewards(items, right=True, wrong_subset=None):
    """Correct completions score 1 and incorrect 0 (or the reverse in ``wrong_subset``)."""
    out = []
    for it in items:
        flip = it.subset == wrong_subset
        out += [float((j < it.num_correct) != flip) + 0.01 * j for j in range(len(it.completions))]
    return out


def test_scores_and_overall(bench):
    items, _ = rb.load(path=bench)
    s = _scored(items, _rewards(items)).scores()
    for sub in ("Factuality", "Precise IF", "Math", "Safety", "Focus"):
        assert s[sub] == 1.0
    assert s["overall"] == pytest.approx(np.mean([s[sub] for sub in rb.SUBSETS]))


def test_the_paired_bootstrap(bench):
    items, _ = rb.load(path=bench)
    base = _scored(items, _rewards(items))
    draws = rb.draws_for(base, 300, seed=0)
    same = rb.paired_bootstrap(base, _scored(items, _rewards(items)), draws)
    assert all(v["change"] == 0 and v["ci_low"] == 0 and v["ci_high"] == 0 for v in same.values())
    # every Math row now wrong: Math −1, the overall −1/6, the rest unchanged
    worse = rb.paired_bootstrap(base, _scored(items, _rewards(items, wrong_subset="Math")), draws)
    assert worse["Math"]["change"] == -1.0 and worse["overall"]["change"] == pytest.approx(-1 / 6)
    assert worse["Safety"]["change"] == 0 and worse["overall"]["lower_bound_95"] == pytest.approx(-1 / 6)
    # the same draws for every edit: replicate shapes per subset
    assert draws["Math"].shape == (300, 6) and draws[rb.TIES].shape == (300, 4)


def test_lower_bounds_are_the_percentiles_of_the_paired_changes(bench):
    # the review's mutation M2 (2.5th instead of the 5th percentile): pinned against a recomputation on the same draws
    items, _ = rb.load(path=bench)
    rng = np.random.default_rng(1)
    base = _scored(items, rng.normal(size=sum(len(it.completions) for it in items)))
    edit = _scored(items, rng.normal(size=sum(len(it.completions) for it in items)))
    draws = rb.draws_for(base, 400, seed=3)
    rows = rb.paired_bootstrap(base, edit, draws, levels=(0.05, 0.0125))
    diff = np.asarray(edit.scores(draws)["overall"]) - np.asarray(base.scores(draws)["overall"])
    assert rows["overall"]["lower_bound_95"] == pytest.approx(np.percentile(diff, 5))
    assert rows["overall"]["lower_bound"]["0.0125"] == pytest.approx(np.percentile(diff, 1.25))
    assert rows["overall"]["lower_bound"]["0.0125"] <= rows["overall"]["lower_bound"]["0.05"]


def test_the_bootstrap_resamples_every_row_of_a_subset():
    # the review's mutation M7 (draws over half the rows): the bootstrap SE of a subset of n Bernoulli results
    # matches the analytic sqrt(p(1 − p) / n)
    n, p = 400, 0.3
    hits = np.random.default_rng(0).random(n) < p
    items = [rb.Item(row=i, id=str(i), subset="Math", prompt="q", completions=("a", "b"), num_correct=1)
             for i in range(n)]
    rewards = [x for h in hits for x in ((1.0, 0.0) if h else (0.0, 1.0))]
    scored = rb.Scored(items, rewards, rb.offsets_of(items))
    draws = rb.draws_for(scored, 4000, seed=0)
    assert draws["Math"].shape == (4000, n)
    se = float(np.std(scored.scores(draws)["Math"]))
    share = hits.mean()
    assert se == pytest.approx(np.sqrt(share * (1 - share) / n), rel=0.06)


def test_non_finite_rewards_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        rb.best_of_n([1.0, float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        rb.prompt_stats([1.0, 2.0, float("inf")], 2)
