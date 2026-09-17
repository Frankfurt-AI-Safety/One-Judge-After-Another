# One Judge After Another

Demographic and intersectional protected-attribute bias in **language reward models**, and
whether mechanistic reward shaping removes it or merely hides it.

A reward model is the shared evaluator of modern alignment: one preference model is trained
once, then reused across RLHF, best-of-$N$ selection and data filtering. A systematic error in
that one component is not a local defect — it propagates into everything trained against it.

**The method is inference-only.** Forward passes, last-layer activation extraction, and linear
algebra. No weights are trained.

---

## Repository layout — and the review map

Each root folder is one stage of the pipeline and one **review session**. Read them in this
order; each depends only on the ones above it.

Review in the following order: substrates -> pairs -> probes -> scoring -> runners

| # | folder | stage | size |
|---|---|---|---|
| 1 | `substrates/` | real corpora → records, and how a record becomes text | 923 lines |
| 2 | `pairs/` | single-axis marker injection, the validation gate, the manifest | 1,069 lines |
| 3 | `probes/` | **the mechanistic core**: difference-of-means, null-space projection, LEACE | 798 lines |
| 4 | `scoring/` | model loading, dataset plumbing, experiment orchestration, metrics | 2,217 lines |
| 5 | `runners/` | CLI entry points — one per experiment arm | 2,432 lines |
| 6 | `review_site/` | static site generator for team review of the setups | 1,548 lines |
| 7 | `cluster/` | hessian.AI 42 cluster deployment | 309 lines |
| — | `tests/` `configs/` | read alongside the stage they cover | 1,756 lines |

`probes/` is small and load-bearing: it is where the actual intervention lives, and where a
subtle error would be least visible in the results. Worth the most attention per line.

## The pipeline in one pass

```
corpus record                     (substrates/)
  -> rendered text + ONE injected marker clause, A/B         (pairs/)
  -> structural gate: single-slot diff, length + readability parity
  -> reward model forward pass -> last-layer hidden state     (scoring/)
  -> difference-of-means direction from contrastive pairs     (probes/)
  -> h' = h - alpha * (v.h) v   before the scalar head
  -> metrics: auto-influence, cross-influence, identity gap   (scoring/)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt        # pulls requirements.txt too

# corpora are user-downloaded and gitignored; the loaders print the exact command
python runners/generate_credit.py
python runners/run_experiment.py --config configs/demographic_credit_sex_qwen06.yaml
```

Running on the cluster: see [`cluster/README.md`](cluster/README.md).

## Provenance

This is a **derivative** of [`drfein/OneBiasAfterAnother`](https://github.com/drfein/OneBiasAfterAnother)
(Apache-2.0), the codebase for arXiv:2603.03291. It is not a fork: the extension is the whole
contribution, and we do not reproduce the upstream results. See [`NOTICE`](NOTICE) for the
required attribution and the list of modifications.

Deliberately **not** carried over:

- the upstream **length / position / sycophancy / uncertainty** arms — we neither run nor
  report them (RQ0b, a LEACE retrofit to those biases, would re-add them deliberately);
- the **MLX (Apple Silicon) backend** — it existed to prototype without GPUs, and was removed
  once CUDA was validated on the hessian.AI cluster. It was the only implementation of the
  `ModelBackend` abstraction, so that went with it;
- the **synthetic CV substrate**, superseded by Bias-in-Bios in 2026-08;
- the **pilot results** (`results/`, `artifacts/`), which came off the MLX prototype.
  Everything is re-run on CUDA.

## Known gaps — read before trusting a number

- **The reasoning arm has not been ported to the real substrate.** It was authored against the
  synthetic `CandidateRecord` and its "experience" claim type reads
  `getattr(record, "years_experience", "several")`, so on a Bias-in-Bios record it does not fail —
  it silently falls back to "several years". It is also hiring-only. Pinned by
  `tests/test_decision_response.py::TestSubstratePortingGap`. (The decision-response arm was ported
  on 2026-09-16: it covers all three domains and reads no record fields.)
- **LEACE has only been applied to the reasoning concepts**, never to a demographic direction.
  The reward-vs-representation claim is scoped accordingly.
- **`auto_influence` is a preference rate.** It saturates at 1.00 for any consistently-signed
  effect and degenerates to noise once an effect is nulled — and both ends are where we read
  it. Report `mean_gap` / `abs_mean_gap` as the primary magnitude alongside it.
- **The Bias-in-Bios scrub leaves residual signal** (+0.10 over chance), of which ~+0.056 is
  the irreducible occupational gender prior. The matched-pair contrast survives; the claim
  that the item is sex-neutral apart from the marker does not.
  *Measured with the scrub before 2026-09-17.* The scrub now also drops bios that still name a
  person by a sex-coded first name (about 36% of otherwise clean bios did) or contain further
  gendered words; `validate_bios_scrub.py` has to be re-run.
- **No confidence intervals or multi-seed runs** anywhere yet.

## Acknowledgement

Required verbatim in any publication using these results:

> We gratefully acknowledge support from the hessian.AI Service Center (funded by the German
> Federal Ministry of Research, Technology and Space, BMFTR, grant no. 16IS22091) and the
> hessian.AI Innovation Lab (funded by the Hessian Ministry for Digital Strategy and
> Innovation, grant no. S-DIW04/0013/003).

It names funders and would deanonymise a double-blind submission — add it at camera-ready, not
at submission.
