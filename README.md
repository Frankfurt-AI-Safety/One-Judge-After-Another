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

| # | folder | stage | size | line-by-line review |
|---|---|---|---|---|
| 1 | `substrates/` | real corpora → records, and how a record becomes text | 3,855 lines | **done** (2026-09-24) |
| 2 | `pairs/` | the factorial marker designs, the validation gate, the manifest, the decision responses | 2,254 lines | **done** (2026-09-27) |
| 3 | `probes/` | **the mechanistic core**: difference-of-means, null-space projection, LEACE | 1,418 lines | **done** (2026-09-27) |
| 4 | `scoring/` | model loading, the record split, metrics and their intervals | 2,338 lines | **done** (2026-09-28) |
| 5 | `runners/` | CLI entry points — one per experiment arm | 4,817 lines | **done** (2026-09-30) |
| 6 | `cluster/` | hessian.AI 42 cluster deployment | 591 lines | not yet |
| — | `tests/` `configs/` | read alongside the stage they cover | 8,953 lines | with their stage |

**Review status.** `substrates/` (finished 2026-09-24, including the class-imbalance audit's substrate items),
`pairs/` and `probes/` (both finished 2026-09-27), `scoring/` (finished 2026-09-28) and `runners/` (finished
2026-09-30) have been reviewed line by line; `cluster/` remains. During the `pairs/` session the education corpus became ASAP 2.0 only, which rewrote
`substrates/education_ingest.py` and `education_clean.py` (with tests; see the working notes of 2026-09-27). The
`probes/` session changed what the pipeline does in three places worth knowing before any run: an input longer
than `max_length` is refused (`InputTooLong`) instead of truncated; the embedding cache's fingerprint names the
environment (device, GPU, attention implementation, library versions), which invalidated every earlier cache
once; and the cross-marker geometry reads each cosine against its ceiling √(rel_a · rel_b) from full-sample
reliabilities.

The `scoring/` session changed, before any run:
- **the direct arm's runner is `runners/run_battery.py`.** `runners/run_experiment.py` never ran anything (it
  exited after building the config, from the initial commit on) and was removed with the evaluation half of
  `scoring/experiment.py`; the exporter's auto-influence macros read the battery's cells;
- **the probe split is counted in records only** (`probe_records`; the pair count `probe_size` is gone), and an
  unknown config key is an error;
- **exact ties count ½** in every win rate (rewards are bf16, and ties are common after nulling), and a remaining
  preference is read from the **signed** intervals (`mean_gap`, `pref_a_rate`): auto-influence and |gap| are
  folded and positive under noise alone;
- **model loading:** Gemma-2 (Skywork-Gemma, QRM) loads with eager attention, since sdpa silently dropped its
  attention soft-cap; a model partly offloaded to CPU/disk is refused; results record the Hub commit the weights
  came from; remote code is off by default;
- `scoring/plotting.py` (unused, upstream plots with per-pair error bars) is deleted; figures will be built anew
  from the final result files.

The `runners/` session (108 findings over its 18 scripts, all fixed; working notes 2026-09-28 to 2026-09-30) changed,
before any run:
- **every data file must be regenerated.** Generator version 0.4.0: the manifest names each data file and each
  corpus file by SHA-256, and every runner refuses a data file its manifest does not describe (all data built
  before 0.4.0 is refused). Hiring is fixed at **12,000 bios** (`DEFAULT_N_BIOS`, never changed after the pilot: another
  size relabels the same bios); the education stage design uses every essay × template;
- **every result names its model** (`{arm}_{domain}_{model}[_{encoding}].json`, no more `*_qwen06.json`), carries
  `meta` (the config with the loaded model commit, the code commit, the data files' SHA-256) and is never replaced
  without `--overwrite`; inputs are checked before the model loads; every runner takes the shared overrides
  (`--model --revision --batch-size --device --probe-records`);
- **no direction nulls the items it was fitted on**: the decision, reasoning-flip and A2 arms exclude or hold out
  the probe records; the real-field check nulls leave-one-out and with the synthetic direction;
- **methods changed** (each decided in the session): additivity compares the corner with the *vector* sum and
  reads a three-way residual share with a sign-flip test; the reasoning probe and the LEACE test hold the wording
  out, and the LEACE test states the decision as its own sentence and reads every row against a bag-of-words
  lexical control (the surface of the old verdicts alone gave its "entangled" result); the scrub check reads the
  probe's accuracy beyond the occupation rule, paired, on the manifest's pool, with 1,000 + 1,000 bios; the pilot's
  probe grid runs to N = 500, and its sizing tool applies the pre-stated rule (two-way interactions only, the
  maximum over models);
- `runners/export_paper_numbers.py` reads the new names for `--model`, exports every signed statistic with its
  interval (`<macro>Lo` / `<macro>Hi`), skips results without `meta`, and lists the write-up macros an export
  would break.

`cluster/` has changed a lot since the prototype and has not had its review session yet. Treat its code as
unverified; `cluster/pilot.sh` still uses the old result names.

`probes/` is small and load-bearing: it is where the actual intervention lives, and where a
subtle error would be least visible in the results.

## The pipeline in one pass

```
corpus record                     (substrates/)
  -> rendered text + ONE injected marker clause, A/B         (pairs/)
  -> structural gate: single-slot diff, length + readability parity
  -> reward model forward pass -> last-layer hidden state     (scoring/)
  -> difference-of-means direction from contrastive pairs     (probes/)
  -> h' = h - alpha * (v.h) v   before the scalar head
  -> metrics: auto-influence, identity gap                    (scoring/)
```

That is the **direct-scoring** arm, since 2026-09-24 the *mechanism layer*: the document is recited as
the assistant turn, so it shows that the reward is sensitive to protected attributes under controlled
substitution, and gives the sharpest directions. The **harm evidence** is the cross-marker decision
design (`runners/run_cross_marker.py`): one record in all 8 factorial cells in the USER turn, responses
that name no attribute value (approve / neutral decline / coded decline / overt decline / evasive), and
the disparity of the decision margin D = r(approve) − r(decline) between protected and reference cells,
with decision-format cross-influence on strong and weak records and its own mechanism layer
(cross-fitted prompt / interaction / unfair directions, cross-nulling, placement check). The blatant
decision-response arm (`runners/run_decision_response.py`) is the floor: does the RM at least punish an
openly stated discriminatory verdict?

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt        # pulls requirements.txt too

# corpora are user-downloaded and gitignored; the loaders print the exact command
python runners/generate_credit.py
python runners/run_battery.py --config configs/demographic_credit_sex_qwen06.yaml --axes sex --encodings explicit
# -> artifacts/results/demographic/battery_credit_Skywork-Reward-V2-Qwen3-0.6B__sex__explicit.json
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

- **The reasoning arm is hiring-only.** Its porting gap is closed (2026-09-30): the "experience" claim type,
  which read `getattr(record, "years_experience", "several")` and so had no truth value on a Bias-in-Bios record,
  was deleted; every reasoning item reads no record field but the role
  (`tests/test_decision_response.py::TestSubstratePortingGap`). The blatant decision-response arm covers all
  three domains and reads no record fields.
- **LEACE has only been applied to the reasoning concepts**, never to a demographic direction.
  The reward-vs-representation claim is scoped accordingly. The earlier "entangled" result is not evidence: the
  surface of the old verdicts alone reproduces it (bag-of-words vectors, MLP after LEACE 0.98); the redesigned
  test has not been run.
- **`auto_influence` is a preference rate.** It saturates at 1.00 for any consistently-signed
  effect and degenerates to noise once an effect is nulled — and both ends are where we read
  it. Report `mean_gap` / `abs_mean_gap` as the primary magnitude alongside it.
- **Whether the Bias-in-Bios scrub leaves sex signal beyond the occupation is not established.** The old
  check (+0.10 over chance, ~+0.056 of it the occupational prior) sampled another population (45% professors, no
  profession cap) and compared point estimates whose spread at its size is as large (2026-09-30). The occupation
  itself predicts gender and cannot be removed, so the item is not sex-neutral apart from the marker; the
  matched-pair contrast survives. The redesigned `validate_bios_scrub.py` (the probe beyond the occupation rule,
  paired, with its interval, on the manifest's pool) has to be run.
- **No multi-seed runs** anywhere yet. Every arm now reports cluster-bootstrap 95% intervals (records,
  or essays for A2; `scoring/intervals.py`), the reasoning arm included since 2026-09-30. The intervals are
  uncorrected until the headline family and its multiplicity correction are fixed.
- **Direct-form cross-influence was dropped** (2026-09-24): its premise, that the RM judges applicant
  quality in an off-task recitation, does not hold. It is measured in decision format now; the old runner
  is in the git history.

## Acknowledgement

Required verbatim in any publication using these results:

> We gratefully acknowledge support from the hessian.AI Service Center (funded by the German
> Federal Ministry of Research, Technology and Space, BMFTR, grant no. 16IS22091) and the
> hessian.AI Innovation Lab (funded by the Hessian Ministry for Digital Strategy and
> Innovation, grant no. S-DIW04/0013/003).

It names funders and would deanonymise a double-blind submission — add it at camera-ready, not
at submission.
