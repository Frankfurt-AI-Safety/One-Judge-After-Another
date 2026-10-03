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
| 6 | `cluster/` | hessian.AI 42 cluster deployment | ~800 lines | revised for the pilot (2026-10-01) |
| — | `tests/` `configs/` | read alongside the stage they cover | 8,953 lines | with their stage |

**Review status.** `substrates/` (finished 2026-09-24, including the class-imbalance audit's substrate items),
`pairs/` and `probes/` (both finished 2026-09-27), `scoring/` (finished 2026-09-28) and `runners/` (finished
2026-09-30) have been reviewed line by line; `cluster/` was revised for the pilot re-run on 2026-10-01 (see below). During the `pairs/` session the education corpus became ASAP 2.0 only, which rewrote
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

**Built after the reviews and reviewed separately** (built 2026-09-30, reviewed 2026-10-01 by three independent
reviewers, 31 findings, all fixed; working notes of both dates): the comparative (two-applicant) arm —
`pairs/comparative.py`, `scoring/comparative_metrics.py`, `probes/comparative_directions.py`,
`runners/run_comparative.py`, three `configs/*comparative*` and their tests — plus its support in
`runners/pilot_sizing.py`. The review changed, before any run: the pairing pools are split from the strata (equal
capacity per pairing by exact quotas; credit ≈ 60–65 pairs each), the pairs no longer depend on `--encodings`/`--templates`/
`--directions`, the strong–weak statistics are read against the unmarked prompts (exchange rate, overturn, rescue),
and every setting changed on the CLI enters the result name — in `run_cross_marker.py` too.

`cluster/` was revised for the pilot re-run on 2026-10-01, without a separate review round (the user's choice, to
run the pilot): `stage.sh` stages a committed tree only and records its commit (`STAGED_COMMIT`, read by
`pairs.manifest.code_provenance`, since the staged copy has no `.git`); `prepare_data.sh` regenerates every manifest
on the cluster and `check_data.py` checks them; `pilot.sh` runs the sizing steps and a test run of every arm in the
lanes `smoke`, `small`, `8b` and `70b` (in parallel on the workspace's 4 GPUs). The lanes were dry-run locally
(`smoke` on MPS); the cluster run is their real test.

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
openly stated discriminatory verdict? The **comparative arm** (`runners/run_comparative.py`, exploratory) puts two
applications in the USER turn and scores the choice of one: record pairs strong–strong, strong–weak and weak–weak,
the applicants differing in one axis (or all three), each pair under both marker assignments and both orders, so
the records and the position cancel from the marker effect r(choose protected) − r(choose reference). The
**placement matrix** (`runners/run_placement_matrix.py`, exploratory, built and reviewed 2026-10-01) scores the
comparative pairs' records in all three placements and nulls each placement's direction in the others, with one fold
assignment and one bootstrap over pairs. Across **domains and encodings** (`runners/run_demographic_transfer.py`,
exploratory, built and reviewed once 2026-10-03) every direct-arm direction is projected out of every domain's direct gap and decision
disparity, with proxy first names held out of the fit.

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

- **The reasoning arm runs on all three domains since 2026-10-01, but its premises differ in strength.** Per
  domain (`pairs.verdicts.REASONING_FRAMES`): hiring parental leave → availability, control six months of travel
  abroad (it replaced a long commute); credit age 30 → fewer years to build a credit history, control an unpaid
  sabbatical → less income (another claim, bearing on repayment more directly; credit's 13 records whose employment
  or job reads unemployed are not drawn); education low income → a fee-charging writing program is harder to afford,
  control the higher out-of-district fee (not pass/fail grading, which no claim about the household bears on). Each
  intersection premise is the factorial's corner and names the whole identity as the cause, though its claim follows
  from one component. Age is partly a legitimate
  credit risk factor, so "sound but discriminatory" is weaker there. The control is nulled with each demographic
  direction too (a placebo), and `nulling_vs_control` reports what nulling changes beyond that. The reasoning
  probe, the erasure test and the cross-domain transfer (`runners/run_reasoning_transfer.py`; new, reviewed and
  redesigned twice on 2026-10-01) state the decision without a connective, add a favourable-truth premise per domain
  (no notice period / a permanent pay raise / the in-district fee) so truth and valence are crossed, fit correctness
  and valence across the two non-demographic premises, and hold the wording out in three folds of six-entry pools
  (two held out each, with no word that separates true from false shared between entries of any claim). The
  erasure test erases correctness and valence jointly (pooled over the two premises, each is the other XOR the
  premise). Whether a correctness direction is a reasoning direction is read against a bag-of-words lexical control
  (erasure test, transfer runner) and, in the probe, from the valence check (with a positive control; inconclusive
  where the direction does not transfer between premises). The criteria a premise must meet are in the
  working notes (2026-10-01). Every reasoning item reads no record field but the role
  (`tests/test_decision_response.py::TestSubstratePortingGap`; the "experience" claim, which had no truth value on
  a Bias-in-Bios record, was deleted on 2026-09-30). The blatant decision-response arm covers all three domains and
  reads no record fields.
- **LEACE on the demographic attributes is built but has not run** (`runners/run_demographic_erasure.py`, built
  and reviewed once 2026-10-02): the direct arm's states, every axis and its two-way interactions, LEACE on the
  concept and on every cell, held-out proxy names (crossed records × names intervals), a bag-of-words lexical control
  of the marker clause and the reward gap after each erasure. Until it runs, the reward-vs-representation claim is scoped to the reasoning concepts,
  whose earlier "entangled" result is not evidence either: the surface of the old verdicts alone reproduces it
  (bag-of-words vectors, MLP after LEACE 0.98); the redesigned test has not been run.
- **The RewardBench 2 accuracy guardrail (RQ4) is built but has not run** (`runners/run_rewardbench_guardrail.py`,
  built 2026-10-02, reviewed once 2026-10-03): every direct-arm direction, LEACE eraser and joint projection against the
  unedited model on RewardBench 2, non-inferiority at 2 points (the base paper used 5), after the unedited scores
  reproduce the published ones.
- **The transfer of the demographic directions across domains and encodings (RQ5) is built but has not run**
  (`runners/run_demographic_transfer.py` + `probes/transfer_directions.py`, built and reviewed once 2026-10-03):
  every direct-arm direction (domain × encoding × axis) projected out of every target, on the direct gap and on the
  cross-marker decision disparity, each row labelled by what it shares with the target (own, encoding, domain; a
  component where an intersection and a single axis overlap; off-axis and unrelated as the specificity controls); a
  held-out-template row, the single-axis directions against the intersection, a pooled other-domains direction and
  random directions; proxy first names held out of every fit (the domains share the name pools); a paired-accuracy
  reading with a bag-of-words and a bag-of-tokens control of the marker clauses. Only sex (three domains) and age
  (credit, hiring) exist in more than one domain, and credit and hiring share most marker words. Proxy targets'
  intervals condition on the name pools.
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
- **The comparative arm has not run on the cluster** (built 2026-09-30, exploratory until the headline family is
  fixed; pair counts from the pilot). Its own direction, the direct → comparative transfer and the placement matrix
  (`runners/run_placement_matrix.py`, built and reviewed 2026-10-01: direct / cross-marker / comparative
  directions on the comparative pairs' records, cross-fitted over the comparative pair folds) are built; LEACE with a
  non-linear probe on the comparative states is not. The tiny test models cannot show its direction: their
  last-token state does not register a marker swap ~600 tokens back (the matrix is tested on planted states).
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
