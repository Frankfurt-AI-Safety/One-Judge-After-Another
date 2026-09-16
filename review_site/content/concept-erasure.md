title: Concept erasure — LEACE + non-linear probe
subtitle: Is the bias genuinely removed from the representation, or only from the reward?
blurb: The discriminator at the heart of the project — and an honest note about which concept it has actually been run on.
lead: Erase a concept optimally with respect to linear probes, then ask whether a non-linear probe can still recover it.
---
## setup
Null-space projection removes a direction's contribution to the reward. This arm asks a harder
question: is the *concept* gone from the representation, or merely no longer readable by the
linear head?

Three conditions are compared, each scored by both a linear and a non-linear (MLP) probe on a
held-out split:

| condition | what it does |
|---|---|
| `none` | no erasure — the sanity check that both probes can read the concept at all |
| `diffmean` | project out the difference-of-means direction (our method) |
| `leace` | LEACE — provably optimal *linear* erasure |

After LEACE the linear probe drops to chance by construction. **The MLP accuracy is the
answer.** Near chance means the concept is genuinely low-complexity; well above chance means it
is linearly erasable but non-linearly recoverable — entangled.

## rationale
This is the discriminator the project's central claim rests on. If a cheap post-hoc edit makes
an evaluator look unbiased on the metric while the discriminating information remains available
to any downstream non-linear computation, then "we removed the bias" is the wrong description of
what happened — and robustness, not immediate effect, is the load-bearing question for anything
compliance-grade.

**The scope caveat, stated plainly.** This test has so far been run on the **reasoning-correctness
and conclusion concepts** of the model-response arm, in the hiring domain. It has *not* been run
on the demographic directions. The reward-vs-representation gap is therefore established for the
concept that dominates verdict scores, not yet for protected attributes — and closing that gap is
the immediate next experiment. Anywhere the write-up or these pages describe the finding, it
should carry that scope.

## pipeline
Build diversified items so the concept is decorrelated from any single phrase, split train and
eval by applicant so nothing leaks across the split, extract activations, then fit each eraser on
the training split and score both probes on held-out data. The majority-class rate is reported
alongside as the chance baseline.

## controls
- **The `none` condition** is a sanity check: if both probes cannot read the concept before
  erasure, nothing downstream is interpretable.
- **The held-out split, partitioned by applicant**, so probe accuracy is generalisation rather
  than memorisation.
- **Diversified items** (paraphrases, multiple claim types) so the probe cannot latch onto a
  fixed surface string.
- **Two concepts and two premises**, including the non-demographic commute premise, so the
  result is not specific to one framing.
- **`diffmean` alongside `leace`** — our own method next to the provably optimal one, which is
  what makes "linear erasure was not the limitation" a supportable claim.

## statistics
Four cells (two premises × two concepts) × three conditions, with linear accuracy, MLP accuracy
and the chance rate reported for each. The design is small enough to read in full rather than
summarise.
