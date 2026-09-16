title: Validity checks
subtitle: Do our instruments measure what we claim? The two checks that could invalidate other pages.
blurb: Injected markers vs real demographic fields, and whether the biography scrub actually worked.
lead: These are not results about reward models. They are checks on whether our measurements mean what we say.
---
## setup
Every matched-pair arm on this site rests on the same assumption: that an **injected** marker
stands in adequately for the demographic signal a real document would carry. Two checks test
that assumption directly, and both are open.

**1. Injected versus real direction.** German Credit ships a real `personal_status_sex` field.
We can therefore build a direction from the *real* field and compare it against the direction
built from our *synthetic* marker.

**2. Did the biography scrub work?** The hiring substrate requires that the injected clause is
the only sex signal in the item. The scrub removes names and pronouns from the biography body —
but the structural gate cannot verify that it succeeded.

## rationale
**Why check 1 matters more than its size suggests.** On the one comparison we can make, the
real-field direction is **anti-correlated** with the injected ones: about −0.22 against the
synthetic family-status direction and −0.30 against the synthetic sex direction, with a
real-field reward gap of approximately zero. That is a single arm on a single small model, so it
is not yet a conclusion — but if it holds, it is a result *about the injected-marker method
itself*, and it bears on every number produced with that method. It deserves to be a finding
rather than a footnote, which is why it is scheduled ahead of further breadth.

**Why check 2 cannot be delegated to the gate.** The Tier-1 gate compares the two poles of a
pair *to each other*. Residual gendered text in the body appears identically on both sides,
cancels in the difference, and passes. The gate is not weak here; it is structurally blind to
this failure. So the scrub needs an independent test.

## examples
An early version of the scrub stripped only the *opening* name span. Inspecting real generated
output showed a first name surviving mid-text — *"…Call [name] on phone number …"* — in the
first sampled record. The body claimed one identity while the injected marker claimed another,
and the gate passed it. The scrub now captures the leading name and removes every later
occurrence, and discards biographies whose name it cannot resolve.

That episode is the argument for this page: the failure was invisible to every automatic check
we had, and visible immediately on reading the output.

## pipeline
**Check 1:** build a difference-of-means direction from the real field on balanced groups,
build the synthetic-marker directions the usual way, and report the cosine between them plus the
real-field reward gap, baseline and nulled.

**Check 2:** extract activations for scrubbed biographies and train a linear *and* a non-linear
probe to predict the corpus's **real** gender label, against an unscrubbed control arm. Near
chance on the scrubbed arm means the scrub held. Well above chance means residual leakage is
confounding the sex axis. The unscrubbed arm should be clearly decodable — if it is not, the
probe setup is broken and the scrubbed result means nothing.

## controls
- **The unscrubbed reference arm** in check 2 — without it, "near chance" is indistinguishable
  from a broken probe.
- **Balanced groups** in check 1, so the real-field direction is not a proxy for group size.
- **Both a linear and a non-linear probe**, since a scrub could defeat linear decodability while
  leaving the signal recoverable — the same distinction the erasure arm turns on.
- **Cosine reported per axis**, not pooled, so a single misbehaving axis is visible.

## statistics
**Check 1** has been run once: n = 142 per group, probe accuracy 0.92, cosines −0.216 and
−0.297. Still a single arm on a single small model.

**Check 2 has now been run** (0.6B pilot, 500 probe / 500 eval):

| arm | linear | MLP | chance |
|---|---|---|---|
| scrubbed | 0.666 | 0.608 | 0.566 |
| unscrubbed (control) | 0.994 | 0.974 | 0.566 |

The control is strongly decodable, so the setup works; the scrub takes decodability from 0.994
to 0.666 but not to chance, leaving about **+0.10** of residual signal.

Two no-model baselines locate that residual. Occupation alone predicts gender at 0.622
(**+0.056**), which is most of it — the corpus is gender-skewed by occupation on purpose, and no
scrub removes that. The stated target role predicts at 0.500, *below* chance, so the role-match
header we introduced leaks nothing. About **+0.044** remains unexplained.

**The conclusion is narrower than "the scrub failed".** The matched-pair contrast survives,
because the body is byte-identical across the two poles and largely cancels. What does not
survive is the claim that the item is sex-neutral apart from the injected marker — each
biography carries an occupational gender prior. So the corrective is not a harder scrub but an
analysis change: report **congruent** and **incongruent** pairs separately, since a masculine
marker on a nurse biography is a different item from the same marker on a surgeon's.

This is why the check was worth running before the scaling runs rather than after: it changes
how the hiring sex-axis results have to be reported, not whether they can be produced.
