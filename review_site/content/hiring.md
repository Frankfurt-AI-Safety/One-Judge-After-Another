title: Hiring — header markers over real biographies
subtitle: Bias-in-Bios professional biographies, scrubbed and screened against a stated target role.
blurb: Real professional biographies with a role-match quality label; replaced the synthetic CV substrate in August 2026.
lead: The reward model screens a candidate profile for a named role; only a protected-attribute clause differs between the two sides.
---
## setup
Each item is a real professional biography from **Bias in Bios** (De-Arteaga et al., 2019),
scrubbed and presented under a header that names the role being screened for. The reward model
is asked to assess the candidate's suitability *for the stated role*.

This substrate replaced a fully synthetic CV generator in August 2026. Two things follow from
that change and are worth flagging up front:

- The synthetic generator is retained, because the **erasure, reasoning and decision-response
  results in the write-up were produced on it** and remain reproducible only from it. Those arms
  have not yet been re-run on real biographies.
- The corpus carries a **real gender label**. It is never rendered into the text — the sex
  signal must come from the injected clause alone — but it is kept for validity checks.

## rationale
**Why real biographies.** The hiring domain was carrying our most load-bearing analyses on
entirely synthetic text, which is the least externally valid data in the project. Recruitment is
also a named Annex III high-risk use, so the realism matters for the claim we want to make.

**Why role-match is the quality label.** Bias-in-Bios has no ordinal quality score, only a
28-way occupation label. Rather than invent a quality heuristic (bio length, seniority words),
we define quality against the dataset's own label: the header names a target role, and a
candidate is *strong* exactly when their true occupation is that role. Roughly half the records
are assigned their own profession and half a different one.

This matters beyond bookkeeping. Cross-influence — does injecting a protected attribute damage
the model's ability to rank a strong candidate above a weak one? — is only interpretable if the
model can judge quality at all. On credit and the old synthetic CVs it cannot (accuracy sits at
chance), which made the metric unreadable there. Role fit is something a language model can
plausibly assess.

**Why the bodies are scrubbed.** The corpus mirror ships text with names and pronouns intact.
Our design needs a neutral body, so the scrub strips the leading name, removes every later
occurrence of it, neutralises gendered pronouns and titles, and drops phone numbers. Biographies
whose name cannot be identified are discarded rather than trusted.

## pipeline
Fetch the corpus once, scrub each biography, assign a balanced role-match label, render under a
header naming the target role, inject the marker clause into the single slot, and run the pair
through the Tier-1 gate.

The body is concatenated verbatim rather than formatted into the template — real biographies
contain braces, and a `.format()` call on them would raise. This is the reason this arm has its
own renderer rather than reusing the synthetic CV one.

## controls
- **The single-axis strip check**, as everywhere: the two sides differ by exactly the clause.
- **The scrub, which the gate cannot check.** The gate compares the two poles *to each other*,
  so residual gendered text appears identically on both sides, cancels in the difference, and
  passes silently. The scrub is therefore validated separately by probing the corpus's **real**
  gender label out of scrubbed activations. **This check has now been run — see the result
  below.**
- **Discarding unresolvable names** rather than keeping them: with ~257k biographies available,
  precision is cheaper than recall.
- **Two templates**, and **held-fixed axes** recorded per pair, as in the other substrates.
- **A known residual risk:** a first name appearing mid-text in a biography that opens without a
  recognisable name span is not removed. The probe check is what would catch this at scale.

## scrub check result
Run on the 0.6B pilot model, 500 probe / 500 eval items, predicting the corpus's **real** gender
label from activations:

| arm | linear probe | MLP probe | chance |
|---|---|---|---|
| scrubbed | 0.666 | 0.608 | 0.566 |
| unscrubbed (control) | 0.994 | 0.974 | 0.566 |

The control arm is strongly decodable, so the probe setup works. The scrub cuts decodability
hard — 0.994 down to 0.666 — but **does not reach chance**. There is residual signal of about
**+0.10** over chance.

**Where it comes from, and why it is not a scrub defect.** Two no-model baselines locate it:

- Predicting gender from **occupation alone** gives 0.622, i.e. **+0.056** — over half the
  residual. The occupations in this corpus are heavily gender-skewed by construction (surgeon
  14% female, nurse 83% female); that skew *is what Bias-in-Bios was built to study*, and no
  name-or-pronoun scrub can remove it without destroying the substrate.
- Predicting from the **stated target role** gives 0.500, *below* chance. The role-match header
  is randomised for the mismatched half of the records, so the design element we added
  introduces no leak. That was the thing most at risk of being our own fault, and it is clean.

That leaves roughly **+0.044** unexplained — plausibly finer lexical correlates (specialisms,
institutions, activities) that also track gender.

**What this does and does not invalidate.**

- It does **not** break the matched-pair contrast. The body is byte-identical across A and B, so
  occupational content sits on both sides and largely cancels in the difference. Auto-influence
  still measures the effect of the injected clause, and a difference-of-means probe built from
  these pairs is dominated by the clause rather than the body.
- It **does** falsify the stronger claim that the item is sex-neutral apart from the marker. It
  is not. Each biography carries an occupational gender prior.
- The consequence is a **congruency interaction** we should measure rather than assume away:
  "the applicant is a man" lands differently on a nurse biography than on a surgeon one. The
  right response is to report congruent and incongruent pairs separately, not to scrub harder.
