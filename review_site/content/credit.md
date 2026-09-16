title: Credit scoring — header markers
subtitle: Protected-attribute clauses injected into neutral loan-application profiles built from UCI German Credit.
blurb: Real loan applications, rendered as neutral profiles, with a single demographic clause swapped.
lead: The reward model is asked to assess a credit application; only a protected-attribute clause differs between the two sides.
---
## setup
Each item is a short loan-application summary rendered from a **real** German Credit record:
the A-coded attributes (checking status, credit history, purpose, savings, employment, housing,
job) are decoded into plain sentences. The rendered profile is deliberately **pronoun-free and
demographically neutral**, with exactly one slot where a marker clause is injected.

The reward model scores the profile as an assistant turn under a fixed, neutral assessment
prompt. A matched pair is two renderings of the *same* record that differ only in that clause.

Credit is the one substrate on this site shown **in full**: German Credit is CC-BY-4.0, and the
profiles are generated from attribute codes rather than reproduced prose.

## rationale
Creditworthiness is a named high-risk use in Annex III of the EU AI Act, which makes it a
domain where a biased evaluator has a direct legal analogue. Two design choices are worth
challenging:

**Why synthesise the marker instead of using the real field.** German Credit ships a
`personal_status_sex` attribute, but it entangles sex with marital status in a single code, so
it cannot give a clean single-axis contrast. We therefore hold the real field out of the
rendering and inject a controlled clause instead. That buys experimental control at the cost of
realism — and the [validity checks](validity-checks.html) page shows why that trade is not
free: on the one arm where we can compare, the real-field direction is *anti-correlated* with
the injected one.

**Why both explicit and proxy encodings.** An explicit statement ("The applicant is a woman")
is clean but unrealistic; a real application would carry a name or a date. The proxy encodings
(first name, graduation year, employment gap) are what a document would plausibly contain. The
pair of encodings is the interesting comparison, not either alone.

## pipeline
Load the 1,000 German Credit records and decode the A-codes into a neutral profile. For each
(axis, encoding) cell, sample a record and a template, build the two poles by injecting the
marker clause into the single slot, and run the pair through the Tier-1 gate. Keep the
survivors, record the discards.

The gate is the guarantee that makes the whole design work, so it is worth stating precisely.
It checks three things: stripping the clause from each side leaves **byte-identical**
remainders (single-axis), the character and token length deltas are within bounds, and the
Flesch readability delta is within bound. All three must pass.

## controls
- **The single-axis strip check** is the primary control: it is not a heuristic, it is an exact
  string identity, so non-marker drift cannot pass.
- **Two templates** per domain, so an effect that only appears under one phrasing is visible as
  template instability rather than being averaged away.
- **Held-fixed axes** are recorded per pair: when sex varies, age and family status are pinned,
  and the manifest names what was held.
- **Length and readability parity** guard the brittleness confound — a marker that is simply
  *longer* would be a length cue, not a demographic one.
- **The intersection cell** composes the three marginal clauses verbatim, which is what makes
  the additivity test meaningful: the composed direction can be compared against the sum of the
  marginals.
