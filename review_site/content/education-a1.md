title: Education (A1) — header markers over real essays
subtitle: Real student essays from PERSUADE 2.0 and ASAP-AES, graded with a demographic clause in the header.
blurb: Real student essays, body held byte-identical, with the demographic signal in a metadata header.
lead: The reward model grades a student essay; only a header clause naming the student differs between the two sides.
---
## setup
Each item is a **real student essay**, presented for assessment under a short header. The essay
body is copied verbatim and held byte-identical across the pair; the header carries the single
marker slot. This is approach **A1** — identity as incidental metadata, the kind of thing a
grading system might have attached to a submission.

Two corpora are used: **PERSUADE 2.0** (argumentative essays, grades 6–12, holistic score 1–6)
and **ASAP-AES** (eight essay sets, per-set score scales) as a licence-cleaner cross-check.
Quality is thresholded from the holistic score into a clean strong/weak contrast with the middle
dropped.

## rationale
**Why education, and why it matters more than it looks.** Education and vocational assessment
are Annex III high-risk uses, but the methodological reason this domain earns its place is
different: **essay quality is what reward models natively score.** On credit and hiring the model
is at or near chance on the quality label, which makes the reliability-harm metric uninterpretable
there. On essays it is not. This is the domain where the trade-off between removing bias and
preserving legitimate accuracy can actually be measured.

**Why the body is never touched.** The essay is real text we do not control, so the only
defensible design is to leave it exactly as it is and put the entire intervention in the header.
That also makes the strip check trivially exact.

**Why a name proxy for ethnicity.** There is no clean explicit form for ethnicity on a student
submission that would not be obviously artificial, so we use a name grid in the
Bertrand–Mullainathan / Haim lineage. This is the standard method and it is also the weakest
axis we run: names conflate ethnicity with class, region and religion, so the result is properly
described as a *name-proxy effect*, not a pure ethnicity effect. The name pools are flagged in
the code as an unvalidated starter set, and validating them is an open task.

## pipeline
Load the corpus, drop essays outside a length band, threshold the holistic score into strong or
weak (dropping the middle), and neutralise redaction tags so that injected names are the only
name signal. Then, per (axis, encoding) cell: render the header, inject the clause, gate the
pair, keep the survivors.

The renderer formats **only the header** and concatenates the essay body — real essays contain
braces, and formatting the whole string would raise on them.

## controls
- **The single-axis strip check**, with the essay body byte-identical by construction.
- **The ethnicity grid holds sex fixed.** Both poles of an ethnicity pair use names coded to the
  same sex, chosen per pair; the held sex is recorded in the pair's exemplar. Symmetrically, the
  sex axis draws both poles from the same ethnicity-coded pool.
- **Two corpora.** PERSUADE and ASAP are independent substrates with different prompts, grade
  ranges and scoring scales, so an effect appearing in one and not the other is visible as
  corpus instability rather than a finding.
- **Two header templates**, for the same reason at the phrasing level.
- **A known imbalance to look at:** the held-sex choice is an independent coin flip per pair,
  not a stratified split, so the balance is multinomial noise rather than exactly even. The
  statistics section below shows the realized split.
