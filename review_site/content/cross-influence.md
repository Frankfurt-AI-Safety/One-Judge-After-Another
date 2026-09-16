title: Cross-influence (reliability harm)
subtitle: Does injecting a protected attribute damage the model's ability to rank a strong candidate above a weak one?
blurb: The reliability-harm readout — and the built-in flag that tells you when it is not interpretable.
lead: Five renderings per strong/weak pair, comparing quality ranking with and without a protected marker.
---
## setup
Auto-influence asks whether the score moves when an attribute changes. Cross-influence asks
something more consequential: whether the attribute damages the model's **ability to tell good
from bad**. For each pairing of a strong and a weak record we render five variants:

| variant | what it is |
|---|---|
| `strong_neutral` | strong record, no marker |
| `weak_neutral` | weak record, no marker |
| `weak_protected` | weak record carrying the marked pole |
| `weak_reference` | weak record carrying the reference pole |
| `strong_protected` | strong record carrying the marked pole |

`acc_baseline` is how often the model prefers strong over weak with no marker present.
Cross-influence is the drop in that accuracy when the weak candidate carries the protected
marker.

## rationale
This is the metric that maps most directly onto real harm: not "the score shifted" but "the
evaluator got the ranking wrong *because of* the attribute."

It is also the metric with a precondition, and the pipeline enforces it explicitly. If the model
cannot rank quality in the first place, there is no accuracy to damage and the number means
nothing. The runner therefore emits a **`baseline_tracks_quality`** flag, set when `acc_baseline`
clears 0.6, and prints a caveat when it does not.

That flag has been earning its keep. On credit and the synthetic CV substrate the model sits at
chance on quality, so cross-influence there is uninterpretable and is reported as a negative
result rather than a finding. On essay grading — where quality is what reward models natively
score — it clears the bar and the metric can be read. This is the single strongest argument for
the education domain, and for defining hiring quality as role match.

## examples
Each of the five variants is a rendering of a real record from the substrate, built with the same
marker machinery documented on the substrate pages. The `weak_reference` variant is what makes
the comparison two-sided: without it, any accuracy drop could be a marker main-effect rather than
something specific to the *marked* pole.

## pipeline
Split the records into strong and weak by the substrate's quality label, pair them, cycle the
templates, and render all five variants per pair. Score them in one pass, then compute
preference accuracy for each contrast, both at baseline and after null-space projection.

## controls
- **`weak_reference`** — the reference-pole counterpart, so a symmetric marker effect is
  distinguishable from a protected-attribute-specific one.
- **`strong_protected`** — measures whether the marker helps or hurts when attached to the
  strong candidate, which separates a directional bias from a general degradation.
- **`baseline_tracks_quality`** — the interpretability precondition, reported alongside every
  number rather than checked once and forgotten.
- **Nulled and baseline runs** of the same items, so the effect of the intervention is measured
  on identical inputs.

## statistics
Pairs are capped by the smaller of the strong and weak pools. Note that pairing is currently
unconstrained with respect to the target role on the hiring substrate, so a pair may span two
different roles — constraining that is an open task before this arm is re-run there.
