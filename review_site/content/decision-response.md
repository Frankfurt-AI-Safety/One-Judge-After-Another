title: Decision-response arm
subtitle: The reward model scores a hiring recommendation about the applicant, rather than the applicant.
blurb: Would the reward model reward a policy that discriminates? Fair vs discriminatory verdicts on the same candidate.
lead: Same applicant, four candidate responses. The question is which recommendation the model prefers.
---
## setup
Instead of scoring the applicant profile, this arm scores a **response about** the applicant —
a short hiring recommendation. For one candidate we author four responses and score each:

- **fair** — advances the candidate and explicitly says the protected attribute is irrelevant;
- **discriminatory** — advances or declines while citing the protected attribute as a reason;
- **neutral** — advances on the merits without mentioning the attribute at all;
- **evasive** — declines to give a recommendation either way.

The headline number is the rate at which the discriminatory response outscores the fair one.

## rationale
This is the closest we currently get to the claim that motivates the whole project: a reward
model is reused across RLHF, best-of-*N* selection and data filtering, so its preferences
propagate into policy behaviour. If an evaluator prefers a discriminatory recommendation, a
policy optimised against it inherits that preference.

Two honest caveats belong on the same page as that claim:

**The verdicts are authored, not generated.** They are templated text we wrote, length-matched
across variants. That buys control — the four responses differ in exactly the intended way —
at the cost of realism. A referee arm using model-generated verdicts is designed but not built.

**This is a proxy for propagation, not a measurement of it.** Nothing here trains or samples a
policy. Whether a debiased reward model actually yields a less biased policy is untested, and it
is the follow-up the project has not yet run.

## pipeline
Take a strong applicant record, render it with the protected-attribute clause, and attach the
fixed screening question. Then generate the four responses from a shared scaffold so that
openings, connectives and closings are common and only the middle clause varies. Score each
response independently.

## controls
- **The evasion control.** A non-committal answer must not outscore a substantive fair one. If
  it does, the metric is rewarding hedging rather than fairness, and the fair-vs-discriminatory
  comparison cannot be read at face value.
- **The neutral variant** separates "mentions the attribute approvingly" from "does not mention
  it at all" — without it, the fair response's explicit fairness language is a confound.
- **Length matching across variants**, reported by the generator, so the comparison is not a
  length preference in disguise.
- **Multiple axes**, including the intersectional one, since the effect concentrates on the
  legally salient family-status and intersectional axes rather than on sex or age.

## statistics
The item space is fully enumerable: four verdict variants × four axes × two templates, over the
strong-labelled records of the substrate. Items are constructed deterministically from a seed, so
the examples above reproduce exactly.
