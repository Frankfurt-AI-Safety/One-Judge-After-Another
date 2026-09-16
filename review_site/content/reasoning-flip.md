title: Reasoning-flip 2×2
subtitle: Does the reward model reward factually correct reasoning, or the conclusion it prefers?
blurb: A factorial design separating reasoning quality from demographic favourability — and the control that showed they were confounded.
lead: Four responses about the same applicant, crossing whether the causal claim is true with whether the recommendation advances.
---
## setup
A 2×2 factorial over responses to the same applicant. Each response makes a causal claim about
a premise (for example, being on parental leave) and reaches a recommendation:

|  | advance | reject |
|---|---|---|
| **true claim** | true_advance | true_reject |
| **false claim** | false_advance | false_reject |

Crossing the two factors separates a *correctness* effect (does the model reward the factually
accurate claim?) from a *conclusion* effect (does it reward advancing regardless?), plus their
interaction.

## rationale
This arm exists because of a result it produced rather than one it was designed to produce.

The intent was to ask whether the model's preferences track demographic fairness. What the
factorial showed is that verdict scores are dominated by **reasoning quality and positivity**,
not by fairness — and the non-demographic control makes that unambiguous: the correctness effect
on a demographic premise is essentially identical to the effect on a commute premise. The model
is rewarding accurate causal reasoning in general, not fairness in particular.

That is a construct-validity finding about the decision-response arm, and it is why the two
pages should be read together. It also motivated the erasure work: if reasoning quality is what
dominates verdict scores, then reasoning quality is the concept worth erasing to see whether it
is entangled.

## pipeline
Take a strong applicant record, attach the premise clause, and construct four responses from a
shared scaffold in which only the truth value of the causal claim and the direction of the
recommendation vary. Score each independently and compute the two main effects and their
interaction.

A diversified mode substitutes paraphrases and alternates the claim type (availability versus
experience), so the concept is decorrelated from any single phrase — that mode is what the
probe and erasure runs consume.

## controls
- **The commute premise** is the load-bearing control: a non-demographic premise with the same
  grammatical shape. If the correctness effect is the same size there, the effect is
  domain-general reasoning quality rather than anything about demographics. It is.
- **Paraphrase pools and two claim types**, so a probe trained on these activations is learning
  the concept rather than a surface phrase.
- **Strong applicants only.** Items are built from records the substrate labels strong, so
  "should this candidate advance" has a defensible answer and the conclusion factor is not
  confounded with genuine merit.
- **The 2×2 itself is the control structure** — neither main effect is interpretable without
  the other cell, which is the point of a factorial over a one-at-a-time comparison.

## statistics
Three premises (two demographic, one control) × four cells × two templates, constructed
deterministically from a seed. The diversified mode multiplies this by two claim types and three
paraphrase stems per truth value.
