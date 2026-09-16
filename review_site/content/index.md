title: Experiment review — reward-model bias
subtitle: What each experiment presents to the reward model, why it is built that way, and where to leave feedback.
---
## what-this-is
These pages exist so the working group can review the **experiment settings** before the
scaling runs spend the cluster allocation. Every page shows the same six things: the setup,
the reasoning behind the design, real contrastive pairs as the reward model receives them, how
the data pipeline produces them, which controls are in place, and descriptive statistics for
the generated set.

Everything except the prose is **generated from the pipeline's own artifacts** — the manifests
and matched-pair files the generators write — so a page cannot quietly drift from the code it
documents. Where a page states a number, that number was recomputed from the data at build
time, not typed in.

## the-core-design
Every matched-pair arm rests on one idea. Take a piece of real content, hold it
**byte-identical**, and change exactly one thing: a clause naming a protected attribute. Any
difference in the reward model's score is then attributable to that clause and nothing else.

A structural gate enforces this before a pair is kept: the two sides must differ by exactly the
injected clause, and be within bounds on length and readability. Pairs that fail are discarded
and the discard rate is recorded.

## what-to-look-for
The most useful feedback is on **construct validity** — whether the thing we built measures the
thing we claim:

- Do the injected clauses read naturally in context, or do they sound bolted on?
- Are the two poles genuinely matched in register and length, or is one subtly more fluent?
- Are the controls the right controls, and is anything missing?
- For proxies (names, graduation years, career gaps): is the proxy plausibly what a real
  document would contain?

Comments are per page. Quote the section heading you are responding to.

## a-note-on-excerpts
The corpora behind these experiments are not all redistributable: PERSUADE is CC BY-NC-SA and
our data README commits to not republishing derived essays, and Bias-in-Bios contains
biographies of identifiable real people. So on those pages the examples show a **window around
the injected clause** rather than the full body, with the length and provenance stated. German
Credit is CC-BY-4.0 and is shown in full.

This costs very little for review purposes: the body is held byte-identical across the pair, so
it carries none of the contrast being judged. Where topical context matters — mainly for
judging whether a standpoint sentence fits its essay — the pages show corpus **metadata**
instead.
