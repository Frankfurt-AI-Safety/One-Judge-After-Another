#!/usr/bin/env python3
"""
Build the experiment-review website from the pipeline's own artifacts.

The site exists so the team can review the experiment settings *before* the scaling runs spend
the cluster allocation: what each setup presents to the reward model, why it is built that way,
which controls it carries, and what the generated set actually looks like.

Design:

- **Generated, not written.** Examples, statistics, template hashes and thresholds all come from
  `manifest.json` / `pairs.jsonl` or from calling the item builders directly, so the pages
  cannot drift from the pipeline. Only the prose (`site/content/*.md`) is hand-written.
- **Output goes to a separate repo** via `--out`, the same split
  `experiments/export_paper_numbers.py` uses to write into the paper repo. The research repo
  keeps the build inputs; the site repo holds only rendered HTML.
- **Redistribution is enforced here, not trusted.** Every corpus excerpt goes through
  `review_site/redact.py`, and `assert_no_leak` re-checks the finished HTML before it is
  written, so a mis-rendered page cannot be published even if the tests were skipped.

Runs fully offline: no model, no GPU, no network.

Usage:
    python experiments/build_review_site.py --out ../rm-bias-experiment-review
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from review_site import charts, stats
from review_site.redact import assert_no_leak, excerpt, get_corpus
from review_site.render import (
    GiscusConfig,
    Page,
    STYLESHEET,
    Section,
    esc,
    md,
    render_index,
    render_page,
)

DATA_ROOT = PROJECT_ROOT / "data" / "demographic"
CONTENT_ROOT = PROJECT_ROOT / "site" / "content"
GISCUS_PATH = PROJECT_ROOT / "site" / "giscus.json"
REPO_URL = "https://github.com/nondatur/OneBiasAfterAnotherFork"

GROUP_SUBSTRATE = "Matched-pair substrates"
GROUP_RESPONSE = "Model-response arms"
GROUP_READOUT = "Readouts & validity"


# --------------------------------------------------------------------------- content files
def parse_content(path: Path) -> Dict[str, Any]:
    """Parse a content file: `key: value` metadata, `---`, then `## section-id` blocks."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing page content {path}. Every page needs hand-written prose; create it "
            f"with at least a title and the sections the builder asks for."
        )
    raw = path.read_text()
    meta_block, _, body = raw.partition("\n---\n")
    meta: Dict[str, str] = {}
    for line in meta_block.splitlines():
        if ":" in line and not line.startswith("#"):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()

    sections: Dict[str, str] = {}
    current, buf = None, []
    for line in body.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return {"meta": meta, "sections": sections}


def prose(content: Dict[str, Any], key: str, *, required: bool = True) -> str:
    text = content["sections"].get(key, "")
    if not text and required:
        raise KeyError(
            f"Content file is missing a '## {key}' section. The page template expects it."
        )
    content.setdefault("_used", set()).add(key)
    return md(text)


def extra_sections(content: Dict[str, Any]) -> List[Section]:
    """Render any sections the page template did not consume, rather than dropping them silently."""
    used = content.get("_used", set())
    return [
        Section(k, k.replace("-", " ").replace("_", " ").capitalize(), md(v))
        for k, v in content["sections"].items()
        if k not in used and v.strip()
    ]


# --------------------------------------------------------------------------- shared blocks
def facts_table(d: stats.DatasetStats) -> str:
    corpus = get_corpus(d.key)
    rows = [
        ("Substrate", f"{corpus.label} ({corpus.licence})"),
        ("Dataset", f"<code>data/demographic/{esc(d.key)}/</code>"),
        ("Matched pairs", f"{d.n_records:,}"),
        ("Cells (axis × encoding)", str(len(d.cells))),
        ("Generator version / seed", f"{esc(d.generator_version)} / {d.seed}"),
        ("Gate discard rate", f"{d.total_discard_rate:.1%}"),
        ("Templates", ", ".join(f"<code>{esc(t)}</code>" for t in sorted(d.templates))),
    ]
    body = "".join(f"<tr><th>{esc(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table class='facts'><tbody>{body}</tbody></table>"


def pair_block(rec: Dict[str, Any], corpus_key: str) -> str:
    """One contrastive pair, side by side, with the injected clause highlighted."""
    ex_a = excerpt(rec["text_a"], rec["clause_a"], corpus_key=corpus_key)
    ex_b = excerpt(rec["text_b"], rec["clause_b"], corpus_key=corpus_key)
    meta_bits = [
        f"<span><strong>axis:</strong> {esc(rec['varied_axis'])}</span>",
        f"<span><strong>encoding:</strong> {esc(rec['encoding'])}</span>",
        f"<span><strong>template:</strong> {esc(rec['template_id'])}</span>",
    ]
    held = rec.get("held_fixed") or []
    if held:
        meta_bits.append(f"<span><strong>held fixed:</strong> {esc(', '.join(held))}</span>")
    exemplar = {k: v for k, v in (rec.get("exemplar") or {}).items() if isinstance(v, (str, int))}
    if exemplar:
        bits = ", ".join(f"{esc(k)}={esc(v)}" for k, v in exemplar.items())
        meta_bits.append(f"<span><strong>drawn:</strong> {bits}</span>")
    return (
        f'<div class="meta-row">{"".join(meta_bits)}</div>'
        f'<div class="pair">'
        f'<div><div class="pole">A — {esc(rec.get("label_a", "a"))}</div>'
        f'<div class="rendered">{ex_a.to_html()}</div>'
        f'<div class="provenance">{esc(ex_a.provenance_note())}</div></div>'
        f'<div><div class="pole">B — {esc(rec.get("label_b", "b"))}</div>'
        f'<div class="rendered">{ex_b.to_html()}</div>'
        f'<div class="provenance">{esc(ex_b.provenance_note())}</div></div>'
        f"</div>"
    )


def examples_section(d: stats.DatasetStats, *, per_axis: int = 1) -> tuple:
    """One featured pair per axis, with the remaining encodings behind a disclosure.

    Returns (html, texts_used) so the caller can run the leak assertion over the real bodies.
    """
    axes: List[str] = []
    for c in d.cells:
        if c.axis not in axes:
            axes.append(c.axis)

    out, texts = [], []
    for axis in axes:
        cells = [c for c in d.cells if c.axis == axis]
        # Explicit first where both exist, then the proxy. Both are rendered inline rather than
        # behind a disclosure: the explicit-vs-proxy contrast is one of the main things being
        # reviewed, so it should not cost a click.
        cells.sort(key=lambda c: (c.encoding != "explicit", c.encoding))
        out.append(f"<h3>{esc(axis)}</h3>")
        for c in cells:
            recs = stats.examples(DATA_ROOT, d.key, axis, c.encoding, n=per_axis)
            if not recs:
                continue
            if len(cells) > 1:
                out.append(f"<h4>{esc(axis)} <span class='muted'>({esc(c.encoding)})</span></h4>")
            for r in recs:
                out.append(pair_block(r, d.key))
                texts += [r["text_a"], r["text_b"]]
    return "".join(out), texts


def stats_section(d: stats.DatasetStats) -> str:
    rows = []
    for c in d.cells:
        rows.append(
            f"<tr><td>{esc(c.axis)}</td><td>{esc(c.encoding)}</td><td>{c.n}</td>"
            f"<td>{esc(c.label_a)} / {esc(c.label_b)}</td>"
            f"<td>{c.distinct_sources}</td>"
            f"<td>{c.max_char_delta} / {d.threshold_for(c.axis)}</td>"
            f"<td>{c.max_token_delta} / {d.thresholds.get('max_token_delta', '—')}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>axis</th><th>encoding</th><th>pairs</th><th>poles (A / B)</th>"
        "<th>distinct source records</th><th>max |Δchars| / gate</th>"
        "<th>max |Δtokens| / gate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    out = [table]

    # Does the gate bind anywhere? One row per cell, since the bound differs by axis.
    out.append(
        "<h3>Length parity against the gate</h3>"
        "<p class='muted'>The largest realized character difference in each cell, against the "
        "bound that cell has to pass. Bars are drawn as a share of the gate because the bound is "
        "not the same for every axis. If everything sits well short of the line, the 0% discard "
        "rate reflects genuinely length-matched markers rather than a permissive gate.</p>"
    )
    out.append(
        charts.gate_headroom(
            [(f"{c.axis} / {c.encoding}", c.max_char_delta, d.threshold_for(c.axis))
             for c in d.cells],
            caption="Worst-case |Δchars| per cell, as a share of that cell's gate.",
        )
    )

    # A histogram only where the difference actually varies. Marker clauses are fixed strings, so
    # in most cells it is a single constant and a histogram would be a one-bar chart.
    varying = [c for c in d.cells if len(set(c.char_deltas)) > 2]
    if varying:
        out.append(
            "<p class='muted'>Only the name-proxy cells vary at all — names differ in length, "
            "whereas every other clause is a fixed string, so its length difference is constant "
            "across the cell. Those distributions:</p>"
        )
        for c in varying:
            out.append(
                charts.delta_histogram(
                    c.char_deltas,
                    threshold=d.threshold_for(c.axis),
                    caption=f"{c.axis}/{c.encoding} — realized |Δchars| across {c.n} pairs.",
                )
            )

    # Marker draw distributions, only where a pool is actually sampled.
    pooled = [c for c in d.cells if c.samples_a_pool]
    if pooled:
        for c in pooled:
            for fname, counter in sorted(c.exemplar_counts.items()):
                if len(counter) < 2:
                    continue      # a constant is not a distribution
                out.append(
                    charts.draw_distribution(
                        dict(counter),
                        caption=f"{c.axis}/{c.encoding} — <{fname}> draws across {c.n} pairs.",
                    )
                )
    else:
        out.append(
            '<p class="muted">No marker pools are sampled in this dataset: every clause is a '
            "fixed constant, so there is no draw distribution to show.</p>"
        )

    constants = []
    for c in d.cells:
        for fname, counter in sorted(c.exemplar_counts.items()):
            if len(counter) == 1:
                constants.append((c.axis, fname, next(iter(counter))))
    if constants:
        body = "".join(
            f"<tr><td>{esc(a)}</td><td>{esc(f)}</td><td>{esc(v)}</td></tr>" for a, f, v in constants
        )
        out.append(
            "<h3>Fixed (non-sampled) clause values</h3>"
            "<p class='muted'>These are constants rather than draws, so they have no "
            "distribution — listed so the wording is reviewable.</p>"
            f"<table><thead><tr><th>axis</th><th>field</th><th>value</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    return "".join(out)


def pipeline_section(d: stats.DatasetStats, manual: str) -> str:
    thr = d.thresholds
    thr_rows = "".join(
        f"<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>" for k, v in sorted(thr.items())
    )
    tmpl_rows = "".join(
        f"<tr><td><code>{esc(t)}</code></td><td><code>{esc(h)}</code></td></tr>"
        for t, h in sorted(d.templates.items())
    )
    disc_rows = "".join(
        f"<tr><td>{esc(cell)}</td><td>{v.get('kept', 0)}</td><td>{v.get('examined', 0)}</td>"
        f"<td>{v.get('discard_rate', 0):.1%}</td>"
        f"<td>{esc(v.get('failure_reasons') or '—')}</td></tr>"
        for cell, v in sorted(d.discard_report.items())
    )
    return (
        manual
        + "<h3>Validation gate thresholds</h3>"
        + f"<table><thead><tr><th>bound</th><th>value</th></tr></thead><tbody>{thr_rows}</tbody></table>"
        + "<h3>Templates (content-hashed)</h3>"
        + f"<table><thead><tr><th>template</th><th>sha256[:12]</th></tr></thead><tbody>{tmpl_rows}</tbody></table>"
        + "<h3>Gate outcome per cell</h3>"
        + "<table><thead><tr><th>cell</th><th>kept</th><th>examined</th><th>discard</th>"
        + f"<th>failure reasons</th></tr></thead><tbody>{disc_rows}</tbody></table>"
        + f"<p class='provenance'>Attribution recorded in the manifest: {esc(d.attribution)}</p>"
    )


# --------------------------------------------------------------------------- page builders
@dataclass
class DatasetPageSpec:
    slug: str
    key: str
    group: str = GROUP_SUBSTRATE


def build_dataset_page(spec: DatasetPageSpec) -> Page:
    d = stats.load(DATA_ROOT, spec.key)
    content = parse_content(CONTENT_ROOT / f"{spec.slug}.md")
    meta = content["meta"]

    ex_html, ex_texts = examples_section(d)
    assert_no_leak(ex_html, ex_texts, corpus_key=spec.key)

    sections = [
        Section("setup", "Setup", prose(content, "setup") + facts_table(d),
                lead=meta.get("lead", "")),
        Section("rationale", "Why it is designed this way", prose(content, "rationale")),
        Section("examples", "What the reward model sees", ex_html,
                lead="One matched pair per axis. Only the highlighted clause differs; "
                     "everything else is byte-identical."),
        Section("pipeline", "How the data pipeline works",
                pipeline_section(d, prose(content, "pipeline"))),
        Section("controls", "Controls", prose(content, "controls")),
        Section("statistics", "Descriptive statistics", stats_section(d)),
        *extra_sections(content),
    ]
    return Page(slug=spec.slug, title=meta.get("title", spec.slug),
                subtitle=meta.get("subtitle", ""), sections=sections,
                group=spec.group, blurb=meta.get("blurb", ""))


def build_response_page(slug: str, builder: Callable[[], tuple]) -> Page:
    """A model-response arm: items are constructed in code, so examples are rendered live."""
    content = parse_content(CONTENT_ROOT / f"{slug}.md")
    meta = content["meta"]
    ex_html, ex_texts, corpus_key = builder()
    assert_no_leak(ex_html, ex_texts, corpus_key=corpus_key)
    sections = [
        Section("setup", "Setup", prose(content, "setup"), lead=meta.get("lead", "")),
        Section("rationale", "Why it is designed this way", prose(content, "rationale")),
        Section("examples", "What the reward model sees", ex_html,
                lead="The applicant profile is shared; the scored responses differ."),
        Section("pipeline", "How the items are built", prose(content, "pipeline")),
        Section("controls", "Controls", prose(content, "controls")),
        Section("statistics", "Item space", prose(content, "statistics")),
    ]
    return Page(slug=slug, title=meta.get("title", slug), subtitle=meta.get("subtitle", ""),
                sections=sections, group=GROUP_RESPONSE, blurb=meta.get("blurb", ""))


def build_prose_page(slug: str, group: str) -> Page:
    """A readout or validity page: no generated dataset behind it, prose plus constants."""
    content = parse_content(CONTENT_ROOT / f"{slug}.md")
    meta = content["meta"]
    order = [("setup", "Setup"), ("rationale", "Why it is designed this way"),
             ("examples", "What is compared"), ("pipeline", "How it is computed"),
             ("controls", "Controls"), ("statistics", "What to look at")]
    sections = [
        Section(sid, title, prose(content, sid))
        for sid, title in order if content["sections"].get(sid)
    ]
    return Page(slug=slug, title=meta.get("title", slug), subtitle=meta.get("subtitle", ""),
                sections=sections, group=group, blurb=meta.get("blurb", ""))


def _decision_examples() -> tuple:
    from substrates.domains import get_domain
    from pairs.verdicts import VERDICT_VARIANTS, build_decision_item

    dom = get_domain("cv")
    records = dom.load_records()[:40]
    rng = random.Random(7)
    out, texts = [], []
    for axis in ("sex", "intersection"):
        item = build_decision_item(records[0] if axis == "sex" else records[1], axis, "explicit",
                                   dom.render_fn, rng, template_id=dom.template_ids[0])
        clause = item["meta"].get("protected_pole", "")
        ex = excerpt(item["user_prompt"], clause, corpus_key="cv")
        texts.append(item["user_prompt"])
        rows = "".join(
            f'<div><div class="pole">{esc(v)}</div>'
            f'<div class="rendered">{esc(item["verdicts"][v])}</div></div>'
            for v in VERDICT_VARIANTS if v in item["verdicts"]
        )
        out.append(
            f"<h3>{esc(axis)}</h3>"
            f'<div class="meta-row"><span><strong>scored object:</strong> the response, '
            f"not the applicant</span></div>"
            f'<div class="rendered">{ex.to_html()}</div>'
            f'<div class="provenance">{esc(ex.provenance_note())}</div>'
            f'<div class="pair">{rows}</div>'
        )
    return "".join(out), texts, "cv"


def _reasoning_examples() -> tuple:
    from substrates.domains import get_domain
    from pairs.verdicts import REASONING_CELLS, build_reasoning_item

    dom = get_domain("cv")
    records = dom.load_records()[:40]
    rng = random.Random(11)
    out, texts = [], []
    for premise in ("parental_leave", "commute"):
        item = build_reasoning_item(records[2], premise, dom.render_fn, rng,
                                    template_id=dom.template_ids[0])
        clause = item["meta"].get("premise_clause", "")
        ex = excerpt(item["user_prompt"], clause, corpus_key="cv")
        texts.append(item["user_prompt"])
        rows = "".join(
            f'<div><div class="pole">{esc(c)}</div>'
            f'<div class="rendered">{esc(item["cells"][c])}</div></div>'
            for c in REASONING_CELLS if c in item["cells"]
        )
        demo = item["meta"].get("demographic")
        out.append(
            f"<h3>{esc(premise)} <span class='muted'>"
            f"({'demographic premise' if demo else 'non-demographic control'})</span></h3>"
            f'<div class="rendered">{ex.to_html()}</div>'
            f'<div class="provenance">{esc(ex.provenance_note())}</div>'
            f'<div class="pair">{rows}</div>'
        )
    return "".join(out), texts, "cv"


# --------------------------------------------------------------------------- main
def main() -> None:
    global DATA_ROOT

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="Path to the site repo checkout")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = ap.parse_args()
    DATA_ROOT = args.data_root

    generated = _dt.date.today().isoformat()
    giscus = GiscusConfig.load(GISCUS_PATH)

    pages: List[Page] = []
    for spec in (
        DatasetPageSpec("credit", "credit"),
        DatasetPageSpec("hiring", "cv"),
        DatasetPageSpec("education-a1", "education/persuade"),
        DatasetPageSpec("education-a2", "education_positioned/persuade"),
    ):
        pages.append(build_dataset_page(spec))

    pages.append(build_response_page("decision-response", _decision_examples))
    pages.append(build_response_page("reasoning-flip", _reasoning_examples))
    for slug in ("cross-influence", "concept-erasure", "validity-checks"):
        pages.append(build_prose_page(slug, GROUP_READOUT))

    index_content = parse_content(CONTENT_ROOT / "index.md")
    index = Page(
        slug="", title=index_content["meta"].get("title", "Experiment review"),
        subtitle=index_content["meta"].get("subtitle", ""),
        sections=[Section(k, k.replace("-", " ").title(), md(v))
                  for k, v in index_content["sections"].items()],
    )

    out: Path = args.out
    (out / "experiments").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)

    for p in pages:
        htmlout = render_page(p, pages=pages, giscus=giscus, repo_url=REPO_URL,
                              generated=generated, depth=1)
        (out / "experiments" / f"{p.slug}.html").write_text(htmlout)

    (out / "index.html").write_text(
        render_index(index, pages=pages, giscus=giscus, repo_url=REPO_URL, generated=generated)
    )
    (out / "assets" / "style.css").write_text(STYLESHEET)
    (out / ".nojekyll").write_text("")

    status = "configured" if giscus.configured else "PLACEHOLDER (giscus.json not filled in)"
    print(f"built {len(pages)} pages + index -> {out}")
    print(f"comments: {status}")
    if not giscus.configured:
        print("  enable Discussions on the site repo, install giscus, then fill site/giscus.json")


if __name__ == "__main__":
    main()
