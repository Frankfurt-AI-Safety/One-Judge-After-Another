"""
Page templating for the review site: shell, nav, section layout, comment slot, stylesheet.

Two things here are load-bearing rather than cosmetic.

**The giscus seam.** Comments are configured by `site/giscus.json`. When it carries real ids
the page emits the giscus `<script>`; when it does not, the same slot renders a labelled
placeholder explaining that Discussions is not wired up yet. So the site is complete and
reviewable before anyone has admin rights on the repo, and turning comments on later is two
ids and a rebuild — no code change. `render_comments` is the only place that branches on it.

**Theming.** Colours are CSS custom properties from the validated default palette, declared for
light, `prefers-color-scheme: dark`, and an explicit `data-theme` stamp. Because the charts are
*inlined* SVG rather than linked images, their marks reference the same variables and follow
the theme automatically.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from markdown_it import MarkdownIt

_MD = MarkdownIt("commonmark").enable(["table", "strikethrough"])


def md(text: str) -> str:
    """CommonMark + tables. Linkify is deliberately off (it needs an extra package)."""
    return _MD.render(text or "")


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


@dataclass
class Section:
    id: str
    title: str
    body: str            # HTML
    lead: str = ""       # optional one-line summary under the heading


@dataclass
class Page:
    slug: str            # "" for the index
    title: str
    subtitle: str = ""
    sections: List[Section] = field(default_factory=list)
    group: str = ""
    blurb: str = ""      # shown on the index card


@dataclass
class GiscusConfig:
    repo: str = ""
    repo_id: str = ""
    category: str = ""
    category_id: str = ""
    enabled: bool = False

    @property
    def configured(self) -> bool:
        """Both the ids *and* the explicit switch.

        The switch is separate on purpose. Installing the giscus GitHub App is a web-only
        step, and embedding the widget before it is installed renders a giscus error box —
        strictly worse for a reviewer than our own labelled placeholder. So the ids can be
        filled in ahead of time and the embed turned on in one edit once the app is there.
        """
        return bool(self.enabled and self.repo and self.repo_id and self.category_id)

    @classmethod
    def load(cls, path: Path) -> "GiscusConfig":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(
            repo=raw.get("repo", ""), repo_id=raw.get("repoId", ""),
            category=raw.get("category", ""), category_id=raw.get("categoryId", ""),
            enabled=bool(raw.get("enabled", False)),
        )


def render_comments(cfg: GiscusConfig, *, page_title: str) -> str:
    """The comment slot. Falls back to a labelled placeholder when giscus is not configured."""
    if not cfg.configured:
        return (
            '<section class="comments" id="comments"><h2>Feedback</h2>'
            '<div class="placeholder"><p><strong>Comments are not wired up yet.</strong> '
            "This page is ready for review; the comment box appears here once GitHub "
            "Discussions is enabled on the site repo and the giscus ids are filled into "
            "<code>site/giscus.json</code>. Until then, please send feedback in the usual "
            "channel, quoting the section heading.</p></div></section>"
        )
    return (
        '<section class="comments" id="comments"><h2>Feedback</h2>'
        '<p class="muted">Comments are threaded per page via GitHub Discussions. '
        "Sign in with GitHub to post.</p>"
        '<script src="https://giscus.app/client.js"'
        f' data-repo="{esc(cfg.repo)}"'
        f' data-repo-id="{esc(cfg.repo_id)}"'
        f' data-category="{esc(cfg.category)}"'
        f' data-category-id="{esc(cfg.category_id)}"'
        ' data-mapping="title" data-strict="1"'
        f' data-term="{esc(page_title)}"'
        ' data-reactions-enabled="1" data-emit-metadata="0" data-input-position="top"'
        ' data-theme="preferred_color_scheme" data-lang="en" crossorigin="anonymous" async>'
        "</script></section>"
    )


def _nav(pages: Sequence[Page], current: str, depth: int) -> str:
    up = "../" * depth
    groups: Dict[str, List[Page]] = {}
    for p in pages:
        if p.slug:
            groups.setdefault(p.group, []).append(p)
    out = [f'<a class="brand" href="{up}index.html">Experiment review</a>']
    for group, items in groups.items():
        out.append(f'<div class="nav-group"><span class="nav-group-title">{esc(group)}</span><ul>')
        for p in items:
            cls = ' class="current"' if p.slug == current else ""
            out.append(f'<li><a href="{up}experiments/{p.slug}.html"{cls}>{esc(p.title)}</a></li>')
        out.append("</ul></div>")
    return "".join(out)


def _toc(sections: Sequence[Section]) -> str:
    if not sections:
        return ""
    items = "".join(f'<li><a href="#{esc(s.id)}">{esc(s.title)}</a></li>' for s in sections)
    return f'<nav class="toc" aria-label="On this page"><span>On this page</span><ul>{items}<li><a href="#comments">Feedback</a></li></ul></nav>'


def render_page(
    page: Page,
    *,
    pages: Sequence[Page],
    giscus: GiscusConfig,
    repo_url: str,
    generated: str,
    depth: int = 1,
) -> str:
    up = "../" * depth
    body = []
    for s in page.sections:
        lead = f'<p class="lead">{s.lead}</p>' if s.lead else ""
        body.append(
            f'<section id="{esc(s.id)}"><h2>{esc(s.title)}</h2>{lead}{s.body}</section>'
        )
    subtitle = f'<p class="subtitle">{esc(page.subtitle)}</p>' if page.subtitle else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(page.title)} — Experiment review</title>
<meta name="description" content="{esc(page.subtitle or page.title)}">
<link rel="stylesheet" href="{up}assets/style.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="layout">
<aside class="sidebar">{_nav(pages, page.slug, depth)}</aside>
<main id="main">
<header class="page-head"><h1>{esc(page.title)}</h1>{subtitle}</header>
{_toc(page.sections)}
{"".join(body)}
{render_comments(giscus, page_title=page.title)}
<footer class="page-foot">
<p>Generated from the pipeline artifacts on {esc(generated)} —
do not edit these pages by hand; edit <code>site/content/</code> in the
<a href="{esc(repo_url)}">research repo</a> and rebuild.</p>
</footer>
</main>
</div>
</body>
</html>
"""


def render_index(
    index: Page,
    *,
    pages: Sequence[Page],
    giscus: GiscusConfig,
    repo_url: str,
    generated: str,
) -> str:
    groups: Dict[str, List[Page]] = {}
    for p in pages:
        if p.slug:
            groups.setdefault(p.group, []).append(p)
    cards = []
    for group, items in groups.items():
        cards.append(f'<h2 class="group">{esc(group)}</h2><div class="cards">')
        for p in items:
            cards.append(
                f'<a class="card" href="experiments/{p.slug}.html">'
                f"<h3>{esc(p.title)}</h3><p>{esc(p.blurb or p.subtitle)}</p></a>"
            )
        cards.append("</div>")
    intro = "".join(
        f'<section id="{esc(s.id)}"><h2>{esc(s.title)}</h2>{s.body}</section>'
        for s in index.sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(index.title)}</title>
<meta name="description" content="{esc(index.subtitle)}">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="layout">
<aside class="sidebar">{_nav(pages, "", 0)}</aside>
<main id="main">
<header class="page-head"><h1>{esc(index.title)}</h1>
<p class="subtitle">{esc(index.subtitle)}</p></header>
{intro}
{"".join(cards)}
<footer class="page-foot"><p>Generated on {esc(generated)} from
<a href="{esc(repo_url)}">the research repo</a>.</p></footer>
</main>
</div>
</body>
</html>
"""


STYLESHEET = """/* Review site — generated; edit review_site/render.py, not this file.
   Palette: the validated data-viz defaults (references/palette.md). Chart marks reference
   these same custom properties, so inline SVG follows the theme with no second render. */
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --series-1: #2a78d6;
  --critical: #d03b3b;
  --border: rgba(11,11,11,0.10);
  --mark-bg: #fdf2c9;
  --code-bg: #f0efec;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --series-1: #3987e5; --critical: #d03b3b; --border: rgba(255,255,255,0.10);
    --mark-bg: #4a3f12; --code-bg: #262623;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --series-1: #3987e5; --critical: #d03b3b; --border: rgba(255,255,255,0.10);
  --mark-bg: #4a3f12; --code-bg: #262623;
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--text-primary);
  font: 16px/1.65 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.skip { position: absolute; left: -9999px; }
.skip:focus { left: 8px; top: 8px; background: var(--surface-1); padding: 8px; z-index: 10; }

.layout { display: flex; align-items: flex-start; gap: 32px; max-width: 1200px; margin: 0 auto; padding: 24px; }
.sidebar { position: sticky; top: 24px; flex: 0 0 220px; font-size: 14px; }
.sidebar .brand { display: block; font-weight: 650; margin-bottom: 16px; color: var(--text-primary); text-decoration: none; }
.nav-group { margin-bottom: 14px; }
.nav-group-title { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin-bottom: 4px; }
.sidebar ul { list-style: none; margin: 0; padding: 0; }
.sidebar li { margin: 2px 0; }
.sidebar a { color: var(--text-secondary); text-decoration: none; }
.sidebar a:hover { color: var(--text-primary); }
.sidebar a.current { color: var(--series-1); font-weight: 600; }

main { flex: 1 1 auto; min-width: 0; background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 28px 32px; }
.page-head h1 { margin: 0 0 4px; font-size: 28px; line-height: 1.25; }
.subtitle { margin: 0 0 8px; color: var(--text-secondary); }
h2 { font-size: 20px; margin: 32px 0 8px; padding-top: 8px; border-top: 1px solid var(--grid); }
h3 { font-size: 16px; margin: 20px 0 6px; }
p { margin: 0 0 12px; }
.lead { color: var(--text-secondary); margin-top: 0; }
.muted { color: var(--muted); font-size: 14px; }
a { color: var(--series-1); }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 4px; font-size: 13px; }
pre { background: var(--code-bg); padding: 12px 14px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; }

.toc { font-size: 14px; border: 1px solid var(--grid); border-radius: 8px; padding: 10px 14px; margin: 16px 0 0; }
.toc span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.toc ul { list-style: none; margin: 6px 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 4px 16px; }

table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 8px 0 16px; display: block; overflow-x: auto; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--text-secondary); font-weight: 600; }
td { font-variant-numeric: tabular-nums; }

/* Contrastive pairs, side by side */
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0 18px; }
.pair > div { border: 1px solid var(--grid); border-radius: 8px; padding: 12px 14px; background: var(--page); }
.pair .pole { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin-bottom: 6px; }
.rendered { font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
mark.clause { background: var(--mark-bg); color: var(--text-primary); padding: 1px 2px; border-radius: 3px; font-weight: 600; }
.elided { color: var(--muted); }
.provenance { font-size: 12px; color: var(--muted); margin-top: 8px; }
.meta-row { font-size: 13px; color: var(--text-secondary); margin: 0 0 8px; }
.meta-row span { display: inline-block; margin-right: 14px; }
@media (max-width: 760px) { .pair { grid-template-columns: 1fr; } .layout { flex-direction: column; } .sidebar { position: static; flex: 1 1 auto; } }

/* Charts (inline SVG uses these variables directly) */
.chart-figure { margin: 12px 0 20px; }
.chart { display: block; max-width: 100%; height: auto; }
.chart .mark { fill: var(--series-1); }
.chart .cat, .chart .val, .chart .axis-title { fill: var(--muted); font-size: 11px; font-family: inherit; }
.chart .val { fill: var(--text-secondary); font-variant-numeric: tabular-nums; }
.chart .axis { stroke: var(--axis); stroke-width: 1; }
.chart .ref { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 4 3; }
.chart .ref-label { fill: var(--muted); font-size: 11px; font-family: inherit; }
.chart .ref-crit { stroke: var(--critical); stroke-width: 2; stroke-dasharray: 4 3; }
.chart .ref-crit-label { fill: var(--critical); font-size: 11px; font-weight: 600; font-family: inherit; }
figcaption { font-size: 13px; color: var(--text-secondary); margin-top: 4px; }
.table-view { margin-top: 6px; font-size: 13px; }
.table-view summary { cursor: pointer; color: var(--muted); }

.callout { border-left: 3px solid var(--series-1); background: var(--page); padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 12px 0; }
.callout.warn { border-left-color: var(--critical); }
.callout p:last-child { margin-bottom: 0; }

.group { border-top: none; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin-bottom: 20px; }
.card { display: block; border: 1px solid var(--grid); border-radius: 8px; padding: 14px 16px; text-decoration: none; color: inherit; background: var(--page); }
.card:hover { border-color: var(--series-1); }
.card h3 { margin: 0 0 4px; color: var(--series-1); }
.card p { margin: 0; font-size: 14px; color: var(--text-secondary); }

.comments { margin-top: 32px; }
.placeholder { border: 1px dashed var(--axis); border-radius: 8px; padding: 14px 16px; color: var(--text-secondary); font-size: 14px; }
.page-foot { margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--grid); font-size: 13px; color: var(--muted); }
"""
