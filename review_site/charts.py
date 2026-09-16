"""
Inline-SVG charts for the review site.

**Inline SVG, not PNG**, for three reasons: no matplotlib dependency in the build, no binary
blobs in git, and — because the SVG is inlined into the page rather than linked — the marks can
reference CSS custom properties, so light/dark mode comes free from the stylesheet with no
second rendering pass.

Only two forms are needed, and both are single-series magnitude comparisons:

- `draw_distribution` — how often each marker was drawn from its pool, against a dashed
  *expected-if-uniform* reference. The pools are uniform by design, so the chart doubles as a
  sampling sanity check.
- `delta_histogram` — realized |Δchars| across a cell, against the gate threshold. Shows
  whether a 0% discard rate means "well-matched markers" or "loose gate".

Colours are the validated defaults (`references/palette.md`): series blue for the marks,
status-critical for the threshold rule. Both pairs pass every check of
`validate_palette.js` in light and dark. A single series needs no legend — the caption names
it — and every bar is directly value-labelled, with a `<title>` giving a native hover tooltip
and a `<details>` table view beside the figure for the non-visual path.
"""

from __future__ import annotations

import html
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Geometry (viewBox units; the SVG scales to its container width).
_W = 640
_ROW_H = 26
_BAR_H = 14
_PAD_TOP = 8
_PAD_BOTTOM = 26
_LABEL_W = 168
_VALUE_W = 52
_RADIUS = 4


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _bar_h(x0: float, y: float, w: float, h: float, r: float = _RADIUS) -> str:
    """Horizontal bar path: square at the baseline, rounded at the data end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    x1 = x0 + w
    return (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def _bar_v(x: float, y_top: float, w: float, y_base: float, r: float = _RADIUS) -> str:
    """Vertical bar path: square at the baseline, rounded at the data end."""
    h = y_base - y_top
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    return (f"M{x:.1f},{y_base:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} "
            f"V{y_base:.1f} Z")


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]], caption: str) -> str:
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    return (
        f'<details class="table-view"><summary>Table view</summary>'
        f'<table><caption>{_esc(caption)}</caption><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></details>"
    )


def draw_distribution(
    counts: Dict[str, int],
    *,
    caption: str,
    expected_label: str = "expected if uniform",
    show_expected: bool = True,
) -> str:
    """Horizontal bars: how often each value was drawn, against a uniform reference."""
    items: List[Tuple[str, int]] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if not items:
        return ""
    n_bars = len(items)
    total = sum(v for _, v in items)
    expected = total / n_bars if n_bars else 0
    vmax = max(max(v for _, v in items), expected) * 1.08 or 1

    plot_w = _W - _LABEL_W - _VALUE_W
    height = _PAD_TOP + n_bars * _ROW_H + _PAD_BOTTOM

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{_esc(caption)}" preserveAspectRatio="xMinYMin meet">'
    ]

    for i, (name, value) in enumerate(items):
        y = _PAD_TOP + i * _ROW_H
        bar_y = y + (_ROW_H - _BAR_H) / 2
        w = (value / vmax) * plot_w
        share = value / total * 100 if total else 0
        parts.append(
            f'<text class="cat" x="{_LABEL_W - 10}" y="{bar_y + _BAR_H / 2 + 4}" '
            f'text-anchor="end">{_esc(name)}</text>'
        )
        parts.append(
            f'<path class="mark" d="{_bar_h(_LABEL_W, bar_y, w, _BAR_H)}">'
            f"<title>{_esc(name)}: {value} draws ({share:.1f}%)</title></path>"
        )
        parts.append(
            f'<text class="val" x="{_LABEL_W + w + 8}" y="{bar_y + _BAR_H / 2 + 4}">{value}</text>'
        )

    if show_expected and n_bars > 1:
        ex = _LABEL_W + (expected / vmax) * plot_w
        y_end = _PAD_TOP + n_bars * _ROW_H
        parts.append(
            f'<line class="ref" x1="{ex:.1f}" y1="{_PAD_TOP - 2}" x2="{ex:.1f}" y2="{y_end}"/>'
        )
        parts.append(
            f'<text class="ref-label" x="{ex:.1f}" y="{y_end + 16}" text-anchor="middle">'
            f"{_esc(expected_label)} ({expected:.0f})</text>"
        )

    parts.append("</svg>")
    svg = "".join(parts)
    table = _table(
        ["value", "draws", "share"],
        [(k, v, f"{v / total * 100:.1f}%") for k, v in items],
        caption,
    )
    return f'<figure class="chart-figure">{svg}<figcaption>{_esc(caption)}</figcaption>{table}</figure>'


def gate_headroom(rows: Sequence[tuple], *, caption: str) -> str:
    """How close each cell's realized length difference comes to the gate it must pass.

    `rows` are `(label, realized, gate)`. Bar length is realized/gate, because the gate is not
    the same for every axis (the composed intersection clause gets a relaxed bound), so raw
    lengths would not be comparable across rows. The reference line is the gate itself; the
    direct label keeps the raw numbers visible.

    This replaced a per-cell histogram, which was the wrong form: the marker clauses are fixed
    strings, so within a cell the difference is usually a single constant and the "distribution"
    was one bar.
    """
    items = [(str(l), float(r), float(g)) for l, r, g in rows if g]
    if not items:
        return ""
    n = len(items)
    plot_w = _W - _LABEL_W - _VALUE_W
    height = _PAD_TOP + n * _ROW_H + _PAD_BOTTOM
    # Scale so the gate sits at 80% of the plot, leaving the reference line visible on-chart.
    vmax = max(max(r / g for _, r, g in items), 1.0) * 1.25

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{_esc(caption)}" preserveAspectRatio="xMinYMin meet">'
    ]
    for i, (label, realized, gate) in enumerate(items):
        y = _PAD_TOP + i * _ROW_H
        bar_y = y + (_ROW_H - _BAR_H) / 2
        frac = realized / gate
        w = (frac / vmax) * plot_w
        parts.append(
            f'<text class="cat" x="{_LABEL_W - 10}" y="{bar_y + _BAR_H / 2 + 4}" '
            f'text-anchor="end">{_esc(label)}</text>'
        )
        parts.append(
            f'<path class="mark" d="{_bar_h(_LABEL_W, bar_y, w, _BAR_H)}">'
            f"<title>{_esc(label)}: |Δ| {realized:.0f} of a {gate:.0f}-char bound "
            f"({frac * 100:.0f}% of the gate)</title></path>"
        )
        parts.append(
            f'<text class="val" x="{_LABEL_W + max(w, 2) + 8}" y="{bar_y + _BAR_H / 2 + 4}">'
            f"{realized:.0f} / {gate:.0f}</text>"
        )

    gx = _LABEL_W + (1.0 / vmax) * plot_w
    y_end = _PAD_TOP + n * _ROW_H
    parts.append(f'<line class="ref-crit" x1="{gx:.1f}" y1="{_PAD_TOP - 2}" x2="{gx:.1f}" y2="{y_end}"/>')
    parts.append(
        f'<text class="ref-crit-label" x="{gx:.1f}" y="{y_end + 16}" text-anchor="middle">'
        f"gate</text>"
    )
    parts.append("</svg>")
    table = _table(
        ["cell", "max |Δchars|", "gate", "share of gate"],
        [(l, f"{r:.0f}", f"{g:.0f}", f"{r / g * 100:.0f}%") for l, r, g in items],
        caption,
    )
    return f'<figure class="chart-figure">{"".join(parts)}<figcaption>{_esc(caption)}</figcaption>{table}</figure>'


def delta_histogram(
    values: Sequence[int],
    *,
    threshold: Optional[int],
    caption: str,
    unit: str = "chars",
) -> str:
    """Vertical histogram of realized |Δ| per pair, against the gate threshold."""
    if not values:
        return ""
    counts: Dict[int, int] = {}
    for v in values:
        counts[int(v)] = counts.get(int(v), 0) + 1

    # Plot only the observed range. Extending to the gate wasted most of the canvas (a gate of 40
    # against data at 2-5 left ~85% empty and squashed the variation into a corner) and duplicated
    # a question `gate_headroom` already answers for every cell at once. The threshold is stated in
    # the caption instead of drawn.
    buckets = list(range(0, max(counts) + 1))
    cmax = max(counts.values())

    top_pad = 18                                # room for the max-count label
    height = 190
    plot_h = height - top_pad - 40
    left = 44
    plot_w = _W - left - 16
    step = plot_w / max(len(buckets), 1)
    bar_w = max(min(step * 0.7, 48), 2)         # scales with available width, leaves a real gap
    y_base = top_pad + plot_h

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{_esc(caption)}" preserveAspectRatio="xMinYMin meet">'
    ]
    parts.append(f'<line class="axis" x1="{left}" y1="{y_base}" x2="{_W - 16}" y2="{y_base}"/>')
    parts.append(
        f'<text class="cat" x="{left - 8}" y="{top_pad + 4}" text-anchor="end">{cmax}</text>'
    )
    parts.append(f'<text class="cat" x="{left - 8}" y="{y_base + 4}" text-anchor="end">0</text>')

    total = len(values)
    for i, b in enumerate(buckets):
        c = counts.get(b, 0)
        x = left + i * step + (step - bar_w) / 2
        h = (c / cmax) * plot_h if cmax else 0
        if c:
            parts.append(
                f'<path class="mark" d="{_bar_v(x, y_base - h, bar_w, y_base)}">'
                f"<title>Δ = {b} {_esc(unit)}: {c} pairs ({c / total * 100:.1f}%)</title></path>"
            )
            parts.append(
                f'<text class="val" x="{x + bar_w / 2:.1f}" y="{y_base - h - 5:.1f}" '
                f'text-anchor="middle">{c}</text>'
            )
        parts.append(
            f'<text class="cat" x="{x + bar_w / 2:.1f}" y="{y_base + 16}" '
            f'text-anchor="middle">{b}</text>'
        )

    parts.append(
        f'<text class="axis-title" x="{left + plot_w / 2:.1f}" y="{height - 6}" '
        f'text-anchor="middle">|Δ{_esc(unit)}| between the two sides of a pair</text>'
    )
    parts.append("</svg>")
    svg = "".join(parts)
    full = caption
    if threshold is not None:
        full = (f"{caption} Range shown is the observed one; every pair is inside the "
                f"{int(threshold)}-char gate (see the chart above).")
    table = _table(
        [f"|Δ{unit}|", "pairs", "share"],
        [(b, counts.get(b, 0), f"{counts.get(b, 0) / total * 100:.1f}%") for b in buckets],
        caption,
    )
    return f'<figure class="chart-figure">{svg}<figcaption>{_esc(full)}</figcaption>{table}</figure>'
