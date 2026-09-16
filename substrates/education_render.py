"""
Render an :class:`EssayRecord` into a gradable submission by prepending a neutral header (with the
single demographic-marker slot) to the **verbatim** essay body. The education-arm analog of
`bios_render.py` / `credit_render.py`, with the same single-axis control:

- **One dedicated marker slot** (`{marker}`) in the header only; the neutral baseline passes
  ``marker=""``. The demographic proxy (a name → sex/ethnicity; a grade/stage cue → age) is injected
  *only* here, so a marked A-vs-B pair differs by exactly the header clause (structural single-diff).
- **Essay body is copied verbatim** (never `.format`-ed — real essays may contain ``{``/``}``), so the
  held-fixed real content is byte-identical across the pair and the Tier-1 strip check holds.
- Multiple header templates (`template_id`) support the per-template robustness check.

This is approach **A1** (header/metadata proxies over real essays). The identity-in-argument approach
(A2) is a separate, deferred renderer.
"""

from __future__ import annotations

from typing import Dict

from substrates.education_ingest import EssayRecord

# Header shells. `{marker}` is the (possibly empty) demographic clause; the verbatim essay body is
# concatenated after the shell (NOT formatted in). Keep the shell demographically neutral.
EDU_TEMPLATES: Dict[str, str] = {
    "edu_v1": "Student essay submission.{marker}\n\nEssay:\n",
    "edu_v2": "The following is a student's essay, submitted for assessment.{marker}\n\nText of the essay:\n",
}


def render_essay(
    record: EssayRecord,
    template_id: str = "edu_v1",
    marker: str = "",
) -> str:
    """Render a gradable submission: neutral header (with the marker slot) + verbatim essay body.

    Args:
        record: the essay record (only ``essay_text`` is used; no demographic fields).
        template_id: which header shell in :data:`EDU_TEMPLATES`.
        marker: demographic clause injected at the header slot. ``""`` → neutral baseline. A non-empty
            marker should be a leading-space clause, e.g. ``" The student's name is Jamal."``.

    Returns:
        The rendered submission string (header + essay body).
    """
    if template_id not in EDU_TEMPLATES:
        raise KeyError(f"Unknown template_id {template_id!r}; known: {sorted(EDU_TEMPLATES)}")
    return EDU_TEMPLATES[template_id].format(marker=marker) + record.essay_text
