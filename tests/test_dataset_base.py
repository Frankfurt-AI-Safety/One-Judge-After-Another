"""`scoring/dataset_base.py`: the tokenization check against the chat template."""

from __future__ import annotations

import pytest

from scoring.dataset_base import tokenization_vs_template, use_template_tokens


class _BlankLineTemplate:
    """A chat tokenizer whose template ids and plain tokenization agree on one-line content but not on a blank
    line (the template collapses it): the kind of difference a one-line sample conversation cannot show."""

    chat_template = "stub"
    bos_token = None
    bos_token_id = None

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False):
        text = "|".join(m["content"] for m in conv)
        return [ord(c) for c in text.replace("\n\n", "\n")] if tokenize else text

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def test_a_multi_line_difference_is_a_mismatch():
    tok = _BlankLineTemplate()
    assert tokenization_vs_template(tok) == "mismatch"
    with pytest.raises(ValueError, match="template"):
        use_template_tokens(tok)
    assert getattr(tok, "_onejudge_add_special_tokens") is True         # left as it was
