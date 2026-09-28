"""
Base classes for the matched-pair datasets, and how a prompt/response pair becomes the text a reward model scores.

A dataset's examples come in groups: several matched pairs cut from one record. It splits whole records into a
probe split (for building a direction) and a test split (for evaluation), so no record straddles the two. The probe
split is sized in records, ``probe_records``, stratified by ``_get_stratum_key`` (the record's quality label). It
used to be sizable in pairs (``probe_size``) too, which fixed the number of records only through the pairs each
record contributes (8 per single axis, 2 for the intersection: 300 pairs were ~38 records per single axis and 150
for the intersection); that mode was removed on 2026-09-28.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContrastivePair:
    """A pair of texts for building a direction: ``mean(positive) − mean(negative)`` over the probe pairs.

    In the matched-pair datasets ``positive`` is side A, the axis's pole A (the level hypothesised to be
    penalised, e.g. female), and ``negative`` is side B; a direction therefore points from pole B to pole A.
    """

    positive_text: str
    """Side A's text (pole A)."""

    negative_text: str
    """Side B's text (pole B)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata about the pair."""


@dataclass
class EvalExample:
    """An example for evaluation with associated metrics."""

    texts: Dict[str, str]
    """Dictionary mapping variant names to formatted texts."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata about the example (e.g., correct answer, question text)."""


class ProbeDataset(ABC):
    """A dataset split by record: ``probe_records`` whole records, stratified by quality, form the probe split,
    the remaining records the test split (optionally capped at ``max_test_examples`` pairs). The split is a
    deterministic function of ``split_seed`` and the record ids."""

    def __init__(
        self,
        source: str,
        probe_records: Optional[int] = None,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            source: Path to data file
            probe_records: Number of records for the probe split, stratified by ``_get_stratum_key``
            split_seed: Seed for deterministic splitting
            max_test_examples: Cap on test examples, in pairs (None = use all)
        """
        if probe_records is None or int(probe_records) < 1:
            raise ValueError(f"probe_records must be a positive number of records, got {probe_records!r}; set "
                             f"probe_records in the config (the probe split is counted in records)")
        self.source = source
        self.probe_records = int(probe_records)
        self.split_seed = split_seed
        self.max_test_examples = max_test_examples

        self._raw_data: Optional[List[Any]] = None
        self._probe_indices: Optional[List[int]] = None
        self._test_indices: Optional[List[int]] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name for logging and identification."""
        pass

    @abstractmethod
    def _load_raw_data(self) -> List[Any]:
        """Load raw data from source. Override in subclass."""
        pass

    @abstractmethod
    def _make_contrastive_pair(self, raw_example: Any, tokenizer: Any) -> Optional[ContrastivePair]:
        """Convert raw example to contrastive pair for probe building.

        Args:
            raw_example: Raw data from _load_raw_data()
            tokenizer: Tokenizer for formatting conversations

        Returns:
            ContrastivePair or None if example should be skipped
        """
        pass

    @abstractmethod
    def _make_eval_example(self, raw_example: Any, tokenizer: Any) -> Optional[EvalExample]:
        """Convert raw example to evaluation example.

        Args:
            raw_example: Raw data from _load_raw_data()
            tokenizer: Tokenizer for formatting conversations

        Returns:
            EvalExample or None if example should be skipped
        """
        pass

    @abstractmethod
    def _get_group_key(self, example: Any) -> str:
        """The record an example was cut from: examples sharing it always land in the same split."""
        pass

    def _ensure_loaded(self) -> None:
        """Ensure data is loaded and splits are computed."""
        if self._raw_data is None:
            self._raw_data = self._load_raw_data()
            self._compute_splits()

    def _compute_splits(self) -> None:
        """The record split (`_record_split`), then the ``max_test_examples`` cap on the test split."""
        probe_indices, test_indices = self._record_split(self.probe_records)
        if self.max_test_examples is not None:
            test_indices = test_indices[:self.max_test_examples]
        self._probe_indices = probe_indices
        self._test_indices = test_indices
        logger.info("Split: %d probe, %d test (from %d total)",
                    len(probe_indices), len(test_indices), len(self._raw_data))

    def _get_stratum_key(self, example: Any) -> Optional[Hashable]:
        """The stratum a record belongs to, read from its first example. ``None`` (the default) puts every
        record in one stratum: unstratified."""
        return None

    def _group_hash(self, key: str) -> bytes:
        return hashlib.sha256(f"{self.split_seed}|{key}".encode("utf-8")).digest()

    def _groups(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for idx, example in enumerate(self._raw_data):
            groups.setdefault(self._get_group_key(example), []).append(idx)
        return groups

    def _record_split(self, n_records: int) -> tuple:
        """``n_records`` whole records, in hashed order, form the probe split, **stratified** by
        ``_get_stratum_key``: each stratum gets its share of ``n_records`` (largest remainder), taken from its
        first records in hashed order. The remaining records form the test split, interleaved
        (`_interleave`).

        The hash order ignores the axis, so the axes of a manifest get the same probe records wherever they
        cover the same records (every axis and encoding of the credit and hiring manifests of 2026-09-24:
        checked 2026-09-28); the cross-marker runner, which excludes the probe records from its evaluation,
        relies on that. With two strata the probe sets are nested in ``n_records`` (a larger probe adds
        records, never swaps them), so a probe-size curve compares like with like. At least
        ``min(max(records // 5, 1), 50)`` records stay in the test split."""
        groups = self._groups()
        ordered = sorted(groups, key=self._group_hash)
        min_test = min(max(len(ordered) // 5, 1), 50)
        n = min(n_records, max(len(ordered) - min_test, 1))
        if n < n_records:
            logger.warning("%s: requested probe_records=%d but only %d records; using %d for probe, %d for test.",
                           self.name, n_records, len(ordered), n, len(ordered) - n)
        strata: Dict[Any, List[str]] = {}
        for key in ordered:
            strata.setdefault(self._get_stratum_key(self._raw_data[groups[key][0]]), []).append(key)
        quota = _largest_remainder(n, {s: len(keys) for s, keys in strata.items()})
        chosen = {key for s, keys in strata.items() for key in keys[:quota[s]]}
        probe = [i for key in ordered if key in chosen for i in groups[key]]
        return probe, self._interleave([groups[key] for key in ordered if key not in chosen])

    @staticmethod
    def _interleave(test_groups: List[List[int]]) -> List[int]:
        """The test split's order: each record's pairs are **rotated by the record's position** (record *k*
        starts at member ``k mod len(record)``), then taken round-robin across records, so a
        ``max_test_examples`` cap keeps as many distinct records as possible.

        The generators write every record's pairs in the same order (first template, pole-A cell first), so
        without the rotation the first round was always that one member, and a cap no larger than the number
        of test records (every config's ``max_test_examples: 200``) kept a single cell of the other factors
        and a single template (audit 2026-09-23). Every headline number was then a conditional effect at the
        hypothesised worst-case corner, not the factorial marginal. Rotating spreads the first round evenly
        over all member positions (exactly, when records are equally sized and the cap is a multiple of the
        record size: credit sex, cap 200, gives 25 pairs in each of the 2 templates × 4 cells, checked
        2026-09-28) and is deterministic."""
        test_groups = [g[k % len(g):] + g[:k % len(g)] for k, g in enumerate(test_groups)]
        test: List[int] = []
        for rank in range(max((len(g) for g in test_groups), default=0)):
            test.extend(g[rank] for g in test_groups if rank < len(g))
        return test

    def split_report(self) -> Dict[str, Any]:
        """How the split came out: pairs and records per side, and records per stratum on each side (after any
        ``max_test_examples`` cap)."""
        self._ensure_loaded()
        report: Dict[str, Any] = {"probe_pairs": len(self._probe_indices),
                                  "test_pairs": len(self._test_indices)}
        for side, indices in (("probe", self._probe_indices), ("test", self._test_indices)):
            first: Dict[str, int] = {}
            for i in indices:
                first.setdefault(self._get_group_key(self._raw_data[i]), i)
            report[f"{side}_records"] = len(first)
            strata: Dict[str, int] = {}
            for i in first.values():
                s = str(self._get_stratum_key(self._raw_data[i]))
                strata[s] = strata.get(s, 0) + 1
            report[f"{side}_strata"] = dict(sorted(strata.items()))
        return report

    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Get contrastive pairs for probe training.

        Args:
            tokenizer: Tokenizer for formatting conversations

        Returns:
            List of ContrastivePair objects
        """
        self._ensure_loaded()
        pairs = []
        for idx in self._probe_indices:
            pair = self._make_contrastive_pair(self._raw_data[idx], tokenizer)
            if pair is not None:
                pairs.append(pair)
        return pairs

    def get_eval_examples(self, tokenizer: Any) -> List[EvalExample]:
        """Get examples for evaluation.

        Args:
            tokenizer: Tokenizer for formatting conversations

        Returns:
            List of EvalExample objects
        """
        self._ensure_loaded()
        examples = []
        for idx in self._test_indices:
            example = self._make_eval_example(self._raw_data[idx], tokenizer)
            if example is not None:
                examples.append(example)
        return examples

    @property
    def probe_size_actual(self) -> int:
        """Actual number of probe examples (pairs) after loading."""
        self._ensure_loaded()
        return len(self._probe_indices)


def _largest_remainder(n: int, sizes: Dict[Any, int]) -> Dict[Any, int]:
    """Split ``n`` across strata in proportion to ``sizes``: each gets the floor of its exact share, and
    the units left go to the largest remainders (ties by the stratum's name, so the result is
    deterministic). No stratum gets more than its size while ``n <= sum(sizes)``."""
    total = sum(sizes.values())
    exact = {s: n * c / total for s, c in sizes.items()}
    quota = {s: int(e) for s, e in exact.items()}
    left = n - sum(quota.values())
    for s in sorted(exact, key=lambda s: (-(exact[s] - quota[s]), str(s)))[:left]:
        quota[s] += 1
    return quota


def uses_pair_format(tokenizer: Any) -> bool:
    """Check if tokenizer should use pair format (question, answer) instead of chat template.

    Pair format is used for models like DeBERTa that expect tokenizer(text_a, text_b).
    """
    # No chat template means we should use pair format for proper sentence pair encoding
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return True
    return False


def format_conversation(tokenizer: Any, prompt: str, response: str):
    """Format prompt/response as chat conversation or pair.

    Uses tokenizer's chat template if available, otherwise returns tuple for pair encoding.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: User prompt
        response: Assistant response

    Returns:
        Either formatted string (for chat models) or tuple (prompt, response) for pair models
    """
    if uses_pair_format(tokenizer):
        # Return tuple for pair encoding: tokenizer(prompt, response)
        return (prompt, response)

    # Chat template path
    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    formatted = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    # Remove BOS token - it will be added back during tokenization
    if tokenizer.bos_token is not None and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token):]
    return formatted


# The pipeline tokenizes the formatted conversation with the tokenizer's defaults, which adds its special tokens
# (e.g. BOS; `format_conversation` strips a BOS the template already has, so it is not doubled). That is how the
# Skywork model cards and RewardBench score (``apply_chat_template(tokenize=False)``, then the tokenizer), and it is
# kept by default. A model whose authors score the template's own token ids (``apply_chat_template(tokenize=True)``,
# no special tokens on top) is switched with `use_template_tokens`; `scoring.backend.TEMPLATE_TOKENIZED` lists them.
ADD_SPECIAL_TOKENS_ATTR = "_onejudge_add_special_tokens"
# One short conversation and one with the multi-line content the arms score (blank lines, a list): templates
# differ in how they treat whitespace, and a one-line sample would not show it.
_SAMPLE_CONVERSATIONS = (
    ("Should this loan be approved?", "Approve."),
    ("Should this loan be approved?\n\nApplicant: a 30-year-old married woman.\nAmount: EUR 4,000 over 24 months.",
     "Decline.\n\n- The income does not cover the instalments.\n- There is no guarantor."),
)


def add_special_tokens(tokenizer: Any) -> bool:
    """Whether the pipeline lets this tokenizer add its special tokens (default True; see `use_template_tokens`)."""
    return getattr(tokenizer, ADD_SPECIAL_TOKENS_ATTR, True)


def _template_ids(tokenizer: Any, prompt: str, response: str) -> List[int]:
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    ids = tokenizer.apply_chat_template(conv, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "keys"):                      # a BatchEncoding (newer transformers)
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def _relation(tokenizer: Any, prompt: str, response: str) -> str:
    template = _template_ids(tokenizer, prompt, response)
    pipeline = list(tokenizer(format_conversation(tokenizer, prompt, response),
                              add_special_tokens=add_special_tokens(tokenizer))["input_ids"])
    if pipeline == template:
        return "aligned"
    bos = getattr(tokenizer, "bos_token_id", None)
    return "extra_bos" if (bos is not None and pipeline == [bos] + template) else "mismatch"


def tokenization_vs_template(tokenizer: Any) -> str:
    """How the pipeline's token ids relate to the chat template's own (``apply_chat_template(tokenize=True)``) on
    the sample conversations: ``"aligned"``, ``"extra_bos"`` (the tokenizer adds a BOS the template lacks),
    ``"mismatch"`` (any other difference, or the samples disagree) or ``"pair_format"`` (no chat template)."""
    if uses_pair_format(tokenizer):
        return "pair_format"
    relations = {_relation(tokenizer, prompt, response) for prompt, response in _SAMPLE_CONVERSATIONS}
    return relations.pop() if len(relations) == 1 else "mismatch"


def use_template_tokens(tokenizer: Any) -> None:
    """Score this tokenizer's conversations as the chat template's own token ids: no special tokens added on
    top. Raises if the formatted text, tokenized that way, still differs from the template's ids."""
    setattr(tokenizer, ADD_SPECIAL_TOKENS_ATTR, False)
    if tokenization_vs_template(tokenizer) != "aligned":
        setattr(tokenizer, ADD_SPECIAL_TOKENS_ATTR, True)
        raise ValueError("tokenizing the formatted conversation without special tokens does not reproduce the "
                         "chat template's token ids")
