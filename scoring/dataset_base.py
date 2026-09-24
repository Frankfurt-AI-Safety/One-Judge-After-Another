"""
Base classes for bias evaluation datasets.

Each dataset provides separate probe (train) and test splits.
The probe split (500 examples by default) is used for building the bias direction.
The test split is used for evaluation.

A dataset whose examples come in groups (several matched pairs cut from one record) splits whole
groups. Its probe split is then best sized in groups: ``probe_records`` takes that many groups,
stratified by ``_get_stratum_key`` (the record's quality label), where ``probe_size`` would count
examples and so fix the number of records only through the pairs each record happens to contribute.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple, Type


@dataclass
class DatasetExample:
    """Base class for dataset examples."""
    
    id: str
    """Unique identifier for the example."""


@dataclass
class ContrastivePair:
    """A pair of texts for contrastive probe learning.
    
    Used to compute probe direction as: mean(positive) - mean(negative)
    """
    
    positive_text: str
    """Text representing the positive direction (e.g., uncertain, long, sycophantic)."""
    
    negative_text: str
    """Text representing the negative direction (e.g., direct, short, independent)."""
    
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
    """Abstract base class for probe datasets.
    
    Each dataset provides:
    - probe_train: Examples for building the probe direction (default 500)
    - probe_test: Examples for evaluation
    
    Uses deterministic hash-based splitting for reproducibility.
    """
    
    PROBE_SIZE: int = 500
    """Default number of examples for probe training."""
    
    def __init__(
        self,
        source: str,
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
        probe_records: Optional[int] = None,
    ):
        """Initialize dataset.
        
        Args:
            source: Path to data file or HuggingFace dataset ID
            probe_size: Number of examples for probe training
            split_seed: Seed for deterministic splitting
            max_test_examples: Cap on test examples (None = use all)
            probe_records: Number of GROUPS (records) for probe training, stratified by
                ``_get_stratum_key``; replaces ``probe_size`` when set. Needs a grouped split.
        """
        self.source = source
        self.probe_size = probe_size
        self.split_seed = split_seed
        self.max_test_examples = max_test_examples
        self.probe_records = probe_records
        
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
    
    def _ensure_loaded(self) -> None:
        """Ensure data is loaded and splits are computed."""
        if self._raw_data is None:
            self._raw_data = self._load_raw_data()
            self._compute_splits()
    
    def _compute_splits(self) -> None:
        """Compute deterministic probe/test split using hash-based assignment.
        
        Ensures at least 20% of examples are reserved for testing.
        If probe_size exceeds 80% of available data, it's reduced accordingly.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        n_total = len(self._raw_data)
        grouped = bool(self._raw_data) and self._get_group_key(self._raw_data[0]) is not None

        if self.probe_records is not None:
            if not grouped:
                raise ValueError(f"{self.name}: probe_records counts records and needs a split grouped "
                                 f"by record; this dataset splits single examples (use probe_size)")
            probe_indices, test_indices = self._record_split(self.probe_records)
        else:
            # Reserve at least 20% for testing, or 50 examples, whichever is smaller
            min_test_size = min(max(n_total // 5, 1), 50)
            max_probe_size = max(n_total - min_test_size, 1)

            actual_probe_size = min(self.probe_size, max_probe_size)

            if actual_probe_size < self.probe_size:
                logger.warning(
                    "Requested probe_size=%d but only %d total examples. "
                    "Using %d for probe, %d for test.",
                    self.probe_size, n_total, actual_probe_size, n_total - actual_probe_size
                )

            # Use hash-based assignment for deterministic split
            all_indices = list(range(n_total))

            # Sort by hash to get deterministic ordering
            def hash_key(idx: int) -> float:
                example = self._raw_data[idx]
                key = f"{self.split_seed}|{idx}|{self._get_example_key(example)}".encode("utf-8")
                h = hashlib.sha256(key).digest()
                return int.from_bytes(h[:8], "big") / 2**64

            if not grouped:
                sorted_indices = sorted(all_indices, key=hash_key)

                # Split: first actual_probe_size go to probe, rest to test
                probe_indices = sorted_indices[:actual_probe_size]
                test_indices = sorted_indices[actual_probe_size:]
            else:
                logger.warning(
                    "%s: probe_size=%d counts PAIRS, so the number of probe records depends on how many "
                    "pairs each record contributes; set probe_records to count records.",
                    self.name, self.probe_size)
                probe_indices, test_indices = self._grouped_split(actual_probe_size, min_test_size)
        
        # Apply max_test_examples cap
        if self.max_test_examples is not None:
            test_indices = test_indices[:self.max_test_examples]
        
        self._probe_indices = probe_indices
        self._test_indices = test_indices
        
        logger.info("Split: %d probe, %d test (from %d total)", 
                   len(probe_indices), len(test_indices), n_total)
    
    def _get_example_key(self, example: Any) -> str:
        """Get unique key for example. Override for custom hashing."""
        return str(example)

    def _get_group_key(self, example: Any) -> Optional[str]:
        """Examples sharing a group key always land in the same split. ``None`` (the default) means
        no grouping: every example is split on its own."""
        return None

    def _get_stratum_key(self, example: Any) -> Optional[Hashable]:
        """The stratum a group belongs to, read from its first example, for the ``probe_records``
        split. ``None`` (the default) puts every group in one stratum: unstratified."""
        return None

    def _group_hash(self, key: str) -> bytes:
        return hashlib.sha256(f"{self.split_seed}|{key}".encode("utf-8")).digest()

    def _groups(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for idx, example in enumerate(self._raw_data):
            groups.setdefault(self._get_group_key(example), []).append(idx)
        return groups

    def _grouped_split(self, probe_size: int, min_test_size: int) -> tuple:
        """Whole groups, in hashed order, fill the probe split until it holds at least ``probe_size``
        examples; the remaining groups form the test split (at least ``min_test_size`` examples if
        the data allows). The test split is ordered round-robin across its groups, so a
        ``max_test_examples`` cap keeps as many distinct groups as possible.

        Each test group's members are **rotated by the group's position** before the round-robin:
        group *k* starts at member ``k mod len(group)``. The generators write every record's pairs in
        the same order (first template, pole-A cell first), so without the rotation the first round
        was always that one member, and a cap no larger than the number of test groups — every config's
        ``max_test_examples: 200`` — kept a single cell of the other factors and a single template
        (audit 2026-09-23: credit, hiring and education alike). Every headline number was then a
        conditional effect at the hypothesised worst-case corner, not the factorial marginal, and the
        per-template check on eval had one template. Rotating spreads the first round evenly over all
        member positions (exactly, when groups are equally sized and the cap is a multiple of the
        group size) and is deterministic; the uncapped test split contains the same examples as
        before, in a different order."""
        groups = self._groups()
        ordered = sorted(groups, key=self._group_hash)
        n_total = len(self._raw_data)
        probe: List[int] = []
        test_groups: List[List[int]] = []
        for key in ordered:
            members = groups[key]
            room = n_total - len(probe) - len(members) >= min_test_size
            if len(probe) < probe_size and room:
                probe.extend(members)
            else:
                test_groups.append(members)
        return probe, self._interleave(test_groups)

    def _record_split(self, n_records: int) -> tuple:
        """``n_records`` whole groups, in hashed order, form the probe split, **stratified** by
        ``_get_stratum_key``: each stratum gets its share of ``n_records`` (largest remainder), taken
        from its first groups in hashed order. The remaining groups form the test split, interleaved as
        in ``_grouped_split``.

        Counting records rather than pairs matters once a record contributes several pairs: the
        factorial gives 8 per single axis but 2 for the intersection, so ``probe_size: 300`` meant
        ~38 records for every single axis and 150 for the intersection (audit 2026-09-23, item 4.1).
        The hash order ignores the axis, so every axis of a manifest gets the same probe records. With
        two strata the probe sets are nested in ``n_records`` (a larger probe adds records, never swaps
        them), so a probe-size curve compares like with like. At least ``min(max(groups // 5, 1), 50)``
        groups stay in the test split."""
        import logging
        groups = self._groups()
        ordered = sorted(groups, key=self._group_hash)
        min_test = min(max(len(ordered) // 5, 1), 50)
        n = min(n_records, max(len(ordered) - min_test, 1))
        if n < n_records:
            logging.getLogger(__name__).warning(
                "%s: requested probe_records=%d but only %d records; using %d for probe, %d for test.",
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
        """Rotate each group by its position, then round-robin across groups (see ``_grouped_split``)."""
        test_groups = [g[k % len(g):] + g[:k % len(g)] for k, g in enumerate(test_groups)]
        test: List[int] = []
        for rank in range(max((len(g) for g in test_groups), default=0)):
            test.extend(g[rank] for g in test_groups if rank < len(g))
        return test

    def split_report(self) -> Dict[str, Any]:
        """How the split came out: its mode, examples and (for a grouped split) records per side, and
        in ``records`` mode the records per stratum on each side (after any ``max_test_examples`` cap)."""
        self._ensure_loaded()
        grouped = bool(self._raw_data) and self._get_group_key(self._raw_data[0]) is not None
        report: Dict[str, Any] = {"mode": "records" if self.probe_records is not None else "pairs",
                                  "probe_pairs": len(self._probe_indices),
                                  "test_pairs": len(self._test_indices)}
        if not grouped:
            return report
        for side, indices in (("probe", self._probe_indices), ("test", self._test_indices)):
            first: Dict[str, int] = {}
            for i in indices:
                first.setdefault(self._get_group_key(self._raw_data[i]), i)
            report[f"{side}_records"] = len(first)
            if self.probe_records is not None:
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
        """Actual number of probe examples after loading."""
        self._ensure_loaded()
        return len(self._probe_indices)
    
    @property
    def test_size_actual(self) -> int:
        """Actual number of test examples after loading."""
        self._ensure_loaded()
        return len(self._test_indices)


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


class DatasetRegistry:
    """Registry for dataset classes."""
    
    _datasets: Dict[str, Type[ProbeDataset]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class."""
        def decorator(dataset_cls: Type[ProbeDataset]):
            cls._datasets[name] = dataset_cls
            return dataset_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[ProbeDataset]:
        """Get dataset class by name."""
        if name not in cls._datasets:
            raise KeyError(f"Unknown dataset: {name}. Available: {list(cls._datasets.keys())}")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return list(cls._datasets.keys())


def uses_pair_format(tokenizer: Any) -> bool:
    """Check if tokenizer should use pair format (question, answer) instead of chat template.
    
    Pair format is used for models like DeBERTa that expect tokenizer(text_a, text_b).
    """
    # No chat template means we should use pair format for proper sentence pair encoding
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return True
    return False


def format_conversation(tokenizer: Any, prompt: str, response: str, force_pair: bool = False):
    """Format prompt/response as chat conversation or pair.
    
    Uses tokenizer's chat template if available, otherwise returns tuple for pair encoding.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt: User prompt
        response: Assistant response
        force_pair: If True, always return tuple for pair encoding
        
    Returns:
        Either formatted string (for chat models) or tuple (prompt, response) for pair models
    """
    if force_pair or uses_pair_format(tokenizer):
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

