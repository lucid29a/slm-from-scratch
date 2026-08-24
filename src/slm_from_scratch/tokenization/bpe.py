"""Byte-Pair Encoding, implemented by hand: learns merges and applies them.

This is the textbook algorithm (Sennrich et al. 2016, as popularized by GPT-2):
start from a corpus of words split into symbols (characters, or byte-characters for
:class:`~slm_from_scratch.tokenization.pretokenizer.ByteLevelPreTokenizer`), and
repeatedly merge the most frequent adjacent symbol pair into a new symbol, until the
vocabulary reaches its target size. The *encoder* then re-applies those exact merges,
in the order they were learned, to new text.

The trainer maintains pair counts incrementally rather than rescanning the whole
corpus on every merge: it keeps an index from each pair to the words containing it,
so a merge only touches the words it actually changes.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from slm_from_scratch.tokenization.base import (
    SPECIAL_TOKENS,
    TOKENIZERS,
    Tokenizer,
    TokenizerConfig,
)
from slm_from_scratch.tokenization.pretokenizer import PRETOKENIZERS, PreTokenizer

__all__ = ["BPETokenizer", "BPETokenizerConfig", "BPETrainer", "Merge"]

Merge = tuple[str, str]


@dataclass(frozen=True, kw_only=True)
class BPETokenizerConfig(TokenizerConfig):
    """Configuration for :class:`BPETokenizer`.

    Attributes:
        pretokenizer: Registry key for the :class:`PreTokenizer` strategy, e.g.
            ``"whitespace"`` (didactic) or ``"byte_level"`` (production, lossless).
        min_pair_frequency: Stop training early if the best remaining pair occurs
            fewer than this many times -- avoids learning merges that only fire
            once or twice in the training corpus.
    """

    pretokenizer: str = "whitespace"
    min_pair_frequency: int = 2


class BPETrainer:
    """Learns a sequence of BPE merges from a text corpus.

    Args:
        pretokenizer: Strategy used to split text into words and words into their
            initial symbol sequence.
        min_pair_frequency: See :class:`BPETokenizerConfig`.
    """

    def __init__(self, pretokenizer: PreTokenizer, *, min_pair_frequency: int = 2) -> None:
        self._pretokenizer = pretokenizer
        self._min_pair_frequency = min_pair_frequency

    def train(self, texts: Iterable[str], *, num_merges: int) -> list[Merge]:
        """Learn up to ``num_merges`` merge rules from ``texts``.

        Args:
            texts: Training documents.
            num_merges: Maximum number of merges to learn.

        Returns:
            The learned merges, in the order they were applied (this order matters:
            :class:`BPETokenizer` replays them in the same order at encode time).
        """
        word_freq = self._count_words(texts)
        pair_counts, pair_to_words = self._index_pairs(word_freq)

        merges: list[Merge] = []
        for _ in range(num_merges):
            if not pair_counts:
                break
            best_pair, best_count = self._most_frequent_pair(pair_counts)
            if best_count < self._min_pair_frequency:
                break
            merges.append(best_pair)
            self._apply_merge(best_pair, word_freq, pair_counts, pair_to_words)

        return merges

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _count_words(self, texts: Iterable[str]) -> dict[tuple[str, ...], int]:
        counts: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for chunk in self._pretokenizer.split(text):
                counts[self._pretokenizer.to_symbols(chunk)] += 1
        return dict(counts)

    @staticmethod
    def _index_pairs(
        word_freq: dict[tuple[str, ...], int],
    ) -> tuple[Counter[Merge], dict[Merge, set[tuple[str, ...]]]]:
        pair_counts: Counter[Merge] = Counter()
        pair_to_words: dict[Merge, set[tuple[str, ...]]] = defaultdict(set)
        for word, freq in word_freq.items():
            for pair in itertools.pairwise(word):
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)
        return pair_counts, pair_to_words

    @staticmethod
    def _most_frequent_pair(pair_counts: Counter[Merge]) -> tuple[Merge, int]:
        # Ties broken lexicographically so training is deterministic across runs
        # and platforms, not dependent on dict/Counter iteration order.
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        return best_pair, pair_counts[best_pair]

    def _apply_merge(
        self,
        pair: Merge,
        word_freq: dict[tuple[str, ...], int],
        pair_counts: Counter[Merge],
        pair_to_words: dict[Merge, set[tuple[str, ...]]],
    ) -> None:
        merged_symbol = pair[0] + pair[1]
        affected = pair_to_words.pop(pair, set())
        del pair_counts[pair]

        for old_word in affected:
            freq = word_freq.pop(old_word, None)
            if freq is None:
                continue  # already replaced by a previous merge this round

            for old_pair in itertools.pairwise(old_word):
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                pair_to_words[old_pair].discard(old_word)

            new_word = _merge_symbols(old_word, pair, merged_symbol)
            word_freq[new_word] = word_freq.get(new_word, 0) + freq

            for new_pair in itertools.pairwise(new_word):
                pair_counts[new_pair] += freq
                pair_to_words[new_pair].add(new_word)


def _merge_symbols(word: tuple[str, ...], pair: Merge, merged: str) -> tuple[str, ...]:
    """Replace every non-overlapping occurrence of ``pair`` in ``word`` with ``merged``."""
    out: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


@TOKENIZERS.register("bpe")
class BPETokenizer(Tokenizer):
    """Applies a learned BPE merge table to encode and decode text.

    Train with :meth:`train`, which runs :class:`BPETrainer` internally and builds
    the vocabulary (special tokens, then single symbols, then merged symbols in
    learned order).
    """

    kind: ClassVar[str] = "bpe"
    config_cls: ClassVar[type[TokenizerConfig]] = BPETokenizerConfig

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)
        assert isinstance(config, BPETokenizerConfig)
        self._pretokenizer: PreTokenizer = PRETOKENIZERS.build(config.pretokenizer)
        self._merges: list[Merge] = []
        self._merge_rank: dict[Merge, int] = {}

    def train(self, texts: Iterable[str]) -> BPETokenizer:
        """Learn merges from a corpus and build the vocabulary.

        Args:
            texts: Training documents. Consumed twice (once to seed the alphabet,
                once inside the trainer) -- pass a list, not a one-shot generator.

        Returns:
            ``self``, for chaining.
        """
        config = self.config
        assert isinstance(config, BPETokenizerConfig)
        texts = list(texts)
        if config.lowercase:
            texts = [t.lower() for t in texts]

        alphabet = sorted({sym for t in texts for chunk in self._pretokenizer.split(t)
                            for sym in self._pretokenizer.to_symbols(chunk)})

        specials = list(SPECIAL_TOKENS.as_tuple())
        budget = config.vocab_size - len(specials) - len(alphabet)
        trainer = BPETrainer(self._pretokenizer, min_pair_frequency=config.min_pair_frequency)
        merges = trainer.train(texts, num_merges=max(budget, 0))

        self._merges = merges
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}
        merged_symbols = [a + b for a, b in merges]
        self._install_vocab([*specials, *alphabet, *merged_symbols])
        return self

    def encode(self, text: str) -> list[int]:
        """Pre-tokenize, apply learned merges greedily by rank, then map to ids."""
        if self.config.lowercase:
            text = text.lower()
        ids: list[int] = []
        for chunk in self._pretokenizer.split(text):
            symbols = list(self._pretokenizer.to_symbols(chunk))
            symbols = self._apply_merges(symbols)
            ids.extend(self.token_to_id(s) for s in symbols)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Map ids back to symbols and reassemble via the pre-tokenizer's join."""
        symbols = [self.id_to_token(i) for i in ids]
        return self._pretokenizer.join([self._pretokenizer.symbols_to_word(symbols)])

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        """Repeatedly apply the lowest-rank applicable merge until none remain."""
        while len(symbols) > 1:
            pairs = list(itertools.pairwise(symbols))
            ranked = [
                (self._merge_rank[p], i) for i, p in enumerate(pairs) if p in self._merge_rank
            ]
            if not ranked:
                break
            _, i = min(ranked)
            symbols = [*symbols[:i], symbols[i] + symbols[i + 1], *symbols[i + 2 :]]
        return symbols

    def _state_dict(self) -> dict[str, Any]:
        return {"vocab": self._id_to_token, "merges": [list(p) for p in self._merges]}

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self._install_vocab(list(state["vocab"]))
        self._merges = [tuple(p) for p in state["merges"]]
        self._merge_rank = {pair: i for i, pair in enumerate(self._merges)}
