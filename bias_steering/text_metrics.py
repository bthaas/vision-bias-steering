from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9']+")


def simple_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_spatial_ratio(n_spatial: int, n_descriptive: int) -> float:
    total = n_spatial + n_descriptive
    if total == 0:
        return 0.0
    return float((n_spatial - n_descriptive) / total)


@dataclass(frozen=True)
class RatioCounts:
    spatial_hits: int
    descriptive_hits: int
    normalized_spatial_ratio: float


class SpatialRatioScorer:
    """Count tracked lexicon hits and compute the normalized spatial ratio."""

    def __init__(self, spatial_terms: Iterable[str], descriptive_terms: Iterable[str]):
        self._catalog: list[tuple[int, tuple[str, ...], str]] = []
        max_len = 1
        seen = {}

        for label, terms in (
            ("spatial", spatial_terms),
            ("descriptive", descriptive_terms),
        ):
            for term in terms:
                token_tuple = tuple(simple_tokens(term))
                if not token_tuple:
                    continue
                if token_tuple in seen and seen[token_tuple] != label:
                    raise ValueError(f"Lexicon overlap across labels for phrase: {term!r}")
                seen[token_tuple] = label
                max_len = max(max_len, len(token_tuple))
                self._catalog.append((len(token_tuple), token_tuple, label))

        self._catalog.sort(key=lambda item: (-item[0], item[1]))
        self.max_phrase_len = max_len

    def counts(self, text: str) -> RatioCounts:
        tokens = simple_tokens(text)
        if not tokens:
            return RatioCounts(0, 0, 0.0)

        occupied = [False] * len(tokens)
        spatial_hits = 0
        descriptive_hits = 0

        for phrase_len in range(self.max_phrase_len, 0, -1):
            phrases = [
                (phrase_tokens, label)
                for length, phrase_tokens, label in self._catalog
                if length == phrase_len
            ]
            if not phrases:
                continue

            i = 0
            while i <= len(tokens) - phrase_len:
                if any(occupied[i : i + phrase_len]):
                    i += 1
                    continue
                span = tuple(tokens[i : i + phrase_len])
                matched_label = None
                for phrase_tokens, label in phrases:
                    if span == phrase_tokens:
                        matched_label = label
                        break
                if matched_label is None:
                    i += 1
                    continue
                for j in range(i, i + phrase_len):
                    occupied[j] = True
                if matched_label == "spatial":
                    spatial_hits += 1
                else:
                    descriptive_hits += 1
                i += phrase_len

        return RatioCounts(
            spatial_hits=spatial_hits,
            descriptive_hits=descriptive_hits,
            normalized_spatial_ratio=normalize_spatial_ratio(spatial_hits, descriptive_hits),
        )

    def score(self, text: str) -> float:
        return self.counts(text).normalized_spatial_ratio
