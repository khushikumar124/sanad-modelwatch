"""Reciprocal Rank Fusion (RRF) for combining ranked result lists.

Used to combine a dense (embedding cosine similarity) ranking and a
sparse (BM25 lexical) ranking of the same document's chunks into one
ranking. RRF only uses rank *position*, not the underlying scores, so a
cosine distance in [0, 2] and a BM25 score in [0, ~50] combine correctly
with no normalization step -- normalizing scores from two different
scoring functions onto a shared scale is its own hard problem, and RRF
sidesteps it entirely. This is why hybrid retrieval reaches for RRF
rather than e.g. a weighted sum of the two raw scores.
"""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """rankings: one or more lists of item ids in ranked order (best
    first, ties broken by whatever order the caller supplies).

    Returns {item_id: fused_score}, higher is better. An id missing from
    one of the rankings simply gets no contribution from it -- it is not
    penalized for that ranking's absence, only rewarded for the rankings
    it does appear in.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
