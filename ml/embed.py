"""Sentence-embedding features for the risk classifier's second
experiment (see docs/ml_experiment.md). Reuses the exact same pretrained
model Sanad's own retrieval pipeline already depends on
(`sanad/config.py`'s `embedding_model` default) -- not a new dependency
choice, just the same one applied to a different problem, so a clause's
vector represents its *meaning* rather than its exact vocabulary. This
is the fix for the first experiment's TF-IDF failure mode: with ~10
positive training examples, a bag-of-words model needs near-exact
wording overlap to generalize at all, while embeddings let a "sounds
like a non-compete clause" match fire even on a clause phrased
completely differently from the ones seen during training.
"""
from __future__ import annotations

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # Pinned to CPU -- see sanad/rag/embeddings.py's Embedder for why:
        # letting this auto-select "mps" caused a real, reproducible
        # server crash (no Python traceback, a native-level abort) when
        # combined with the forked worker processes sentence-transformers
        # uses internally for batch encoding.
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encodes every text once; embeddings don't depend on any train/test
    split (unlike a TF-IDF vocabulary, which is fit per-fold), so this is
    computed a single time up front and reused across every
    leave-one-out fold in train_risk_classifier.py."""
    return _get_model().encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
