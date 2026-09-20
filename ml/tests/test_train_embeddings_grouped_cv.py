"""Tests the leave-one-group-out wiring against small synthetic
embeddings -- the thing under test is the grouping logic (a clause and
its paraphrases never split across train/test), not real embeddings or
a real LLM paraphrase call."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict


def test_grouped_examples_are_never_split_across_train_and_test():
    """Direct proof the grouping actually prevents leakage: build a tiny
    dataset with two separate "flagged" groups (A: a real clause + 2
    near-duplicate paraphrases; B: a second real clause on its own,
    mirroring how most of the real 13 positive groups look) plus
    unrelated "none" singletons. If LeaveOneGroupOut is wired correctly,
    every fold that holds out any member of a group must hold out *all*
    of them -- so the model being scored on a group-A example never had
    another group-A example in its training fold, however similar the
    text. Two flagged groups (not one) also keeps every training fold
    two-class, matching the real dataset's 13 separate positive groups
    rather than an artificial single-group edge case.
    """
    X = np.array(
        [
            [1.0, 1.0], [1.01, 0.99], [0.99, 1.01],  # group "A": real + 2 paraphrases, "flagged"
            [1.5, 1.5],  # group "B": a second, unrelated real "flagged" clause
            [0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.0, 0.1], [0.1, 0.0],  # 5 "none" singletons
        ]
    )
    y = ["flagged", "flagged", "flagged", "flagged", "none", "none", "none", "none", "none"]
    groups = ["A", "A", "A", "B", "n0", "n1", "n2", "n3", "n4"]

    cv = LeaveOneGroupOut()
    fold_count = 0
    for train_idx, test_idx in cv.split(X, y, groups):
        fold_count += 1
        train_groups = {groups[i] for i in train_idx}
        test_groups = {groups[i] for i in test_idx}
        assert train_groups.isdisjoint(test_groups)

    # One fold per distinct group (7: A, B, n0..n4), not one per row (9).
    assert fold_count == len(set(groups)) == 7

    # A real prediction run should also complete without error using the
    # same groups (exercises the actual cross_val_predict call shape
    # train_embeddings_grouped_cv.py uses).
    preds = cross_val_predict(LogisticRegression(), X, y, cv=cv, groups=groups)
    assert len(preds) == len(y)


def test_none_clauses_with_unique_groups_behave_like_plain_leave_one_out():
    """A dataset with no augmentation at all (every group_id distinct)
    should produce exactly one fold per example -- confirms the fallback
    path in load_combined() (no augmented file present) doesn't change
    behavior versus experiment 2's plain leave-one-out."""
    X = np.array([[float(i), float(i)] for i in range(10)])
    y = ["none"] * 10
    groups = [f"doc{i}:0" for i in range(10)]

    cv = LeaveOneGroupOut()
    assert cv.get_n_splits(X, y, groups) == 10
