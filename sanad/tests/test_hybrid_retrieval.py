from sanad.rag.hybrid_retrieval import reciprocal_rank_fusion


def test_item_ranked_first_in_both_lists_scores_highest():
    dense = [1, 2, 3]
    sparse = [1, 3, 2]
    scores = reciprocal_rank_fusion([dense, sparse])
    assert max(scores, key=scores.get) == 1


def test_item_missing_from_one_ranking_still_gets_a_score():
    dense = [1, 2]
    sparse = [3]  # 3 never appears in dense
    scores = reciprocal_rank_fusion([dense, sparse])
    assert set(scores) == {1, 2, 3}
    assert scores[3] > 0


def test_appearing_in_both_rankings_beats_appearing_in_only_one():
    dense = [1, 2, 3]
    sparse = [4, 2, 1]
    scores = reciprocal_rank_fusion([dense, sparse])
    # item 2 and 1 both appear in both rankings; 3 and 4 appear in only one
    assert scores[1] > scores[3]
    assert scores[2] > scores[4]


def test_single_ranking_preserves_relative_order():
    scores = reciprocal_rank_fusion([[10, 20, 30]])
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked == [10, 20, 30]


def test_empty_rankings_produce_no_scores():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_smaller_k_makes_rank_position_matter_more():
    dense = [1, 2, 3]
    scores_small_k = reciprocal_rank_fusion([dense], k=1)
    scores_large_k = reciprocal_rank_fusion([dense], k=1000)
    # with small k, the gap between rank 0 and rank 2 is proportionally
    # much larger than with a huge k, where every rank looks similar
    gap_small_k = scores_small_k[1] - scores_small_k[3]
    gap_large_k = scores_large_k[1] - scores_large_k[3]
    assert gap_small_k > gap_large_k
