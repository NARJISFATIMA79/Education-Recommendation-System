"""Requirement 2 and 7: the original code, recommendation results and CSV workflow are not changed."""
import hashlib
import os

import pandas as pd

from conftest import ROOT


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_every_original_file_is_byte_identical():
    bad = []
    for line in open(os.path.join(ROOT, "ORIGINAL_FILES.sha256")):
        digest, path = line.strip().split("  ", 1)
        path = os.path.join(ROOT, path)
        if not os.path.exists(path) or sha(path) != digest:
            bad.append(path)
    assert bad == []


def test_recommendations_equal_the_original_baseline_on_the_same_input():
    """Same input data -> same ranking and scores as the ORIGINAL program produced before this enhancement."""
    from recommender import recommend_channels
    d = os.path.join(ROOT, "data")
    v = pd.read_csv(f"{d}/video_metadata.csv", engine="python")
    c = pd.read_csv(f"{d}/channel_metadata.csv", engine="python")
    k = pd.read_csv(f"{d}/cleaned_comments.csv", engine="python")
    now = recommend_channels("big data analytics", v, c, k, top_n=15)
    base = pd.read_csv(os.path.join(ROOT, "tests", "fixtures", "original_baseline_recommendations.csv"))
    assert list(now["channel_id"]) == list(base["channel_id"])
    for col in ("topic_relevance_score", "sentiment_score", "engagement_score", "views_score",
                "recommendation_score"):
        assert (now[col].to_numpy() - base[col].to_numpy()).__abs__().max() < 1e-12, col


def test_enhanced_pipeline_passes_original_scores_through_unchanged(tmp_path):
    import enhanced_main
    from recommender import recommend_channels
    d = os.path.join(ROOT, "data")
    before = {f: sha(f"{d}/{f}") for f in ("video_metadata.csv", "channel_metadata.csv", "raw_comments.csv",
                                           "cleaned_comments.csv")}
    from intelligence.pipeline import run_intelligence
    v = pd.read_csv(f"{d}/video_metadata.csv", engine="python")
    c = pd.read_csv(f"{d}/channel_metadata.csv", engine="python")
    k = pd.read_csv(f"{d}/cleaned_comments.csv", engine="python")
    allrecs = recommend_channels("Big Data Analytics", v, c, k, top_n=10 ** 9)
    run_intelligence("Big Data Analytics", v, c, k, allrecs, out_dir=str(tmp_path), as_of="2026-07-15")
    out = pd.read_csv(tmp_path / "recommendations.csv")
    assert list(out["channel_id"]) == list(allrecs["channel_id"])
    assert (out["recommendation_score"] - allrecs["recommendation_score"].to_numpy()).abs().max() < 1e-12      # CSV text round-trip, not a calculation difference
    assert out["shown_in_original_top_n"].sum() == 10
    after = {f: sha(f"{d}/{f}") for f in before}
    assert before == after                                     # the intelligence layer never edits the CSVs
