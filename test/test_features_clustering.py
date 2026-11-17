import numpy as np
import pandas as pd
import pytest

from intelligence import clustering, features


def test_missing_values_stay_missing_never_zero():
    v = pd.DataFrame({"video_id": ["a", "b", "c"], "channel_id": ["x", "x", "y"], "channel_title": ["X", "X", "Y"],
                      "video_title": ["big data a", "big data b", "big data c"],
                      "views": [1000, 2000, np.nan], "likes": [50, np.nan, 10], "comments": [5, 6, 7],
                      "duration": ["PT10M", "P0D", None], "published_at": ["2024-01-01T00:00:00Z"] * 3})
    t = features.build_video_table(v, "big data", as_of="2025-01-01")
    assert t.loc[0, "like_rate"] == pytest.approx(0.05)
    assert np.isnan(t.loc[1, "like_rate"])                      # likes missing -> NaN, not 0
    assert np.isnan(t.loc[2, "like_rate"]) and np.isnan(t.loc[2, "views"])
    assert np.isnan(t.loc[1, "duration_min"]) and np.isnan(t.loc[2, "duration_min"])   # live/missing != 0


def test_channel_rates_pool_only_complete_videos_and_hidden_subscribers_stay_missing():
    vt = pd.DataFrame({"channel_id": ["x", "x"], "views": [1000.0, 3000.0], "likes": [100.0, np.nan],
                       "comment_count": [10.0, 30.0], "duration_min": [10.0, 20.0],
                       "log_views": np.log1p([1000, 3000]), "log_duration_min": np.log1p([10, 20])})
    rec = pd.DataFrame({"channel_id": ["x"], "channel_title": ["X"], "recommendation_score": [0.5],
                        "topic_relevance_score": [1.0], "analyzed_comments": [0], "raw_sentiment": [0.0],
                        "confidence_adjusted_sentiment": [0.0]})
    meta = pd.DataFrame({"channel_id": ["x"], "subscribers": [5000], "subscriber_count_hidden": [True]})
    t = features.build_channel_table(vt, meta, rec)
    assert t.loc[0, "like_rate"] == pytest.approx(0.1)          # only the video with both numbers
    assert t.loc[0, "comment_rate"] == pytest.approx(40 / 4000)
    assert np.isnan(t.loc[0, "log_subscribers"])                # hidden -> missing
    assert t.loc[0, "sentiment_status"] == "not_measured" and np.isnan(t.loc[0, "sentiment_adjusted"])


def test_unmeasured_and_insufficient_sentiment_are_not_neutral():
    rec = pd.DataFrame({"channel_id": ["a", "b", "c"], "channel_title": list("ABC"),
                        "recommendation_score": [.3, .2, .1], "topic_relevance_score": [1, 1, 1],
                        "analyzed_comments": [0, 5, 50], "raw_sentiment": [0.0, 0.4, 0.3],
                        "confidence_adjusted_sentiment": [0.0, 0.08, 0.2]})
    vt = pd.DataFrame({"channel_id": list("abc"), "views": [1., 2., 3.], "likes": [1., 1., 1.],
                       "comment_count": [1., 1., 1.], "duration_min": [5., 5., 5.], "log_views": [1., 1., 1.],
                       "log_duration_min": [1., 1., 1.]})
    t = features.build_channel_table(vt, None, rec)
    assert list(t["sentiment_status"]) == ["not_measured", "insufficient_comments", "measured"]
    assert t["sentiment_adjusted"].isna().tolist() == [True, True, False]


def test_identifiers_are_never_clustering_features():
    forbidden = {"channel_id", "video_id", "channel_title", "video_title", "cluster", "recommendation_rank"}
    assert forbidden.isdisjoint(features.CHANNEL_FEATURES)


def blobs(n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(m, 0.3, (n_per, 3)) for m in (0, 5, 10)])
    df = pd.DataFrame(X, columns=list("abc"))
    df["id"] = range(len(df))
    return df


def test_clustering_recovers_three_true_groups_with_elbow_and_silhouette_agreeing():
    info, out = clustering.cluster_numeric(blobs(), list("abc"), ["id"], 8, 3)
    truth = np.repeat([0, 1, 2], 20)
    from sklearn.metrics import adjusted_rand_score
    assert info["status"] == "ok" and info["k"] == 3 and info["elbow_k"] == 3
    assert adjusted_rand_score(truth, out["cluster"].astype(int)) == 1.0
    assert info["stability_ari"] > 0.9


def test_k_is_not_assumed_to_be_three():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(m, 0.3, (20, 3)) for m in (0, 6)])
    df = pd.DataFrame(X, columns=list("abc")); df["id"] = range(40)
    info, _ = clustering.cluster_numeric(df, list("abc"), ["id"], 8, 3)
    assert info["k"] == 2


def test_structureless_data_is_not_reported_as_clusters():
    wrong = 0
    for s in range(8):
        rng = np.random.default_rng(100 + s)
        df = pd.DataFrame(rng.normal(0, 1, (40, 3)), columns=list("abc")); df["id"] = range(40)
        info, _ = clustering.cluster_numeric(df, list("abc"), ["id"], 8, 3)
        wrong += info["status"] == "ok"
    assert wrong <= 1                                             # ~5% false-positive rate by construction


def test_too_few_rows_and_missing_features_are_handled_not_faked():
    info, out = clustering.cluster_numeric(blobs().head(5), list("abc"), ["id"], 8, 3)
    assert info["status"] == "skipped" and "at least 8" in info["reason"] and out["cluster"].isna().all()
    df = blobs(); df.loc[:30, "c"] = np.nan                        # 50% coverage -> feature dropped
    info, _ = clustering.cluster_numeric(df, list("abc"), ["id"], 8, 3)
    assert "c" in info["features_dropped"] and info["features_used"] == ["a", "b"]
    df = blobs(); df["const"] = 1.0
    info, _ = clustering.cluster_numeric(df, ["a", "b", "const"], ["id"], 8, 3)
    assert "const" in info["features_dropped"]


def test_incomplete_rows_are_excluded_and_reported():
    df = blobs(); df.loc[0, "a"] = np.nan
    info, out = clustering.cluster_numeric(df, list("abc"), ["id"], 8, 3)
    assert info["n_clustered"] == 59 and info["excluded_incomplete"] == [0] and np.isnan(out.loc[0, "cluster"])


def test_features_are_scaled_so_a_huge_unit_cannot_dominate():
    df = blobs(); a, _ = clustering.cluster_numeric(df, list("abc"), ["id"], 8, 3)
    df2 = df.copy(); df2["a"] = df2["a"] * 1e6                     # same data, absurd units
    b, _ = clustering.cluster_numeric(df2, list("abc"), ["id"], 8, 3)
    assert a["k"] == b["k"] and a["silhouette_best"] == pytest.approx(b["silhouette_best"], abs=1e-6)


def test_text_clustering_finds_planted_topics_and_skips_small_corpora():
    topics = {"cats": "cat kitten purr whiskers feline pet", "cars": "car engine wheel drive road fuel",
              "food": "pizza pasta cheese sauce bake oven"}
    rng = np.random.default_rng(2)
    docs, truth = [], []
    for ti, words in enumerate(topics.values()):
        w = words.split()
        for _ in range(30):
            docs.append(" ".join(rng.choice(w, 5)) + f" extra{int(rng.integers(5))}"); truth.append(ti)
    info, fit = clustering.cluster_texts(docs, min_docs=30, min_cluster_size=5, min_df=2)
    from sklearn.metrics import adjusted_rand_score
    assert info["status"] == "ok" and adjusted_rand_score(truth, fit["labels"]) > 0.9
    small_info, small_fit = clustering.cluster_texts(docs[:10], min_docs=30)
    assert small_info["status"] == "skipped" and small_fit is None


def test_unsupported_clusters_are_never_assigned_to_items():
    """Regression: 'no reliable structure' must not leave cluster labels on the data."""
    rng = np.random.default_rng(100)
    df = pd.DataFrame(rng.normal(0, 1, (40, 3)), columns=list("abc")); df["id"] = range(40)
    for seed in range(100, 120):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.normal(0, 1, (40, 3)), columns=list("abc")); df["id"] = range(40)
        info, out = clustering.cluster_numeric(df, list("abc"), ["id"], 8, 3)
        if info["status"] == "no_structure":
            assert out["cluster"].isna().all() and "centroid_z" not in info
            return
    raise AssertionError("expected at least one structureless dataset in 20 draws")
