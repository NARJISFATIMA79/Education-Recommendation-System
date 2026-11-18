import numpy as np
import pandas as pd

from intelligence import anomalies, comment_themes as ct, insights


def comments(texts, video="v1", channel="c1"):
    return pd.DataFrame({"comment_id": range(len(texts)), "video_id": video, "channel_id": channel,
                         "comment_text": texts})


def test_need_signals_duplicates_promo_and_empty_handling():
    texts = ["I am a beginner, please explain from scratch", "I am a beginner, please explain from scratch",
             "I am confused and I don't understand backpropagation", "Where can I download the notes pdf?",
             "Please make a video on reinforcement learning", "Register now www.spam.com free classes",
             "   ", "Great video thanks", "bhai bohat acha video hai mujhe samajh aaya"]
    df = ct.prepare_comments(comments(texts))
    assert df["is_duplicate"].sum() == 1 and df["is_promotional"].sum() == 1 and df["is_empty"].sum() == 1
    assert df["is_english"].iloc[8] == False                       # Roman-Hindi is not themed as English
    assert df["sig_beginner_friendly"].sum() == 1                  # duplicate not counted twice
    assert df["sig_difficulty_understanding"].sum() == 1 and df["sig_learning_resources"].sum() == 1
    assert df["sig_content_request"].sum() == 2            # "please explain..." and "please make a video..."
    assert not df.loc[5, "analysable"]                             # promotional excluded from need counts


def test_sentiment_is_separate_from_themes_and_empty_comments_are_not_neutral():
    r = ct.run_comment_analysis(comments(["Great video, thank you", "   ", "This is terrible and confusing"]))
    c = r["comments"]
    assert pd.isna(c.loc[1, "sentiment_label"]) and np.isnan(c.loc[1, "sentiment_compound"])   # missing, not "neutral"
    assert c.loc[0, "sentiment_label"] == "positive" and c.loc[2, "sentiment_label"] == "negative"


def test_theme_clustering_is_skipped_with_reason_when_data_is_insufficient():
    r = ct.run_comment_analysis(comments(["Great video thanks a lot for this"] * 10 + ["nice lesson sir"] * 5))
    assert r["theme_info"]["status"] == "skipped" and "at least" in r["theme_info"]["reason"]
    assert r["themes"].empty


def test_demand_vs_supply_flags_only_rare_requested_topics():
    texts = ["please make a video on reinforcement learning"] * 3 + ["video on machine learning basics"] * 3
    for i, t in enumerate(texts):                                  # distinct videos so nothing is deduplicated away
        pass
    df = pd.concat([comments([t], video=f"v{i}") for i, t in enumerate(texts)], ignore_index=True)
    r = ct.run_comment_analysis(df, supply_texts=["Machine learning basics tutorial", "Machine learning course"])
    d = r["demand_supply"].set_index("requested_topic")
    assert d.loc["reinforcement learning", "possible_gap"] and d.loc["reinforcement learning", "supply_videos"] == 0
    assert not d.loc["machine learning basics", "possible_gap"]


def test_anomaly_detection_finds_planted_outlier_and_never_tests_missing_values():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id": range(30), "t": [f"v{i}" for i in range(30)], "like_rate": rng.normal(.03, .004, 30),
                       "log_views": rng.normal(10, 1, 30)})
    df.loc[7, "like_rate"] = 0.12
    df.loc[3:12, "like_rate"] = np.where(np.arange(3, 13) == 7, 0.12, np.nan)
    recs, _ = anomalies.detect(df, "video", "id", "t", ["like_rate"])
    flagged = [r for r in recs if r["feature"] == "like_rate"]
    assert 7 in {r["id"] for r in flagged}                         # planted outlier found
    assert max(flagged, key=lambda r: abs(r["modified_z"]))["id"] == 7      # and it is the most extreme
    assert not ({r["id"] for r in flagged} & (set(range(3, 13)) - {7}))     # rows with missing values never flagged
    assert all("investigat" in r["note"] for r in recs)            # never framed as automatic success
    flat = df.assign(like_rate=0.03)
    assert anomalies.detect(flat, "video", "id", "t", ["like_rate"])[1]["like_rate"].startswith("no spread")
    assert "skipped" in anomalies.isolation_forest(df, "id", "t", ["like_rate", "log_views"])[1]


def _cluster_ctx(effect):
    rng = np.random.default_rng(3)
    rows = []
    for c, base in enumerate([0.05, 0.02, 0.03] if effect else [0.03, 0.03, 0.03]):
        for _ in range(25):
            rows.append({"cluster": float(c), "like_rate": base * float(np.exp(rng.normal(0, 0.15))),
                         "log_views_per_day": float(rng.normal(5, 1)), "log_duration_min": float(rng.normal(3, .5))})
    info = {"status": "ok", "k": 3, "silhouette_best": .4, "null_p_value": .03, "stability_ari": .9}
    prof = pd.DataFrame({"cluster": [0., 1., 2.], "top_terms": ["beginner, basics", "full course", "project"]})
    return pd.DataFrame(rows), info, prof


def test_insight_is_produced_only_when_the_data_supports_it_and_numbers_match():
    v, info, prof = _cluster_ctx(effect=True)
    ins = insights.Insights(); insights.content_insights(ins, v, info, prof)
    df = ins.frame()
    hit = df[df["title"].str.contains("higher like rate")]
    assert len(hit) == 1 and "beginner, basics" in hit.iloc[0]["observed_data"]
    best = v[v["cluster"] == 0]["like_rate"].median()
    assert f"{best:.4g}" in hit.iloc[0]["observed_data"]          # the quoted number is the computed number
    assert hit.iloc[0]["evidence_strength"] in ("low", "moderate")
    assert "sales" in hit.iloc[0]["interpretation"]               # high engagement is not equated with sales

    v2, info2, prof2 = _cluster_ctx(effect=False)
    ins2 = insights.Insights(); insights.content_insights(ins2, v2, info2, prof2)
    titles = " ".join(ins2.frame()["title"])
    assert "higher like rate" not in titles and "No reliable difference" in titles


def test_no_content_insight_when_clustering_was_not_performed():
    ins = insights.Insights()
    insights.content_insights(ins, pd.DataFrame({"a": range(10)}),
                              {"status": "skipped", "reason": "only 10 on-topic videos"}, pd.DataFrame())
    row = ins.frame().iloc[0]
    assert "not performed" in row["title"] and row["evidence_strength"] == "not established"
    assert "Not performed" in row["clustering_result"]


def test_audience_need_insights_require_repeated_evidence():
    texts = (["I am a beginner can you explain from scratch"] * 1 +
             ["beginner friendly please", "total beginner here", "new to this topic, explain basics"])
    df = pd.concat([comments([t], video=f"v{i}") for i, t in enumerate(texts * 8)], ignore_index=True)
    df["comment_text"] = [t + f" number {i}" for i, t in enumerate(df["comment_text"])]
    r = ct.run_comment_analysis(df)
    ins = insights.Insights(); insights.audience_insights(ins, r, "")
    titles = list(ins.frame()["title"])
    assert any("beginner" in t.lower() for t in titles)
    assert not any("price" in t.lower() for t in titles)          # not mentioned -> not reported
