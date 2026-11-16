"""Runs the whole intelligence layer on data produced by the ORIGINAL pipeline and writes the outputs."""
import datetime as dt
import json
import os
import platform

import numpy as np
import pandas as pd
import sklearn

from intelligence import anomalies, clustering, comment_themes, features, insights
from intelligence.report import render_report

MIN_CHANNELS_TO_CLUSTER = 8
MIN_VIDEOS_TO_CLUSTER = 30
FEATURE_LABELS = {
    "log_views": "reach (views)", "like_rate": "like rate", "comment_rate": "comment rate",
    "log_duration_min": "video length", "log_n_videos": "number of on-topic videos",
    "topic_relevance": "topic relevance", "log_subscribers": "subscriber count",
    "sentiment_adjusted": "audience sentiment"}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, dt.datetime)):
        return o.isoformat()
    return str(o)


def build_video_source(videos, market_sample):
    """Topic sample (preferred) united with the original video metadata; sample rows win on duplicates."""
    if market_sample is not None and len(market_sample):
        extra = videos[~videos["video_id"].isin(market_sample["video_id"])]
        return pd.concat([market_sample, extra], ignore_index=True, sort=False), "topic_sample"
    return videos.copy(), "original_selected_videos_only"


def run_intelligence(topic, videos, channels, cleaned_comments, recommendations_all, market_sample=None,
                     out_dir="data/intelligence", as_of=None, api_info=None, top_n=10):
    os.makedirs(out_dir, exist_ok=True)
    summary = {"topic": topic, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                                    "pandas": pd.__version__, "numpy": np.__version__},
               "api_info": api_info or {}, "skipped": {}}

    # ---- video table (on-topic filter identical in spirit to the original relevance filter)
    source, source_name = build_video_source(videos, market_sample)
    vt_all = features.build_video_table(source, topic, as_of=as_of)
    for extra_col in ("category_id", "in_original_selection"):
        if extra_col in source:
            vt_all[extra_col] = source[extra_col].to_numpy()
    vt_all = vt_all.drop_duplicates("video_id")
    video_table = vt_all[vt_all["topic_relevance"] > 0].reset_index(drop=True)
    summary["video_sample"] = {"source": source_name, "videos_before_topic_filter": len(vt_all),
                               "videos_on_topic": len(video_table)}

    # ---- channel table + clustering
    channel_table = features.build_channel_table(video_table, channels, recommendations_all)
    ch_info, channel_table = clustering.cluster_numeric(
        channel_table, list(features.CHANNEL_FEATURES), ["channel_title"], MIN_CHANNELS_TO_CLUSTER, 2,
        k_cap=5, label="channels")
    ch_profile = clustering.describe_clusters(
        channel_table, ch_info, FEATURE_LABELS,
        ["mean_views", "like_rate", "comment_rate", "mean_duration_min", "n_videos", "topic_relevance",
         "sentiment_adjusted", "recommendation_score"])
    summary["channel_clustering"] = ch_info

    # ---- video clustering (content type from text; performance only profiles the clusters)
    vid_info, vid_fit = {"status": "skipped"}, None
    video_profile = pd.DataFrame()
    text_ok = video_table["text"].str.len() > 0
    if len(video_table) < MIN_VIDEOS_TO_CLUSTER:
        vid_info = {"status": "skipped", "n_docs": len(video_table),
                    "reason": f"only {len(video_table)} on-topic videos; at least {MIN_VIDEOS_TO_CLUSTER} are needed "
                              "(the original collector keeps one video per channel; use extra search pages)"}
    else:
        pool = video_table[text_ok]
        vid_info, vid_fit = clustering.cluster_texts(pool["text"], min_docs=MIN_VIDEOS_TO_CLUSTER, k_max=8,
                                                     min_cluster_size=max(5, len(pool) // 20), min_df=2, max_df=0.5)
        video_table["cluster"] = np.nan
        if vid_fit is not None:
            video_table.loc[pool.index, "cluster"] = vid_fit["labels"]
            rows = []
            pidx = pool.index.to_numpy()
            for c, meta in vid_fit["clusters"].items():
                m = video_table[video_table["cluster"] == c]
                rows.append({"cluster": c, "n": len(m), "top_terms": ", ".join(meta["top_terms"][:6]),
                             "representative_titles": " || ".join(
                                 str(video_table.loc[pidx[i], "video_title"])[:90] for i in meta["representative_idx"]),
                             "median_views": m["views"].median(), "median_like_rate": m["like_rate"].median(),
                             "median_comment_rate": m["comment_rate"].median(),
                             "median_views_per_day": m["views_per_day"].median(),
                             "median_duration_min": m["duration_min"].median(),
                             "videos_with_likes": int(m["like_rate"].notna().sum())})
            video_profile = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    if "cluster" not in video_table:
        video_table["cluster"] = np.nan
    summary["video_clustering"] = {k: v for k, v in vid_info.items() if k != "labels"}

    # ---- comments
    comment_result = comment_themes.run_comment_analysis(cleaned_comments, supply_texts=video_table["text"])
    summary["comments"] = comment_result["counts"]
    summary["comment_theme_clustering"] = comment_result["theme_info"]

    # ---- anomalies
    anomaly_df, anomaly_diag = anomalies.run_anomaly_detection(channel_table, video_table)
    summary["anomaly_detection"] = anomaly_diag

    # ---- insights
    meta = {"n_channels": len(channel_table), "n_videos": len(video_table),
            "n_comments": comment_result["counts"]["total"], "n_analysable": comment_result["counts"]["analysable"],
            "n_unmeasured": int((channel_table["sentiment_status"] != "measured").sum())}
    ctx = {"video_table": video_table, "video_info": vid_info, "video_profile": video_profile,
           "channel_table": channel_table, "channel_info": ch_info, "channel_profile": ch_profile,
           "comments": comment_result, "anomalies": anomaly_df, "anomaly_diag": anomaly_diag, "meta": meta}
    insight_df = insights.generate(ctx)

    # ---- write outputs
    rec = recommendations_all.copy()
    rec.insert(0, "rank", range(1, len(rec) + 1))
    rec["shown_in_original_top_n"] = rec["rank"] <= top_n
    rec["sentiment_measured"] = rec["analyzed_comments"] > 0
    outputs = {
        "recommendations.csv": rec,
        "channel_clusters.csv": channel_table.drop(columns=[c for c in ("log_n_videos",) if c in channel_table]),
        "channel_cluster_profiles.csv": ch_profile,
        "video_clusters.csv": video_table.drop(columns=["text"]),
        "video_cluster_profiles.csv": video_profile,
        "comment_signals.csv": comment_result["signals"],
        "comment_themes.csv": comment_result["themes"],
        "comment_level_analysis.csv": comment_result["comments"].drop(columns=["norm"]),
        "question_terms.csv": comment_result["question_terms"],
        "demand_vs_supply.csv": comment_result["demand_supply"],
        "anomalies.csv": anomaly_df,
        "marketing_insights.csv": insight_df,
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(out_dir, name), index=False, encoding="utf-8-sig")
    summary["feature_documentation"] = {k: v[0] for k, v in features.CHANNEL_FEATURES.items()}
    summary["n_insights"] = len(insight_df)
    with open(os.path.join(out_dir, "intelligence_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_jsonable)
    report = render_report(topic, summary, rec, channel_table, ch_info, ch_profile, video_table, vid_info,
                           video_profile, comment_result, anomaly_df, anomaly_diag, insight_df, top_n)
    with open(os.path.join(out_dir, "intelligence_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    return {"summary": summary, "channel_table": channel_table, "video_table": video_table,
            "insights": insight_df, "comments": comment_result, "anomalies": anomaly_df, "out_dir": out_dir}
