"""Feature engineering from the data the original pipeline collects.

Rules applied everywhere:
  * Missing views / likes / subscribers stay NaN. They are never turned into 0.
  * Rates use only rows where numerator AND denominator are present.
  * Unmeasured sentiment is NaN, never neutral.
  * Identifiers (channel_id, video_id, titles) are carried for labelling only and are never features.
"""
import re

import numpy as np
import pandas as pd

from recommender import calculate_topic_relevance          # ORIGINAL function, reused unchanged

MIN_COMMENTS_FOR_SENTIMENT = 10    # below this a channel's sentiment is treated as not reliably measured

# Features used for channel clustering, with the reason each is included (also written to the report).
CHANNEL_FEATURES = {
    "log_views": ("Reach: mean of log(1 + views) over the channel's sampled on-topic videos", "higher"),
    "like_rate": ("Audience response: pooled likes / views", "higher"),
    "comment_rate": ("Discussion: pooled comment count / views", "higher"),
    "log_duration_min": ("Format: mean log(1 + duration in minutes)", "longer"),
    "log_n_videos": ("Supply depth: log(1 + number of on-topic videos in the sample)", "more"),
    "topic_relevance": ("Fit to the searched topic (title word overlap, from the original engine)", "higher"),
    "log_subscribers": ("Channel scale: log(1 + subscribers); hidden counts stay missing", "higher"),
    "sentiment_adjusted": ("Audience tone: the original engine's confidence-adjusted mean VADER score; "
                           "only for channels with enough analysed comments", "more positive"),
}
VIDEO_PERFORMANCE_COLUMNS = ["views", "likes", "like_rate", "comment_rate", "views_per_day", "age_days",
                             "duration_min"]

_ISO = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def parse_iso_duration_minutes(value):
    """'PT1H19M52S' -> minutes. Missing, malformed or zero-length (live streams) -> NaN."""
    if not isinstance(value, str):
        return np.nan
    m = _ISO.match(value.strip())
    if not m:
        return np.nan
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    minutes = (d * 86400 + h * 3600 + mi * 60 + s) / 60.0
    return minutes if minutes > 0 else np.nan


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _safe_rate(numerator, denominator):
    """numerator / denominator, NaN when either is missing or the denominator is not positive."""
    out = numerator / denominator.where(denominator > 0)
    return out.clip(lower=0, upper=1)


def build_video_table(videos, topic, as_of=None):
    """One row per video with engagement metrics. Missing stays NaN."""
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of, tz="UTC") \
        if pd.Timestamp(as_of).tzinfo is None else pd.Timestamp(as_of)
    df = videos.copy()
    for col in ("video_id", "channel_id", "channel_title", "video_title"):
        if col not in df.columns:
            df[col] = np.nan
    t = pd.DataFrame({
        "video_id": df["video_id"], "channel_id": df["channel_id"],
        "channel_title": df["channel_title"], "video_title": df["video_title"],
    })
    t["views"] = _num(df["views"]) if "views" in df else np.nan
    t["likes"] = _num(df["likes"]) if "likes" in df else np.nan
    comments_col = "comments" if "comments" in df else ("comment_count" if "comment_count" in df else None)
    t["comment_count"] = _num(df[comments_col]) if comments_col else np.nan
    t["duration_min"] = df["duration"].map(parse_iso_duration_minutes) if "duration" in df else np.nan
    published = pd.to_datetime(df["published_at"], utc=True, errors="coerce") if "published_at" in df \
        else pd.Series(pd.NaT, index=df.index)
    t["age_days"] = ((as_of - published).dt.total_seconds() / 86400).clip(lower=1)
    t["like_rate"] = _safe_rate(t["likes"], t["views"])
    t["comment_rate"] = _safe_rate(t["comment_count"], t["views"])
    t["views_per_day"] = t["views"] / t["age_days"]
    t["log_views_per_day"] = np.log1p(t["views_per_day"])
    t["log_views"] = np.log1p(t["views"])
    t["log_duration_min"] = np.log1p(t["duration_min"])
    t["topic_relevance"] = t["video_title"].apply(
        lambda title: calculate_topic_relevance(topic, title) if isinstance(title, str) else 0.0)
    text_parts = [df[c].fillna("").astype(str) for c in ("video_title", "description", "tags") if c in df]
    t["text"] = (text_parts[0] if text_parts else pd.Series("", index=df.index))
    for part in text_parts[1:]:
        t["text"] = t["text"] + " " + part
    t["text"] = t["text"].str.strip()
    return t.reset_index(drop=True)


def build_channel_table(video_table, channels_meta, recommendations_all,
                        min_comments_for_sentiment=MIN_COMMENTS_FOR_SENTIMENT):
    """One row per candidate channel (those ranked by the original engine)."""
    rec = recommendations_all.copy()
    rows = []
    for _, r in rec.iterrows():
        cid = r["channel_id"]
        vids = video_table[video_table["channel_id"] == cid]
        both = vids[vids["views"].notna() & vids["likes"].notna() & (vids["views"] > 0)]
        both_c = vids[vids["views"].notna() & vids["comment_count"].notna() & (vids["views"] > 0)]
        n_comments = int(r.get("analyzed_comments", 0) or 0)
        if n_comments == 0:
            status = "not_measured"
        elif n_comments < min_comments_for_sentiment:
            status = "insufficient_comments"
        else:
            status = "measured"
        rows.append({
            "channel_id": cid, "channel_title": r["channel_title"],
            "recommendation_rank": None, "recommendation_score": r.get("recommendation_score"),
            "n_videos": len(vids),
            "log_n_videos": np.log1p(len(vids)),
            "log_views": vids["log_views"].mean() if vids["views"].notna().any() else np.nan,
            "mean_views": vids["views"].mean(),
            "like_rate": both["likes"].sum() / both["views"].sum() if len(both) else np.nan,
            "comment_rate": both_c["comment_count"].sum() / both_c["views"].sum() if len(both_c) else np.nan,
            "mean_duration_min": vids["duration_min"].mean(),
            "log_duration_min": vids["log_duration_min"].mean() if vids["duration_min"].notna().any() else np.nan,
            "topic_relevance": r.get("topic_relevance_score"),
            "analyzed_comments": n_comments,
            "sentiment_status": status,
            "sentiment_raw": r.get("raw_sentiment") if status != "not_measured" else np.nan,
            "sentiment_adjusted": (r.get("confidence_adjusted_sentiment") if status == "measured" else np.nan),
        })
    table = pd.DataFrame(rows)
    table["recommendation_rank"] = range(1, len(table) + 1)      # recommend_channels returns rank order
    meta = channels_meta.drop_duplicates("channel_id") if channels_meta is not None else pd.DataFrame()
    if len(meta):
        sub = _num(meta.get("subscribers", pd.Series(np.nan, index=meta.index)))
        hidden = meta.get("subscriber_count_hidden", pd.Series(False, index=meta.index)).fillna(False).astype(bool)
        sub = sub.where(~hidden)                                  # hidden count -> missing, not 0
        lookup = pd.DataFrame({"channel_id": meta["channel_id"], "subscribers": sub})
        table = table.merge(lookup, on="channel_id", how="left")
    else:
        table["subscribers"] = np.nan
    table["log_subscribers"] = np.log1p(table["subscribers"])
    return table
