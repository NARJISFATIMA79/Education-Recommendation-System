"""Anomaly detection: things that deserve investigation. An anomaly is NOT automatically good news.

Method (chosen for small samples):
  * modified z-score (median / MAD, Iglewicz & Hoaglin): flag |z| > 3.5 on one feature at a time
  * 'engagement relative to reach': residual of like rate regressed on log views with a robust (Theil-Sen) line
  * Isolation Forest (multivariate) only when there are at least 50 videos
Missing values are never filled: an item is simply not tested on a feature it lacks.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MIN_N = 8
Z_THRESHOLD = 3.5
IF_MIN_N = 50
IF_CONTAMINATION = 0.05
SEED = 42

FEATURE_TEXT = {
    "like_rate": "like rate", "comment_rate": "comment rate", "log_views": "views",
    "log_views_per_day": "views per day since publication", "sentiment_adjusted": "audience sentiment",
    "like_rate_vs_views": "like rate relative to what its views would predict",
}
CONFOUNDERS = ("video age, language/region of the audience, topic popularity and whether the creator promotes "
               "the video elsewhere are not controlled here")


def modified_z(series):
    """0.6745 (x - median) / MAD. Returns NaN everywhere if MAD is 0 (no spread to measure against)."""
    x = series.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return 0.6745 * (x - med) / mad


def robust_residual(df, y, x):
    """Residual of y ~ x using the Theil-Sen estimator; NaN where either value is missing."""
    ok = df[[x, y]].dropna()
    out = pd.Series(np.nan, index=df.index)
    if len(ok) < MIN_N or ok[x].nunique() < 3:
        return out
    slope, intercept, _, _ = stats.theilslopes(ok[y], ok[x])
    out.loc[ok.index] = ok[y] - (intercept + slope * ok[x])
    return out


def _records(level, df, id_col, title_col, feature, z, values, method):
    recs = []
    for idx in z.index[z.abs() > Z_THRESHOLD]:
        direction = "unusually high" if z[idx] > 0 else "unusually low"
        recs.append({
            "level": level, "id": df.loc[idx, id_col], "title": df.loc[idx, title_col], "feature": feature,
            "direction": direction, "value": values[idx], "peer_median": float(values.median()),
            "modified_z": float(z[idx]), "method": method, "n_peers": int(values.notna().sum()),
            "note": f"{direction} {FEATURE_TEXT.get(feature, feature)} compared with its peers in this sample; "
                    f"worth investigating, not evidence of success or failure. {CONFOUNDERS}."})
    return recs


def detect(df, level, id_col, title_col, features, min_n=MIN_N):
    """Univariate robust-z anomalies on each feature plus the like-rate-vs-views residual."""
    df = df.reset_index(drop=True)
    recs, skipped = [], {}
    work = df.copy()
    if "like_rate" in work and "log_views" in work:
        work["like_rate_vs_views"] = robust_residual(work, "like_rate", "log_views")
        features = list(features) + ["like_rate_vs_views"]
    for f in features:
        if f not in work:
            skipped[f] = "feature not available"
            continue
        values = work[f]
        n = int(values.notna().sum())
        if n < min_n:
            skipped[f] = f"only {n} values (need {min_n})"
            continue
        z = modified_z(values)
        if z.isna().all():
            skipped[f] = "no spread (MAD = 0)"
            continue
        recs += _records(level, work, id_col, title_col, f, z.dropna(), values, "modified z-score")
    return recs, skipped


def isolation_forest(df, id_col, title_col, features, level="video"):
    """Multivariate check; only for samples large enough for it to be meaningful."""
    cols = [f for f in features if f in df]
    complete = df.dropna(subset=cols)
    if len(complete) < IF_MIN_N or len(cols) < 2:
        return [], f"skipped: {len(complete)} complete rows (need {IF_MIN_N})"
    X = StandardScaler().fit_transform(complete[cols].to_numpy(float))
    model = IsolationForest(contamination=IF_CONTAMINATION, random_state=SEED, n_estimators=300).fit(X)
    flag = model.predict(X) == -1
    score = -model.score_samples(X)
    recs = []
    for i in np.where(flag)[0]:
        row = complete.iloc[i]
        recs.append({"level": level, "id": row[id_col], "title": row[title_col],
                     "feature": "multivariate (" + ", ".join(cols) + ")", "direction": "unusual combination",
                     "value": float(score[i]), "peer_median": float(np.median(score)), "modified_z": np.nan,
                     "method": f"Isolation Forest (contamination {IF_CONTAMINATION:.0%})",
                     "n_peers": len(complete),
                     "note": "unusual combination of engagement features compared with the rest of the sample; "
                             f"worth investigating, not evidence of success or failure. {CONFOUNDERS}."})
    return recs, None


def run_anomaly_detection(channel_table, video_table):
    """Returns (anomalies DataFrame, diagnostics dict)."""
    diag = {}
    recs = []
    c_recs, c_skip = detect(channel_table, "channel", "channel_id", "channel_title",
                            ["like_rate", "comment_rate", "log_views", "sentiment_adjusted"])
    recs += c_recs
    diag["channel_skipped_features"] = c_skip
    diag["channel_n"] = len(channel_table)
    if video_table is not None and len(video_table):
        v_recs, v_skip = detect(video_table, "video", "video_id", "video_title",
                                ["like_rate", "comment_rate", "log_views_per_day"])
        recs += v_recs
        diag["video_skipped_features"] = v_skip
        if_recs, if_msg = isolation_forest(video_table, "video_id", "video_title",
                                           ["like_rate", "comment_rate", "log_views_per_day", "log_duration_min"])
        recs += if_recs
        diag["isolation_forest"] = if_msg or f"ran on {len(video_table)} videos"
        diag["video_n"] = len(video_table)
    cols = ["level", "id", "title", "feature", "direction", "value", "peer_median", "modified_z", "method",
            "n_peers", "note"]
    out = pd.DataFrame(recs, columns=cols)
    if len(out):
        out = out.sort_values(["level", "modified_z"], key=lambda s: s.abs() if s.name == "modified_z" else s,
                              ascending=[True, False]).reset_index(drop=True)
    return out, diag
