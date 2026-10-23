import re
import math
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SENTIMENT_WEIGHT = 0.65
ENGAGEMENT_WEIGHT = 0.25
VIEWS_WEIGHT = 0.10

# Larger values apply stronger confidence adjustment.
CONFIDENCE_STRENGTH = 20

analyzer = SentimentIntensityAnalyzer()


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def tokenize(text):
    """Convert text into lowercase alphanumeric tokens."""
    if not isinstance(text, str) or not text.strip():
        return set()

    return set(re.findall(r"[a-z0-9+#]+", text.lower()))


def calculate_topic_relevance(topic, text):
    """
    Estimate topic relevance using topic-word coverage.

    This is a simple lexical heuristic, not semantic similarity.
    """
    topic_words = tokenize(topic)
    text_words = tokenize(text)

    if not topic_words:
        return 0.0

    matched_words = topic_words.intersection(text_words)
    return len(matched_words) / len(topic_words)


# --------------------------------------------------
# COMMENT SENTIMENT ANALYSIS
# --------------------------------------------------

def analyze_comment_sentiment(comment):
    """Classify a comment using VADER sentiment analysis."""
    if not isinstance(comment, str) or not comment.strip():
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0
        }

    compound = analyzer.polarity_scores(comment)["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment": label,
        "sentiment_score": compound
    }


def analyze_comments(comments):
    """Add sentiment labels and numerical scores to comments."""
    if "comment_text" not in comments.columns:
        raise ValueError(
            "Comments dataset must contain 'comment_text'."
        )

    comments = comments.copy()

    results = comments["comment_text"].apply(
        analyze_comment_sentiment
    )

    comments["sentiment"] = results.apply(
        lambda result: result["sentiment"]
    )

    comments["sentiment_score"] = results.apply(
        lambda result: result["sentiment_score"]
    )

    return comments


# --------------------------------------------------
# CHANNEL SENTIMENT CALCULATION
# --------------------------------------------------

def calculate_channel_sentiment(group):
    """
    Calculate a channel's comment sentiment.

    Comment count adjusts confidence but is not directly
    added as a popularity bonus.
    """
    valid = group[
        group["comment_text"].apply(
            lambda value: isinstance(value, str)
            and bool(value.strip())
        )
    ]

    if valid.empty:
        return {
            "positive_percentage": 0.0,
            "negative_percentage": 0.0,
            "neutral_percentage": 0.0,
            "analyzed_comments": 0,
            "raw_sentiment": 0.0,
            "confidence_adjusted_sentiment": 0.0
        }

    positive = (valid["sentiment"] == "positive").sum()
    negative = (valid["sentiment"] == "negative").sum()
    neutral = (valid["sentiment"] == "neutral").sum()

    total = len(valid)

    positive_percentage = positive / total * 100
    negative_percentage = negative / total * 100
    neutral_percentage = neutral / total * 100

    # Average VADER compound score, ranging from -1 to +1.
    raw_sentiment = valid["sentiment_score"].mean()

    # Pull small samples toward neutral sentiment.
    confidence = total / (total + CONFIDENCE_STRENGTH)

    adjusted_sentiment = raw_sentiment * confidence

    return {
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        "neutral_percentage": neutral_percentage,
        "analyzed_comments": total,
        "raw_sentiment": raw_sentiment,
        "confidence_adjusted_sentiment": adjusted_sentiment
    }


# --------------------------------------------------
# METRIC NORMALIZATION
# --------------------------------------------------

def normalize_positive_series(series):
    """
    Normalize non-negative values to [0, 1] using the maximum.

    If every value is zero, return zero for all candidates.
    """
    values = pd.to_numeric(series, errors="coerce")
    values = values.fillna(0).clip(lower=0)

    maximum = values.max()

    if maximum <= 0:
        return pd.Series(0.0, index=series.index)

    return values / maximum


# --------------------------------------------------
# MAIN RECOMMENDER
# --------------------------------------------------

def recommend_channels(
    topic,
    video_metadata,
    channel_metadata,
    comments=None,
    top_n=10
):
    """
    Rank topic-relevant channels using:

        65% confidence-adjusted audience sentiment
        25% like-rate engagement
        10% logarithmically normalized views

    Topic relevance filters candidates but does not add
    a separate bonus to the final recommendation score.
    """

    required_video_columns = {
        "video_id",
        "channel_id",
        "channel_title",
        "video_title"
    }

    required_channel_columns = {
        "channel_id",
        "channel_title"
    }

    missing_videos = (
        required_video_columns - set(video_metadata.columns)
    )

    missing_channels = (
        required_channel_columns - set(channel_metadata.columns)
    )

    if missing_videos:
        raise ValueError(
            f"Video metadata is missing columns: {missing_videos}"
        )

    if missing_channels:
        raise ValueError(
            f"Channel metadata is missing columns: {missing_channels}"
        )

    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Topic must be a non-empty string.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    videos = video_metadata.copy()
    channels = channel_metadata.copy()

    # Remove records without channel identity.
    videos = videos.dropna(
        subset=["channel_id", "channel_title"]
    ).copy()

    # Ensure expected metrics exist and are numeric.
    for column in ["views", "likes"]:
        if column not in videos.columns:
            videos[column] = 0

        videos[column] = pd.to_numeric(
            videos[column], errors="coerce"
        ).fillna(0).clip(lower=0)

    # Find topic relevance for candidate filtering.
    videos["video_relevance"] = videos["video_title"].apply(
        lambda title: calculate_topic_relevance(topic, title)
    )

    # Aggregate the collected video metrics by channel.
    # Means prevent channels with more collected videos from
    # automatically receiving larger raw totals.
    ranked = (
        videos.groupby("channel_id")
        .agg(
            channel_title=("channel_title", "first"),
            topic_relevance_score=("video_relevance", "mean"),
            matching_videos=("video_id", "nunique"),
            views=("views", "mean"),
            likes=("likes", "mean")
        )
        .reset_index()
    )

    # Retain only topic-relevant candidates.
    ranked = ranked[
        ranked["topic_relevance_score"] > 0
    ].copy()

    if ranked.empty:
        return ranked

    # Merge channel descriptions and other metadata.
    channels = channels.drop_duplicates(
        subset=["channel_id"]
    )

    ranked = ranked.merge(
        channels,
        on="channel_id",
        how="left",
        suffixes=("", "_metadata")
    )

    # Prefer the channel title from channel metadata.
    if "channel_title_metadata" in ranked.columns:
        ranked["channel_title"] = (
            ranked["channel_title_metadata"]
            .fillna(ranked["channel_title"])
        )

    # Initialize sentiment metrics for channels with no comments.
    sentiment_defaults = {
        "positive_percentage": 0.0,
        "negative_percentage": 0.0,
        "neutral_percentage": 0.0,
        "analyzed_comments": 0,
        "raw_sentiment": 0.0,
        "confidence_adjusted_sentiment": 0.0
    }

    for column, default in sentiment_defaults.items():
        ranked[column] = default

    # Analyze comments and aggregate sentiment by channel.
    if comments is not None and not comments.empty:
        required_comment_columns = {
            "channel_id",
            "comment_text"
        }

        missing_comments = (
            required_comment_columns - set(comments.columns)
        )

        if missing_comments:
            raise ValueError(
                "Comments dataset is missing columns: "
                f"{missing_comments}"
            )

        analyzed = analyze_comments(comments)
        analyzed = analyzed.dropna(subset=["channel_id"])

        sentiment_rows = []

        for channel_id, group in analyzed.groupby("channel_id"):
            metrics = calculate_channel_sentiment(group)
            metrics["channel_id"] = channel_id
            sentiment_rows.append(metrics)

        if sentiment_rows:
            sentiment_df = pd.DataFrame(sentiment_rows)

            sentiment_columns = [
                "channel_id",
                "positive_percentage",
                "negative_percentage",
                "neutral_percentage",
                "analyzed_comments",
                "raw_sentiment",
                "confidence_adjusted_sentiment"
            ]

            ranked = ranked.drop(
                columns=[
                    column for column in sentiment_columns[1:]
                    if column in ranked.columns
                ]
            )

            ranked = ranked.merge(
                sentiment_df[sentiment_columns],
                on="channel_id",
                how="left"
            )

            for column in sentiment_columns[1:]:
                ranked[column] = ranked[column].fillna(
                    sentiment_defaults[column]
                )

    # Convert adjusted sentiment from [-1, +1] to [0, 1].
    ranked["sentiment_score"] = (
        ranked["confidence_adjusted_sentiment"] + 1
    ) / 2

    # Calculate like rate safely.
    ranked["like_rate"] = (
        ranked["likes"]
        .div(ranked["views"].where(ranked["views"] > 0))
        .fillna(0)
        .clip(lower=0, upper=1)
    )

    # Normalize engagement against the current candidates.
    ranked["engagement_score"] = normalize_positive_series(
        ranked["like_rate"]
    )

    # Log-transform views to reduce the effect of very large counts.
    ranked["log_views"] = ranked["views"].apply(
        lambda value: math.log1p(value)
    )

    # Normalize view scores to [0, 1].
    ranked["views_score"] = normalize_positive_series(
        ranked["log_views"]
    )

    # Calculate the final recommendation score.
    ranked["recommendation_score"] = (
        SENTIMENT_WEIGHT * ranked["sentiment_score"]
        + ENGAGEMENT_WEIGHT * ranked["engagement_score"]
        + VIEWS_WEIGHT * ranked["views_score"]
    )

    # Higher score ranks first.
    # Channel title provides a stable tie-breaker.
    ranked = ranked.sort_values(
        by=["recommendation_score", "channel_title"],
        ascending=[False, True]
    )

    return ranked.head(top_n).reset_index(drop=True)
