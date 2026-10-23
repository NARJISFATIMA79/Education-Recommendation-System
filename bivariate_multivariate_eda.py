
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMMENTS_FILE = DATA_DIR / "cleaned_comments.csv"
VIDEO_FILE = DATA_DIR / "video_metadata.csv"
CHANNEL_FILE = DATA_DIR / "channel_metadata.csv"


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def save_figure(filename):
    """Save the current figure without closing it."""
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )
    print(f"Graph saved: {filename}")


def find_column(df, possible_names):
    """Find a column using case-insensitive names."""
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def numeric_series(df, column):
    """Return a numeric version of a column."""
    return pd.to_numeric(df[column], errors="coerce")


def print_correlation(df, column_x, column_y, label):
    """Print Pearson correlation using valid numeric pairs."""
    pair = pd.DataFrame({
        "x": numeric_series(df, column_x),
        "y": numeric_series(df, column_y)
    }).dropna()

    if (
        len(pair) < 2
        or pair["x"].nunique() < 2
        or pair["y"].nunique() < 2
    ):
        print(
            f"{label}: correlation cannot be calculated "
            "because there are insufficient varying values."
        )
        return

    correlation = pair["x"].corr(pair["y"])
    print(f"{label}: {correlation:.4f} (n={len(pair)})")


def plot_scatter(
    df,
    x_column,
    y_column,
    title,
    xlabel,
    ylabel,
    filename
):
    """Plot the relationship between two numeric variables."""

    if x_column not in df.columns or y_column not in df.columns:
        print(f"Skipping {title}: required column not found.")
        return

    pair = pd.DataFrame({
        "x": numeric_series(df, x_column),
        "y": numeric_series(df, y_column)
    }).dropna()

    if (
        len(pair) < 2
        or pair["x"].nunique() < 2
        or pair["y"].nunique() < 2
    ):
        print(f"Skipping {title}: insufficient valid variation.")
        return

    plt.figure(figsize=(9, 5))
    plt.scatter(pair["x"], pair["y"], alpha=0.7)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)

    save_figure(filename)
    print_correlation(df, x_column, y_column, title)


def plot_correlation_matrix(df, columns, title, filename):
    """Plot and print the Pearson correlation matrix."""

    available = []

    for column in columns:
        if column is None or column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")

        if values.notna().sum() >= 2 and values.nunique() > 1:
            available.append(column)

    # Avoid duplicate columns in the matrix.
    available = list(dict.fromkeys(available))

    if len(available) < 2:
        print(f"Skipping {title}: fewer than two usable columns.")
        return

    numeric_data = df[available].apply(
        pd.to_numeric,
        errors="coerce"
    )

    correlation = numeric_data.corr()

    figure_size = max(8, len(available) * 1.1)

    plt.figure(figsize=(figure_size, figure_size * 0.8))

    image = plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(image, label="Pearson correlation")

    plt.xticks(
        range(len(available)),
        available,
        rotation=45,
        ha="right"
    )

    plt.yticks(range(len(available)), available)

    for row in range(len(available)):
        for col in range(len(available)):
            value = correlation.iloc[row, col]

            if pd.notna(value):
                plt.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

    plt.title(title)
    save_figure(filename)

    print(f"\n{title}:")
    print(correlation.round(3).to_string())


def standardize_id_column(df, possible_names, standard_name):
    """Rename a recognized ID column to a standard name."""
    actual_column = find_column(df, possible_names)

    if actual_column is None:
        return df, None

    if actual_column != standard_name:
        df = df.rename(columns={actual_column: standard_name})

    return df, standard_name


# ==================================================
# 3. LOAD DATASETS
# ==================================================

for file_path in [COMMENTS_FILE, VIDEO_FILE, CHANNEL_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n"
            "Run the extraction and cleaning scripts first."
        )

comments = pd.read_csv(COMMENTS_FILE)
videos = pd.read_csv(VIDEO_FILE)
channels = pd.read_csv(CHANNEL_FILE)

print("\n" + "=" * 70)
print("BIVARIATE AND MULTIVARIATE EDA")
print("=" * 70)

print("\nDataset shapes:")
print(f"Comments: {comments.shape}")
print(f"Videos:   {videos.shape}")
print(f"Channels: {channels.shape}")

print("\nChannel metadata columns:")
print(channels.columns.tolist())


# ==================================================
# 4. PREPARE COMMENT-LEVEL FEATURES
# ==================================================

if "comment_text" not in comments.columns:
    raise ValueError(
        "The comments dataset has no comment_text column."
    )

if "channel_title" not in comments.columns:
    raise ValueError(
        "The comments dataset has no channel_title column."
    )

comments["comment_length"] = (
    comments["comment_text"]
    .fillna("")
    .astype(str)
    .str.len()
)

if "like_count" in comments.columns:
    comments["like_count"] = pd.to_numeric(
        comments["like_count"],
        errors="coerce"
    )

print("\n" + "-" * 70)
print("PART A: COMMENT-LEVEL RELATIONSHIPS")
print("-" * 70)


# ==================================================
# 5. COMMENT LENGTH VS COMMENT LIKES
# ==================================================

if "like_count" in comments.columns:

    plot_scatter(
        comments,
        "comment_length",
        "like_count",
        "Comment Length vs Comment Likes",
        "Comment length (characters)",
        "Comment likes",
        "comment_length_vs_likes.png"
    )

    # This is a restricted visual view, not data deletion.
    low_like_comments = comments[
        comments["like_count"].between(0, 100)
    ].copy()

    plot_scatter(
        low_like_comments,
        "comment_length",
        "like_count",
        "Comment Length vs Likes (0–100 Likes)",
        "Comment length (characters)",
        "Comment likes",
        "comment_length_vs_likes_0_100.png"
    )


# ==================================================
# 6. COMMENT LIKES ACROSS CHANNELS
# ==================================================

if "like_count" in comments.columns:

    channel_likes = comments[
        ["channel_title", "like_count"]
    ].dropna()

    if not channel_likes.empty:

        channel_order = (
            channel_likes.groupby("channel_title")["like_count"]
            .median()
            .sort_values()
            .index.tolist()
        )

        box_data = [
            channel_likes.loc[
                channel_likes["channel_title"] == channel,
                "like_count"
            ].values
            for channel in channel_order
        ]

        plt.figure(figsize=(14, 7))

        plt.boxplot(
            box_data,
            tick_labels=channel_order,
            showfliers=False
        )

        plt.title("Comment Likes Across Channels")
        plt.xlabel("Channel")
        plt.ylabel("Comment likes")
        plt.xticks(rotation=60, ha="right")

        save_figure("comment_likes_by_channel.png")

        print(
            "Note: boxplot outlier markers are hidden for readability. "
            "No observations were removed from the CSV."
        )


# ==================================================
# 7. COMMENT LENGTHS ACROSS CHANNELS
# ==================================================

channel_lengths = comments[
    ["channel_title", "comment_length"]
].dropna()

if not channel_lengths.empty:

    channel_order = (
        channel_lengths.groupby("channel_title")["comment_length"]
        .median()
        .sort_values()
        .index.tolist()
    )

    box_data = [
        channel_lengths.loc[
            channel_lengths["channel_title"] == channel,
            "comment_length"
        ].values
        for channel in channel_order
    ]

    plt.figure(figsize=(14, 7))

    plt.boxplot(
        box_data,
        tick_labels=channel_order,
        showfliers=False
    )

    plt.title("Comment Lengths Across Channels")
    plt.xlabel("Channel")
    plt.ylabel("Comment length (characters)")
    plt.xticks(rotation=60, ha="right")

    save_figure("comment_length_by_channel.png")


# ==================================================
# 8. AGGREGATE COMMENTS BY VIDEO
# ==================================================

if "video_id" in comments.columns:

    aggregation = {
        "comment_length": ["mean", "median", "count"]
    }

    if "like_count" in comments.columns:
        aggregation["like_count"] = ["mean", "median"]

    video_comment_stats = (
        comments.groupby("video_id")
        .agg(aggregation)
    )

    video_comment_stats.columns = [
        "_".join(str(part) for part in column if part)
        for column in video_comment_stats.columns
    ]

    video_comment_stats = video_comment_stats.reset_index()

    video_comment_stats = video_comment_stats.rename(
        columns={
            "comment_length_mean": "avg_comment_length",
            "comment_length_median": "median_comment_length",
            "comment_length_count": "collected_comment_count",
            "like_count_mean": "avg_comment_likes",
            "like_count_median": "median_comment_likes"
        }
    )

else:
    video_comment_stats = pd.DataFrame()


# ==================================================
# 9. PREPARE VIDEO METADATA
# ==================================================

print("\n" + "-" * 70)
print("PART B: VIDEO-LEVEL RELATIONSHIPS")
print("-" * 70)

videos, video_id_col = standardize_id_column(
    videos,
    ["video_id", "videoid"],
    "video_id"
)

video_views_col = find_column(
    videos,
    ["views", "view_count", "video_views"]
)

video_likes_col = find_column(
    videos,
    ["likes", "like_count", "video_likes"]
)

video_comments_col = find_column(
    videos,
    ["comments", "comment_count", "video_comments"]
)

if (
    video_id_col is not None
    and not video_comment_stats.empty
):
    video_analysis = videos.merge(
        video_comment_stats,
        on="video_id",
        how="left",
        validate="one_to_one"
    )
else:
    video_analysis = videos.copy()

print("\nVideo analysis columns:")
print(video_analysis.columns.tolist())


# ==================================================
# 10. VIDEO-LEVEL SCATTER PLOTS
# ==================================================

if video_views_col and video_likes_col:
    plot_scatter(
        video_analysis,
        video_views_col,
        video_likes_col,
        "Video Views vs Video Likes",
        "Video views",
        "Video likes",
        "video_views_vs_likes.png"
    )

if video_views_col and video_comments_col:
    plot_scatter(
        video_analysis,
        video_views_col,
        video_comments_col,
        "Video Views vs Video Comment Count",
        "Video views",
        "Video comment count",
        "video_views_vs_comment_count.png"
    )

if video_likes_col and video_comments_col:
    plot_scatter(
        video_analysis,
        video_likes_col,
        video_comments_col,
        "Video Likes vs Video Comment Count",
        "Video likes",
        "Video comment count",
        "video_likes_vs_comment_count.png"
    )

if (
    video_views_col
    and "avg_comment_length" in video_analysis.columns
):
    plot_scatter(
        video_analysis,
        video_views_col,
        "avg_comment_length",
        "Video Views vs Average Collected Comment Length",
        "Video views",
        "Average comment length (characters)",
        "video_views_vs_avg_comment_length.png"
    )

if (
    video_views_col
    and "avg_comment_likes" in video_analysis.columns
):
    plot_scatter(
        video_analysis,
        video_views_col,
        "avg_comment_likes",
        "Video Views vs Average Likes on Collected Comments",
        "Video views",
        "Average likes per collected comment",
        "video_views_vs_avg_comment_likes.png"
    )


# ==================================================
# 11. VIDEO-LEVEL CORRELATION MATRIX
# ==================================================

print("\n" + "-" * 70)
print("PART C: MULTIVARIATE VIDEO ANALYSIS")
print("-" * 70)

video_numeric_candidates = [
    video_views_col,
    video_likes_col,
    video_comments_col,
    "avg_comment_length",
    "median_comment_length",
    "collected_comment_count",
    "avg_comment_likes",
    "median_comment_likes"
]

plot_correlation_matrix(
    video_analysis,
    video_numeric_candidates,
    "Video and Collected Comment Feature Correlations",
    "video_comment_correlation_matrix.png"
)


# ==================================================
# 12. CHANNEL-LEVEL ANALYSIS
# ==================================================

print("\n" + "-" * 70)
print("PART D: CHANNEL-LEVEL RELATIONSHIPS")
print("-" * 70)

channel_aggregation = {
    "comment_length": ["mean", "median", "count"]
}

if "like_count" in comments.columns:
    channel_aggregation["like_count"] = ["mean", "median"]

channel_comment_stats = (
    comments.groupby("channel_title")
    .agg(channel_aggregation)
)

channel_comment_stats.columns = [
    "_".join(str(part) for part in column if part)
    for column in channel_comment_stats.columns
]

channel_comment_stats = channel_comment_stats.reset_index()

channel_comment_stats = channel_comment_stats.rename(
    columns={
        "comment_length_mean": "avg_comment_length",
        "comment_length_median": "median_comment_length",
        "comment_length_count": "collected_comment_count",
        "like_count_mean": "avg_comment_likes",
        "like_count_median": "median_comment_likes"
    }
)

# Standardize the channel ID column if available.
channels, channel_id_col = standardize_id_column(
    channels,
    ["channel_id", "channelid"],
    "channel_id"
)

channel_name_col = find_column(
    channels,
    ["channel_title", "channel_name", "name"]
)

if channel_name_col and channel_name_col != "channel_title":
    channels = channels.rename(
        columns={channel_name_col: "channel_title"}
    )

# Build the channel analysis table robustly.
if (
    channel_id_col is not None
    and "channel_id" in comments.columns
):
    # Use ID as the reliable join key, then attach comment statistics
    # using a single canonical channel_title column.
    channel_names = (
        comments[["channel_id", "channel_title"]]
        .dropna(subset=["channel_id"])
        .drop_duplicates(subset=["channel_id"])
    )

    # Remove a metadata channel title if one exists so that the
    # canonical title from comments does not get suffixed.
    if "channel_title" in channels.columns:
        channels = channels.drop(columns=["channel_title"])

    channel_analysis = channels.merge(
        channel_names,
        on="channel_id",
        how="left",
        validate="one_to_one"
    )

    channel_analysis = channel_analysis.merge(
        channel_comment_stats,
        on="channel_title",
        how="left",
        validate="one_to_one"
    )

elif "channel_title" in channels.columns:
    # If IDs cannot be used, merge on the channel name.
    channel_analysis = channels.merge(
        channel_comment_stats,
        on="channel_title",
        how="left",
        validate="one_to_one"
    )

else:
    # Fallback: still analyze collected comment statistics by channel.
    print(
        "Warning: channel metadata has no recognized channel ID or "
        "channel name. Using comment-derived channel statistics only."
    )
    channel_analysis = channel_comment_stats.copy()

print("\nChannel analysis columns:")
print(channel_analysis.columns.tolist())

subscribers_col = find_column(
    channel_analysis,
    ["subscriber_count", "subscribers", "subscribers_count"]
)

channel_views_col = find_column(
    channel_analysis,
    ["total_views", "view_count", "channel_views"]
)

channel_video_count_col = find_column(
    channel_analysis,
    ["video_count", "videos_count", "total_videos"]
)

if (
    subscribers_col
    and "avg_comment_likes" in channel_analysis.columns
):
    plot_scatter(
        channel_analysis,
        subscribers_col,
        "avg_comment_likes",
        "Channel Subscribers vs Average Collected Comment Likes",
        "Channel subscribers",
        "Average likes per collected comment",
        "channel_subscribers_vs_avg_comment_likes.png"
    )

if (
    subscribers_col
    and "collected_comment_count" in channel_analysis.columns
):
    plot_scatter(
        channel_analysis,
        subscribers_col,
        "collected_comment_count",
        "Channel Subscribers vs Collected Comment Count",
        "Channel subscribers",
        "Number of collected comments",
        "channel_subscribers_vs_comment_count.png"
    )

channel_numeric_candidates = [
    subscribers_col,
    channel_views_col,
    channel_video_count_col,
    "avg_comment_length",
    "avg_comment_likes",
    "collected_comment_count"
]

plot_correlation_matrix(
    channel_analysis,
    channel_numeric_candidates,
    "Channel Metadata and Comment Feature Correlations",
    "channel_comment_correlation_matrix.png"
)


# ==================================================
# 13. FINISH
# ==================================================

print("\n" + "=" * 70)
print("BIVARIATE AND MULTIVARIATE EDA COMPLETE")
print("=" * 70)

print(f"Graphs saved in: {OUTPUT_DIR}")
print("CSV data files were not intentionally modified.")

print("\nOpening all generated graphs...")
plt.show()
