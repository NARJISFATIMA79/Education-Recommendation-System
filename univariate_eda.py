
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. File paths and output folder
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMMENTS_FILE = DATA_DIR / "cleaned_comments.csv"
VIDEO_FILE = DATA_DIR / "video_metadata.csv"
CHANNEL_FILE = DATA_DIR / "channel_metadata.csv"


# --------------------------------------------------
# 2. Helper function to save and prepare each graph
# --------------------------------------------------

def save_figure(filename):
    """Save the current graph without closing it."""
    plt.tight_layout()
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graph saved: {output_path.name}")


# --------------------------------------------------
# 3. Load and inspect cleaned comments
# --------------------------------------------------

if not COMMENTS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {COMMENTS_FILE}. "
        "Run clean_comments.py first."
    )

comments = pd.read_csv(COMMENTS_FILE)

print("\n" + "=" * 65)
print("UNIVARIATE EDA: EDUCATIONAL YOUTUBE COMMENTS")
print("=" * 65)

print("\nDataset shape:")
print(f"Rows: {comments.shape[0]}")
print(f"Columns: {comments.shape[1]}")

print("\nColumn names:")
print(comments.columns.tolist())

print("\nData types:")
print(comments.dtypes)

print("\nMissing values:")
print(comments.isnull().sum().to_string())

print("\nComplete duplicate rows:")
print(comments.duplicated().sum())


# --------------------------------------------------
# 4. Analyze comment lengths
# --------------------------------------------------

if "comment_text" in comments.columns:

    comments["comment_text"] = (
        comments["comment_text"].fillna("").astype(str)
    )

    comments["comment_length"] = comments["comment_text"].str.len()

    print("\n" + "-" * 65)
    print("COMMENT LENGTH ANALYSIS")
    print("-" * 65)

    print(comments["comment_length"].describe())

    print(
        "\nEmpty or whitespace-only comments:",
        comments["comment_text"].str.strip().eq("").sum()
    )

    print("\nFive shortest comments:")
    print(
        comments.nsmallest(5, "comment_length")[
            ["comment_text", "comment_length"]
        ].to_string(index=False)
    )

    print("\nFive longest comments:")
    print(
        comments.nlargest(5, "comment_length")[
            ["comment_text", "comment_length"]
        ].to_string(index=False)
    )

    # Graph 1: Distribution of comment lengths
    plt.figure(figsize=(10, 5))
    plt.hist(
        comments["comment_length"],
        bins=30,
        edgecolor="black"
    )
    plt.title("Distribution of YouTube Comment Lengths")
    plt.xlabel("Comment length (characters)")
    plt.ylabel("Number of comments")
    save_figure("comment_length_distribution.png")


# --------------------------------------------------
# 5. Analyze comment likes
# --------------------------------------------------

if "like_count" in comments.columns:

    comments["like_count"] = pd.to_numeric(
        comments["like_count"],
        errors="coerce"
    )

    valid_likes = comments["like_count"].dropna()

    print("\n" + "-" * 65)
    print("COMMENT LIKE-COUNT ANALYSIS")
    print("-" * 65)

    print(valid_likes.describe())

    print(
        "\nComments with zero likes:",
        comments["like_count"].eq(0).sum()
    )

    print(
        "Missing or invalid like counts:",
        comments["like_count"].isna().sum()
    )

    print(
        "Negative like counts:",
        comments["like_count"].lt(0).sum()
    )

    # Graph 2: Full distribution, including extreme values
    plt.figure(figsize=(10, 5))
    plt.hist(
        valid_likes,
        bins=30,
        edgecolor="black"
    )
    plt.title("Distribution of Comment Likes")
    plt.xlabel("Number of likes")
    plt.ylabel("Number of comments")
    save_figure("comment_likes_distribution.png")

    # Graph 3: Focus on comments with 0–100 likes
    low_likes = valid_likes[valid_likes.between(0, 100)]

    plt.figure(figsize=(10, 5))
    plt.hist(
        low_likes,
        bins=25,
        edgecolor="black"
    )
    plt.title("Comment Likes: 0 to 100")
    plt.xlabel("Number of likes")
    plt.ylabel("Number of comments")
    save_figure("comment_likes_0_to_100.png")

    print(
        "\nComments included in the 0–100 likes graph:",
        len(low_likes)
    )
    print(
        "Comments excluded from that graph because they exceed 100 likes:",
        (valid_likes > 100).sum()
    )


# --------------------------------------------------
# 6. Analyze comment publication dates
# --------------------------------------------------

if "comment_published_at" in comments.columns:

    comments["comment_published_at"] = pd.to_datetime(
        comments["comment_published_at"],
        errors="coerce",
        utc=True
    )

    valid_dates = comments.dropna(
        subset=["comment_published_at"]
    ).copy()

    print("\n" + "-" * 65)
    print("COMMENT DATE ANALYSIS")
    print("-" * 65)

    print(
        "Invalid or missing publication dates:",
        comments["comment_published_at"].isna().sum()
    )

    print(
        "Earliest comment:",
        comments["comment_published_at"].min()
    )

    print(
        "Latest comment:",
        comments["comment_published_at"].max()
    )

    monthly_comments = (
        valid_dates
        .assign(
            month=valid_dates["comment_published_at"].dt.strftime("%Y-%m")
        )
        .groupby("month")
        .size()
        .sort_index()
    )

    if not monthly_comments.empty:
        plt.figure(figsize=(13, 5))
        monthly_comments.plot(kind="bar")
        plt.title("Collected Comments by Month")
        plt.xlabel("Month")
        plt.ylabel("Number of collected comments")
        plt.xticks(rotation=60, ha="right")
        save_figure("comments_by_month.png")


# --------------------------------------------------
# 7. Analyze channel coverage
# --------------------------------------------------

if "channel_title" in comments.columns:

    channel_counts = comments["channel_title"].value_counts()

    print("\n" + "-" * 65)
    print("COMMENTS PER CHANNEL")
    print("-" * 65)

    print(channel_counts.to_string())

    print(
        "\nNumber of distinct channels:",
        comments["channel_title"].nunique()
    )

    plt.figure(figsize=(12, 7))
    channel_counts.sort_values().plot(kind="barh")
    plt.title("Collected Comments per Channel")
    plt.xlabel("Number of collected comments")
    plt.ylabel("Channel")
    save_figure("comments_per_channel.png")


# --------------------------------------------------
# 8. Inspect video and channel metadata
# --------------------------------------------------

def inspect_metadata(file_path, dataset_name, file_prefix):

    if not file_path.exists():
        print(
            f"\n{dataset_name} file not found: {file_path.name}"
        )
        return

    df = pd.read_csv(file_path)

    print("\n" + "-" * 65)
    print(f"{dataset_name.upper()} DATASET")
    print("-" * 65)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:
        print("\nNo numeric columns available for histograms.")
        return

    print("\nNumeric summary:")
    print(df[numeric_columns].describe().to_string())

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if values.empty or values.nunique() <= 1:
            print(
                f"Skipping histogram for {column}: "
                "no variation or no valid values."
            )
            continue

        # Numeric metadata histogram
        plt.figure(figsize=(9, 5))
        plt.hist(
            values,
            bins=15,
            edgecolor="black"
        )
        plt.title(
            f"{dataset_name}: Distribution of {column}"
        )
        plt.xlabel(column)
        plt.ylabel("Number of records")

        safe_column = "".join(
            char if char.isalnum() else "_"
            for char in column
        )

        save_figure(f"{file_prefix}_{safe_column}.png")


inspect_metadata(
    VIDEO_FILE,
    "Video metadata",
    "video"
)

inspect_metadata(
    CHANNEL_FILE,
    "Channel metadata",
    "channel"
)


# --------------------------------------------------
# 9. Finish and display all graphs
# --------------------------------------------------

print("\n" + "=" * 65)
print("UNIVARIATE EDA COMPLETE")
print("=" * 65)

print(f"All graphs saved in: {OUTPUT_DIR}")
print("The CSV data files were not intentionally modified.")

print("\nOpening all univariate analysis graphs...")
plt.show()
