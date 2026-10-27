
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "cleaned_comments.csv"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "comment_text",
    "channel_title",
    "video_title",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["comment_text"] = (
    df["comment_text"]
    .fillna("")
    .astype(str)
)

df["channel_title"] = (
    df["channel_title"]
    .fillna("Unknown channel")
    .astype(str)
)

df["video_title"] = (
    df["video_title"]
    .fillna("Unknown video")
    .astype(str)
)


# --------------------------------------------------
# 3. Basic textual overview
# --------------------------------------------------

print("\n" + "=" * 65)
print("TEXTUAL ANALYSIS: EDUCATIONAL YOUTUBE COMMENTS")
print("=" * 65)

print("\nTotal comments:", len(df))
print("Distinct channels:", df["channel_title"].nunique())
print("Distinct videos:", df["video_id"].nunique()
      if "video_id" in df.columns else "Not available")

df["comment_length"] = df["comment_text"].str.len()

print("\nComment length statistics:")
print(df["comment_length"].describe())

print("\nFive longest comments:")
print(
    df.nlargest(5, "comment_length")[
        ["channel_title", "comment_text", "comment_length"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 4. Basic text preprocessing for word analysis
# --------------------------------------------------

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then",
    "is", "am", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "for", "with",
    "at", "by", "from", "as", "it", "its", "this",
    "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them",
    "their", "his", "her", "what", "which", "who",
    "when", "where", "why", "how", "can", "could",
    "would", "should", "will", "just", "not", "do",
    "does", "did", "have", "has", "had", "so", "very",
    "too", "about", "into", "than", "also", "there",
    "here", "all", "more", "some", "any", "been",
    "thanks", "thank", "video", "videos", "channel",
    "watch", "watched", "watching", "sir", "please",
    "really", "one", "get", "got", "make", "made",
    "like", "good", "great", "amazing", "love"
}


def tokenize(text):
    """Extract simple English word tokens."""
    words = re.findall(
        r"[a-zA-Z][a-zA-Z']+",
        text.lower()
    )

    return [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]


all_words = []

for comment in df["comment_text"]:
    all_words.extend(tokenize(comment))

word_counts = Counter(all_words)

print("\n" + "-" * 65)
print("MOST FREQUENT WORDS")
print("-" * 65)

print(pd.DataFrame(
    word_counts.most_common(30),
    columns=["word", "frequency"]
).to_string(index=False))

top_words = pd.DataFrame(
    word_counts.most_common(20),
    columns=["word", "frequency"]
)

if not top_words.empty:
    top_words.to_csv(
        OUTPUT_DIR / "top_comment_words.csv",
        index=False
    )

    plt.figure(figsize=(11, 7))
    plt.barh(
        top_words["word"].iloc[::-1],
        top_words["frequency"].iloc[::-1]
    )
    plt.title("Most Frequent Words in YouTube Comments")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "top_comment_words.png",
        dpi=150
    )
    plt.close()


# --------------------------------------------------
# 5. Transparent rule-based comment categories
# --------------------------------------------------

# These are preliminary signals, not verified labels.
# A comment may belong to multiple categories.

CATEGORY_PATTERNS = {
    "appreciation": [
        r"\bthank you\b",
        r"\bthanks\b",
        r"\bwell explained\b",
        r"\bclearly explained\b",
        r"\bgreat explanation\b",
        r"\bhelpful\b",
        r"\bappreciate\b",
        r"\bexcellent explanation\b",
    ],

    "learning_question": [
        r"\bcan anyone explain\b",
        r"\bcan you explain\b",
        r"\bwhy does\b",
        r"\bwhy do we\b",
        r"\bhow do i\b",
        r"\bhow can i\b",
        r"\bwhat is the difference\b",
        r"\bwhat's the difference\b",
        r"\bplease explain\b",
        r"\bquestion\b",
        r"\bconfused\b",
        r"\bi don't understand\b",
        r"\bi do not understand\b",
    ],

    "learning_guidance": [
        r"\broadmap\b",
        r"\bwhere should i start\b",
        r"\bhow to start\b",
        r"\bwhat should i learn\b",
        r"\bwhich language\b",
        r"\bbeginner\b",
        r"\blearning path\b",
        r"\bwhat to learn first\b",
        r"\bresources\b",
    ],

    "possible_promotion": [
        r"https?://",
        r"\bdiscount code\b",
        r"\benroll now\b",
        r"\bbuy now\b",
        r"\bjoin my course\b",
        r"\bsubscribe to my channel\b",
        r"\buse my code\b",
    ],

    "possible_negative_feedback": [
        r"\bnot clear\b",
        r"\bpoor explanation\b",
        r"\bdoesn't explain\b",
        r"\bdoes not explain\b",
        r"\bwrong explanation\b",
        r"\btoo fast\b",
        r"\bi'm confused\b",
        r"\bi am confused\b",
    ],
}


for category, patterns in CATEGORY_PATTERNS.items():
    combined_pattern = "|".join(patterns)

    df[category] = df["comment_text"].str.contains(
        combined_pattern,
        case=False,
        regex=True,
        na=False
    )


print("\n" + "-" * 65)
print("PRELIMINARY COMMENT CATEGORY COUNTS")
print("-" * 65)

category_columns = list(CATEGORY_PATTERNS.keys())

category_summary = pd.DataFrame({
    "category": category_columns,
    "matching_comments": [
        int(df[col].sum())
        for col in category_columns
    ],
})

category_summary["percentage_of_all_comments"] = (
    category_summary["matching_comments"]
    / len(df) * 100
    if len(df) else 0
)

print(category_summary.to_string(index=False))

category_summary.to_csv(
    OUTPUT_DIR / "comment_category_summary.csv",
    index=False
)


# --------------------------------------------------
# 6. Save comments for manual review
# --------------------------------------------------

# Keep the original comment text and add category flags.
# No comments are deleted.

review_columns = [
    col for col in [
        "comment_id",
        "channel_title",
        "video_title",
        "comment_text",
        "like_count",
    ]
    if col in df.columns
]

review_columns += category_columns

df[review_columns].to_csv(
    OUTPUT_DIR / "comment_category_review.csv",
    index=False
)

print("\nCategory examples:")
for category in category_columns:
    matches = df[df[category]]

    print(f"\n{category.upper()}")

    if matches.empty:
        print("No matches found.")
    else:
        for _, row in matches.head(3).iterrows():
            print("-", row["comment_text"][:350])


# --------------------------------------------------
# 7. Category counts by channel
# --------------------------------------------------

channel_summary = (
    df.groupby("channel_title")[category_columns]
    .sum()
    .sort_values(
        by="learning_question",
        ascending=False
    )
)

print("\n" + "-" * 65)
print("PRELIMINARY CATEGORY COUNTS BY CHANNEL")
print("-" * 65)

print(channel_summary.to_string())

channel_summary.to_csv(
    OUTPUT_DIR / "comment_categories_by_channel.csv"
)

channel_summary.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Rule-Based Comment Signals by Channel")
plt.xlabel("Channel")
plt.ylabel("Number of matching comments")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "comment_categories_by_channel.png",
    dpi=150
)
plt.close()


# --------------------------------------------------
# 8. Export a full annotated copy for investigation
# --------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "comments_with_text_signals.csv",
    index=False
)

print("\n" + "=" * 65)
print("TEXTUAL ANALYSIS COMPLETE")
print("=" * 65)

print(f"Outputs saved in: {OUTPUT_DIR}")
print("Original raw and cleaned CSV files were not overwritten.")
print(
    "Reminder: rule-based matches are preliminary signals, "
    "not verified sentiment or learning-quality labels."
)
