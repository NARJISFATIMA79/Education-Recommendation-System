
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw_comments.csv"


def main():
    print("=" * 65)
    print("RAW YOUTUBE COMMENTS: QUALITY AUDIT")
    print("=" * 65)

    if not DATA_PATH.exists():
        print(f"\nDataset not found: {DATA_PATH}")
        print("Run Option 2 in main.py to collect comments first.")
        return

    df = pd.read_csv(DATA_PATH)

    # 1. Dataset overview
    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    # 2. Missing values
    print("\n2. MISSING VALUES")
    print("-" * 40)
    print(df.isna().sum().to_string())
    print(f"Total missing cells: {df.isna().sum().sum()}")

    # 3. Duplicate checks
    print("\n3. DUPLICATE CHECKS")
    print("-" * 40)
    print(f"Complete duplicate rows: {df.duplicated().sum()}")

    if "comment_id" in df.columns:
        print(
            "Duplicate comment IDs: "
            f"{df['comment_id'].duplicated().sum()}"
        )

    # 4. Comment text quality
    if "comment_text" in df.columns:
        text = df["comment_text"].fillna("").astype(str)
        text_lengths = text.str.len()

        print("\n4. COMMENT TEXT QUALITY")
        print("-" * 40)
        print(
            "Empty or whitespace-only comments: "
            f"{text.str.strip().eq('').sum()}"
        )
        print(
            "Comments with leading/trailing whitespace: "
            f"{text.ne(text.str.strip()).sum()}"
        )
        print(
            f"Shortest comment: {text_lengths.min()} characters"
        )
        print(
            "Median comment length: "
            f"{text_lengths.median():.0f} characters"
        )
        print(
            "Average comment length: "
            f"{text_lengths.mean():.2f} characters"
        )
        print(
            f"Longest comment: {text_lengths.max()} characters"
        )

        print("\nFive shortest comments:")
        shortest_indices = text_lengths.nsmallest(5).index
        for comment in text.loc[shortest_indices]:
            print(f"- {comment[:200]!r}")

        print("\nFive longest comments:")
        longest_indices = text_lengths.nlargest(5).index
        for comment in text.loc[longest_indices]:
            print(f"- {comment[:200]!r}")

        # 5. Repeated comment text
        # Trim whitespace for comparison only; do not modify the dataset.
        normalized_text = text.str.strip()
        repeated_rows = normalized_text.duplicated(keep=False)

        print("\n5. REPEATED COMMENT TEXT")
        print("-" * 40)
        print(
            "Rows whose trimmed comment text appears more than once: "
            f"{repeated_rows.sum()}"
        )
        print(
            "Distinct repeated comment texts: "
            f"{normalized_text[repeated_rows].nunique()}"
        )

        print("\nExamples of repeated comment text:")
        examples = normalized_text[repeated_rows].value_counts().head(10)

        if examples.empty:
            print("No repeated comment text found.")
        else:
            for comment, count in examples.items():
                print(f"{count} occurrences: {comment[:200]!r}")

    # 6. Channel coverage
    if "channel_title" in df.columns:
        print("\n6. CHANNEL COVERAGE")
        print("-" * 40)
        print(f"Distinct channel names: {df['channel_title'].nunique()}")
        print("\nComments per channel:")
        print(df["channel_title"].value_counts().to_string())

    # 7. Video coverage
    if "video_id" in df.columns:
        print("\n7. VIDEO COVERAGE")
        print("-" * 40)
        print(f"Distinct videos: {df['video_id'].nunique()}")

    # 8. Comment like counts
    if "like_count" in df.columns:
        likes = pd.to_numeric(df["like_count"], errors="coerce")

        print("\n8. COMMENT LIKE COUNTS")
        print("-" * 40)
        print(f"Invalid or missing numeric values: {likes.isna().sum()}")

        if likes.notna().any():
            print(likes.describe().to_string())
            print(f"Negative like counts: {(likes < 0).sum()}")
        else:
            print("No valid numeric like counts to analyse.")

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE - NO DATA WAS MODIFIED")
    print("=" * 65)


if __name__ == "__main__":
    main()
