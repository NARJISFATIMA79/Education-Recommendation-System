
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "raw_comments.csv"
CLEAN_FILE = DATA_DIR / "cleaned_comments.csv"


def clean_comments(raw_file=RAW_FILE, clean_file=CLEAN_FILE):
    """Validate and clean comments, then save the cleaned dataset."""

    raw_file = Path(raw_file)
    clean_file = Path(clean_file)

    # 1. Check whether the raw dataset exists.
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_file}\n"
            "Please run the comment extraction program first."
        )

    # 2. Load the original dataset.
    df = pd.read_csv(
    raw_file,
    engine="python"
)
    original_rows = len(df)

    print("=" * 60)
    print("EDUCATIONAL YOUTUBE RECOMMENDATION SYSTEM")
    print("DATA CLEANING REPORT")
    print("=" * 60)

    print(f"\nOriginal rows: {original_rows}")
    print(f"Original columns: {len(df.columns)}")

    # 3. Check required columns.
    required_columns = [
        "comment_id",
        "video_id",
        "channel_id",
        "comment_text",
        "comment_published_at",
        "comment_updated_at",
        "like_count",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    # 4. Work on a copy so raw data remains unchanged.
    clean_df = df.copy()

    # 5. Remove surrounding whitespace.
    text_before = (
        clean_df["comment_text"].fillna("").astype(str)
    )

    clean_df["comment_text"] = (
        clean_df["comment_text"]
        .astype("string")
        .str.strip()
    )

    whitespace_changes = (
        text_before
        != clean_df["comment_text"].fillna("").astype(str)
    ).sum()

    print(
        "\nComments changed by whitespace trimming:",
        int(whitespace_changes),
    )

    # 6. Identify empty comments without deleting them.
    empty_comments = (
        clean_df["comment_text"].isna()
        | clean_df["comment_text"].str.strip().eq("")
    )

    print("Empty comments found:", int(empty_comments.sum()))

    # 7. Validate like counts.
    clean_df["like_count"] = pd.to_numeric(
        clean_df["like_count"],
        errors="coerce",
    )

    invalid_likes = clean_df["like_count"].isna()
    negative_likes = clean_df["like_count"] < 0

    print(
        "Missing or invalid like counts:",
        int(invalid_likes.sum()),
    )
    print(
        "Negative like counts:",
        int(negative_likes.sum()),
    )

    # 8. Validate timestamps and standardize them to UTC.
    date_columns = [
        "comment_published_at",
        "comment_updated_at",
    ]

    for column in date_columns:
        parsed_dates = pd.to_datetime(
            clean_df[column],
            utc=True,
            errors="coerce",
        )

        invalid_dates = (
            clean_df[column].notna()
            & parsed_dates.isna()
        )

        print(
            f"Invalid {column} values:",
            int(invalid_dates.sum()),
        )

        clean_df[column] = parsed_dates

    # 9. Report duplicate records.
    duplicate_ids = clean_df["comment_id"].duplicated().sum()
    duplicate_rows = clean_df.duplicated().sum()

    print("\nDuplicate comment IDs:", int(duplicate_ids))
    print("Completely duplicated rows:", int(duplicate_rows))

    # 10. Save the cleaned dataset.
    clean_file.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_file, index=False)

    print("\n" + "=" * 60)
    print("DATA CLEANING SUMMARY")
    print("=" * 60)

    print(f"Rows before cleaning: {original_rows}")
    print(f"Rows after cleaning:  {len(clean_df)}")
    print(f"Rows removed:         {original_rows - len(clean_df)}")
    print(f"Columns retained:     {len(clean_df.columns)}")
    print(f"\nCleaned file saved to: {clean_file}")
    print("\nRaw data was not modified.")
    print("CLEANING STAGE COMPLETE")

    return clean_df


if __name__ == "__main__":
    clean_comments()
