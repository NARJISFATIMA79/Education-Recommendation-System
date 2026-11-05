
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport


# Build paths relative to this file, not the terminal's current directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

DATA_PATH = DATA_DIR / "raw_comments.csv"
REPORT_PATH = DATA_DIR / "raw_comments_profile.html"


def main():
    """Analyze the raw comments CSV and generate its HTML report."""

    print("\n" + "=" * 60)
    print("RAW YOUTUBE COMMENTS: DATA PROFILING")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Collect comments using Option 2 in main.py first.")
        return

    df = pd.read_csv(DATA_PATH, engine="python")

    if df.empty:
        print("The raw CSV exists, but contains no comment records.")
        return

    print(f"\nDataset shape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nMissing-value percentages:")
    print((df.isna().mean() * 100).round(2).to_string())

    print("\nDuplicate complete rows:")
    print(df.duplicated().sum())

    if "comment_id" in df.columns:
        print("\nDuplicate comment IDs:")
        print(df["comment_id"].duplicated().sum())

    if "comment_text" in df.columns:
        comment_text = df["comment_text"].fillna("").astype(str)

        print("\nComment text statistics:")
        print(
            "Empty or whitespace-only comments: "
            f"{comment_text.str.strip().eq('').sum()}"
        )
        print(
            "Average comment length: "
            f"{comment_text.str.len().mean():.2f} characters"
        )
        print(
            "Shortest comment length: "
            f"{comment_text.str.len().min()} characters"
        )
        print(
            "Longest comment length: "
            f"{comment_text.str.len().max()} characters"
        )

    if "channel_id" in df.columns:
        print("\nDistinct channels in comments:")
        print(df["channel_id"].nunique())

    if "video_id" in df.columns:
        print("\nDistinct videos in comments:")
        print(df["video_id"].nunique())

    print("\nGenerating the raw profiling report...")

    profile = ProfileReport(
        df,
        title="Raw YouTube Comments - Data Profiling Report",
        explorative=True,
        minimal=True,
    )

    # Create the output directory if it does not exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # A successful run replaces the previous report at this path.
    profile.to_file(str(REPORT_PATH))

    print("\n" + "=" * 60)
    print("RAW PROFILING COMPLETED")
    print("=" * 60)
    print(f"Report saved to: {REPORT_PATH}")
    print("The original raw CSV has not been modified.")


if __name__ == "__main__":
    main()
