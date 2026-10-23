
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport


# Build paths relative to this file.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

RAW_FILE = DATA_DIR / "raw_comments.csv"
CLEAN_FILE = DATA_DIR / "cleaned_comments.csv"
REPORT_FILE = DATA_DIR / "cleaned_comments_profile.html"


def main():
    """Compare raw and cleaned data, then generate the cleaned report."""

    print("\n" + "=" * 60)
    print("RAW VS CLEANED DATASET")
    print("=" * 60)

    if not RAW_FILE.exists():
        print(f"Raw dataset not found: {RAW_FILE}")
        print("Collect comments using Option 2 in main.py first.")
        return

    if not CLEAN_FILE.exists():
        print(f"Cleaned dataset not found: {CLEAN_FILE}")
        print("Run the comment-cleaning step first.")
        return

    raw_df = pd.read_csv(RAW_FILE, engine="python")
    clean_df = pd.read_csv(CLEAN_FILE, engine="python")

    print(f"Raw dataset rows:        {len(raw_df):,}")
    print(f"Cleaned dataset rows:    {len(clean_df):,}")
    print(f"Raw dataset columns:     {len(raw_df.columns)}")
    print(f"Cleaned dataset columns: {len(clean_df.columns)}")

    print(
        "\nColumn names unchanged: "
        f"{list(raw_df.columns) == list(clean_df.columns)}"
    )

    print("\nMissing values in cleaned dataset:")
    print(clean_df.isna().sum().to_string())

    if clean_df.empty:
        print("\nThe cleaned CSV contains no rows. No report was generated.")
        return

    print("\nGenerating the cleaned-data profiling report...")

    report = ProfileReport(
        clean_df,
        title="Cleaned YouTube Comments - Data Profiling Report",
        explorative=True,
        minimal=True,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # A successful run replaces the previous report at this path.
    report.to_file(str(REPORT_FILE))

    print("\n" + "=" * 60)
    print("CLEANED PROFILING COMPLETED")
    print("=" * 60)
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
