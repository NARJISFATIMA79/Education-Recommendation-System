
from pathlib import Path

import pandas as pd

from comment_extractor import extract_comments_for_topic
from youtube_client import search_educational_videos
from clean_comments import clean_comments
from recommender import recommend_channels

# Import the report functions without running them during import.
from profile_comments import main as generate_raw_profile
from profile_cleaned_comments import main as generate_cleaned_profile


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

VIDEO_METADATA_FILE = DATA_DIR / "video_metadata.csv"
CHANNEL_METADATA_FILE = DATA_DIR / "channel_metadata.csv"


def format_count(value):
    """Format numeric counts for readable display."""
    if value is None or value == "":
        return "Not available"

    try:
        if pd.isna(value):
            return "Not available"

        return f"{int(float(value)):,}"

    except (ValueError, TypeError, OverflowError):
        return str(value)


def format_score(value, decimals=3):
    """Format a numeric score safely."""
    try:
        if pd.isna(value):
            return "Not available"

        return f"{float(value):.{decimals}f}"

    except (ValueError, TypeError, OverflowError):
        return "Not available"


def format_percentage(value):
    """Format a percentage safely."""
    try:
        if pd.isna(value):
            return "Not available"

        return f"{float(value):.1f}%"

    except (ValueError, TypeError, OverflowError):
        return "Not available"


def display_videos(videos):
    """Display educational video search results."""

    print("\n" + "=" * 70)
    print("EDUCATIONAL VIDEO SEARCH RESULTS")
    print("=" * 70)

    if videos is None or len(videos) == 0:
        print("No videos found for this topic.")
        return

    for number, video in enumerate(videos, start=1):
        print("\n" + "=" * 70)
        print(f"{number}. {video.get('title', 'Unknown title')}")
        print("=" * 70)

        print(f"Channel: {video.get('channel_title', 'Unknown')}")
        print(f"Subscribers: {format_count(video.get('subscribers'))}")
        print(f"Published: {video.get('published_at', 'Unknown')}")
        print(f"Duration: {video.get('duration', 'Unknown')}")
        print(f"Views: {format_count(video.get('views'))}")
        print(f"Likes: {format_count(video.get('likes'))}")
        print(f"Comments: {format_count(video.get('comments'))}")
        print(f"URL: {video.get('url', 'Not available')}")

        print("\nLANGUAGE METADATA")

        print(
            "Default video language: "
            f"{video.get('default_video_language') or 'Unknown'}"
        )

        print(
            "Default audio language: "
            f"{video.get('default_audio_language') or 'Unknown'}"
        )


def display_recommendations(recommendations):
    """Display ranked educational channels and their score components."""

    print("\n" + "=" * 70)
    print("RECOMMENDED EDUCATIONAL CHANNELS")
    print("=" * 70)

    if recommendations is None or recommendations.empty:
        print("No matching channels were found.")
        print("Try a different or more specific educational topic.")
        return

    for index, (_, channel) in enumerate(
        recommendations.iterrows(),
        start=1,
    ):
        channel_title = channel.get(
            "channel_title",
            "Unknown channel",
        )

        print("\n" + "=" * 60)
        print(f"{index}. {channel_title}")
        print("=" * 60)

        print(
            "Recommendation score: "
            f"{format_score(channel.get('recommendation_score'))}"
        )

        print(
            "Topic relevance: "
            f"{format_score(channel.get('topic_relevance_score'))}"
        )

        print("\nSCORE COMPONENTS")

        print(
            "Sentiment score: "
            f"{format_score(channel.get('sentiment_score'))}"
        )

        print(
            "Engagement score: "
            f"{format_score(channel.get('engagement_score'))}"
        )

        print(
            "Views score: "
            f"{format_score(channel.get('views_score'))}"
        )

        print("\nCOMMENT SENTIMENT ANALYSIS")

        print(
            "Positive comments: "
            f"{format_percentage(channel.get('positive_percentage'))}"
        )

        print(
            "Negative comments: "
            f"{format_percentage(channel.get('negative_percentage'))}"
        )

        print(
            "Neutral comments: "
            f"{format_percentage(channel.get('neutral_percentage'))}"
        )

        print(
            "Comments analyzed: "
            f"{format_count(channel.get('analyzed_comments'))}"
        )

        print("\nCHANNEL STATISTICS")

        print(
            "Matching videos: "
            f"{format_count(channel.get('matching_videos'))}"
        )

        print(
            "Average video views: "
            f"{format_count(channel.get('views'))}"
        )

        print(
            "Average video likes: "
            f"{format_count(channel.get('likes'))}"
        )

        print(
            "Subscribers: "
            f"{format_count(channel.get('subscribers'))}"
        )

        description = channel.get("channel_description", "")

        if isinstance(description, str) and description.strip():
            print("\nDescription:")

            description = " ".join(description.split())
            print(description[:300])

            if len(description) > 300:
                print("...")


def load_metadata(file_path, description):
    """Load a metadata CSV file after checking that it exists."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        engine="python",
    )

    if dataframe.empty:
        print(f"No {description.lower()} records were collected.")

    return dataframe


def run_video_search(topic):
    """Search YouTube and display educational videos."""

    print(f"\nSearching for educational videos about: {topic}")

    videos = search_educational_videos(
        topic,
        max_results=10,
    )

    display_videos(videos)


def run_recommendation(topic):
    """Collect, clean, profile, analyze, and rank the data."""

    print("\n" + "=" * 70)
    print("EDUCATIONAL CHANNEL RECOMMENDATION")
    print("=" * 70)

    print(f"Topic: {topic}")
    print("Target: 15 distinct channels")
    print("Target: Up to 100 top-level comments per selected video")
    print("Collecting current data. This may take some time.")

    # Step 1: Collect fresh comments and metadata.
    extract_comments_for_topic(
        topic=topic,
        target_channels=15,
        max_comments_per_video=100,
    )

    # Step 2: Generate a profile of the raw comments.
    print("\nGenerating raw comments profiling report...")
    generate_raw_profile()

    # Step 3: Clean the collected comments.
    print("\nCleaning collected comments...")

    cleaned_comments = clean_comments()

    if cleaned_comments is None:
        raise RuntimeError(
            "The comment-cleaning function did not return a dataset."
        )

    # Step 4: Profile the cleaned CSV written by clean_comments().
    print("\nGenerating cleaned comments profiling report...")
    generate_cleaned_profile()

    if cleaned_comments.empty:
        print(
            "No comments are available for sentiment analysis. "
            "Recommendations cannot be calculated reliably."
        )
        return

    # Step 5: Load the metadata produced by the extractor.
    print("\nLoading video and channel metadata...")

    videos = load_metadata(
        VIDEO_METADATA_FILE,
        "Video metadata",
    )

    channels = load_metadata(
        CHANNEL_METADATA_FILE,
        "Channel metadata",
    )

    if videos.empty or channels.empty:
        print(
            "Insufficient metadata was collected to recommend channels."
        )
        return

    # Step 6: Calculate recommendation scores for the selected topic.
    print("\nAnalyzing comments and calculating recommendation scores...")

    recommendations = recommend_channels(
        topic=topic,
        video_metadata=videos,
        channel_metadata=channels,
        comments=cleaned_comments,
        top_n=10,
    )

    # Step 7: Display ranked results.
    display_recommendations(recommendations)


def main():
    """Display the menu and execute the selected operation."""

    print("\n" + "=" * 60)
    print("EDUCATIONAL YOUTUBE RECOMMENDATION SYSTEM")
    print("=" * 60)

    print("1. Find Educational Videos")
    print("2. Recommend Educational Channels")

    choice = input("\nChoose an option (1 or 2): ").strip()

    if choice not in ("1", "2"):
        print("Invalid option. Please choose 1 or 2.")
        return

    topic = input("\nEnter an educational topic: ").strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    try:
        if choice == "1":
            run_video_search(topic)

        elif choice == "2":
            run_recommendation(topic)

    except KeyboardInterrupt:
        print("\nOperation cancelled by the user.")

    except Exception as error:
        print("\nAn error occurred while processing your request.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")


if __name__ == "__main__":
    main()
