
from youtube_client import (
    search_educational_videos,
    recommend_educational_channels
)


def format_count(value):
    """Format numeric counts without adding extra metadata."""

    if value is None or value == "":
        return "Not available"

    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def display_videos(videos):
    """Display each video in the requested format."""

    if not videos:
        print("No videos found for this topic.")
        return

    for number, video in enumerate(videos, start=1):
        print("\n" + "=" * 70)
        print(f"{number}. {video['title']}")
        print("=" * 70)
        print(f"Channel: {video['channel_title']}")
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
            f"{video.get('default_video_language', 'Unknown')}"
        )
        print(
            "Default audio language: "
            f"{video.get('default_audio_language', 'Unknown')}"
        )
        # print(
        #     "Caption availability: "
        #     f"{video.get('caption_available', 'Unknown')}"
        # )
        # print(
        #     "Caption language: "
        #     f"{video.get('caption_language', 'Not verified')}"
        # )


def main():
    print("\n" + "=" * 60)
    print("EDUCATIONAL YOUTUBE RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("1. Find Educational Videos")
    print("2. Recommend Educational Channels")

    choice = input("\nChoose an option (1 or 2): ").strip()

    if choice not in ("1", "2"):
        print("Invalid option. Please choose 1 or 2.")
        return

    if choice == "2":
        recommend_educational_channels("")
        return

    topic = input("Enter an educational topic: ").strip()

    if not topic:
        print("Topic cannot be empty.")
        return

    try:
        print(f"\nSearching for educational videos about: {topic}")

        videos = search_educational_videos(
            topic,
            max_results=10
        )

        display_videos(videos)

    except Exception as error:
        print("\nAn error occurred while processing your request.")
        print(f"Error details: {error}")


if __name__ == "__main__":
    main()
