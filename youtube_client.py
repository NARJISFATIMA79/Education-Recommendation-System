
import os
import re
import time

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


def create_youtube_client():
    """Create the YouTube API client."""

    if not API_KEY:
        raise ValueError("YOUTUBE_API_KEY is missing from .env")

    return build(
        "youtube",
        "v3",
        developerKey=API_KEY,
        cache_discovery=False
    )


def timed_api_request(label, request):
    """Execute an API request and measure its duration."""

    start = time.perf_counter()

    try:
        return request.execute()
    finally:
        print(
            f"[TIMING] {label}: "
            f"{time.perf_counter() - start:.2f} seconds"
        )


def convert_duration(duration):
    """Convert ISO 8601 duration into a readable format."""

    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        duration or ""
    )

    if not match:
        return "Unknown"

    hours, minutes, seconds = (
        int(value) if value else 0
        for value in match.groups()
    )

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def search_educational_videos(topic, max_results=10):
    """
    Retrieve videos with only the requested metadata,
    including each video's channel subscriber count.
    """

    total_start = time.perf_counter()
    print("\n[TIMING] Starting video search...")

    try:
        youtube = create_youtube_client()

        search_response = timed_api_request(
            "Video search API",
            youtube.search().list(
                part="snippet",
                q=topic,
                type="video",
                maxResults=max(1, min(max_results, 50)),
                relevanceLanguage="en"
            )
        )

        search_items = search_response.get("items", [])

        if not search_items:
            return []

        video_ids = [
            item["id"]["videoId"]
            for item in search_items
            if item.get("id", {}).get("videoId")
        ]

        details_response = timed_api_request(
            "Video details API",
            youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(video_ids)
            )
        )

        video_details = {
            item["id"]: item
            for item in details_response.get("items", [])
        }

        videos = []

        for item in search_items:
            video_id = item.get("id", {}).get("videoId")
            details = video_details.get(video_id)

            if not details:
                continue

            snippet = details.get("snippet", {})
            stats = details.get("statistics", {})
            content = details.get("contentDetails", {})

            caption_value = content.get("caption")

            if caption_value == "true":
                caption_available = "Yes"
            elif caption_value == "false":
                caption_available = "No"
            else:
                caption_available = "Unknown"

            videos.append({
                "video_id": video_id,
                "title": snippet.get("title", "Unknown"),
                "channel_id": snippet.get("channelId", ""),
                "channel_title": snippet.get(
                    "channelTitle", "Unknown"
                ),
                "published_at": snippet.get(
                    "publishedAt", "Unknown"
                ),
                "duration": convert_duration(
                    content.get("duration", "")
                ),
                "views": stats.get("viewCount"),
                "likes": stats.get("likeCount"),
                "comments": stats.get("commentCount"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "default_video_language": snippet.get(
                    "defaultLanguage", "Unknown"
                ),
                "default_audio_language": snippet.get(
                    "defaultAudioLanguage", "Unknown"
                ),
                "caption_available": caption_available,
                "caption_language": "Not verified"
            })

        # Retrieve subscriber counts for unique channels.
        channel_ids = list(dict.fromkeys(
            video["channel_id"]
            for video in videos
            if video["channel_id"]
        ))

        channel_details = {}

        # YouTube accepts up to 50 channel IDs per request.
        for start in range(0, len(channel_ids), 50):
            batch = channel_ids[start:start + 50]

            response = timed_api_request(
                "Channel details API",
                youtube.channels().list(
                    part="statistics",
                    id=",".join(batch)
                )
            )

            for channel in response.get("items", []):
                channel_details[channel["id"]] = (
                    channel.get("statistics", {})
                )

        for video in videos:
            stats = channel_details.get(
                video["channel_id"], {}
            )
            video["subscribers"] = stats.get("subscriberCount")

        print(f"[INFO] Videos retrieved: {len(videos)}")

        return videos

    finally:
        print(
            "[TIMING] Total video search operation: "
            f"{time.perf_counter() - total_start:.2f} seconds"
        )


def recommend_educational_channels(topic, max_results=10):
    """
    Reserved for future comment-analysis-based recommendations.
    """

    print(
        "\nChannel recommendations are not available yet. "
        "They will be implemented after comment analysis."
    )

    return []
