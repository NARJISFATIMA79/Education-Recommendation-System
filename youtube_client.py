
import os
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")


def create_youtube_client():
    if not API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY is missing from the .env file."
        )

    return build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )


def convert_duration(duration):
    """Convert an ISO 8601 duration into a readable format."""
    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        duration
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
    youtube = create_youtube_client()

    # Step 1: Search for videos matching the topic.
    response = youtube.search().list(
        part="snippet",
        q=topic,
        type="video",
        maxResults=max_results,
        relevanceLanguage="en"
    ).execute()

    search_results = response.get("items", [])

    if not search_results:
        return []

    # Step 2: Extract video IDs from search results.
    video_ids = [
        item["id"]["videoId"]
        for item in search_results
    ]

    # Step 3: Fetch statistics and duration for these videos.
    details_response = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids)
    ).execute()

    details_by_id = {
        item["id"]: item
        for item in details_response.get("items", [])
    }

    videos = []

    # Step 4: Combine the search results and video details.
    for item in search_results:
        video_id = item["id"]["videoId"]
        details = details_by_id.get(video_id)

        # Some videos may no longer be accessible.
        if not details:
            continue

        snippet = details["snippet"]
        statistics = details.get("statistics", {})
        content_details = details.get("contentDetails", {})

        videos.append({
            "video_id": video_id,
            "title": snippet["title"],
            "channel_id": snippet["channelId"],
            "channel_title": snippet["channelTitle"],
            "description": snippet.get("description", ""),
            "published_at": snippet.get("publishedAt", ""),
            "views": statistics.get("viewCount"),
            "likes": statistics.get("likeCount"),
            "duration": convert_duration(
                content_details.get("duration", "")
            ),
            "url": (
                "https://www.youtube.com/watch?v="
                + video_id
            )
        })

    return videos

def recommend_educational_channels(
    topic,
    learning_level="All Levels",
    max_results=10
):
    youtube = create_youtube_client()

    # Step 1: Find videos related to the topic.
    videos = search_educational_videos(
    topic,
    max_results=max_results
)

    for video in videos:
       video["learning_level"] = classify_learning_level(video)

    videos = filter_videos_by_level(
      videos,
      learning_level

)

    if not videos:
        return []

    # Step 2: Group videos by their channel IDs.
    channel_groups = {}

    for video in videos:
        channel_id = video["channel_id"]

        if channel_id not in channel_groups:
            channel_groups[channel_id] = {
                "channel_id": channel_id,
                "channel_title": video["channel_title"],
                "matching_videos": []
            }

        channel_groups[channel_id]["matching_videos"].append(
            video
        )

    # Step 3: Retrieve information about each unique channel.
    channel_ids = list(channel_groups.keys())

    response = youtube.channels().list(
        part="snippet,statistics",
        id=",".join(channel_ids)
    ).execute()

    channel_details = {
        channel["id"]: channel
        for channel in response.get("items", [])
    }

    # Step 4: Combine channel information with matching videos.
    recommendations = []

    for channel_id, group in channel_groups.items():
        details = channel_details.get(channel_id, {})
        snippet = details.get("snippet", {})
        statistics = details.get("statistics", {})

        recommendations.append({
            "channel_id": channel_id,
            "channel_title": snippet.get(
                "title",
                group["channel_title"]
            ),
            "description": snippet.get("description", ""),
            "published_at": snippet.get("publishedAt", ""),
            "subscribers": statistics.get("subscriberCount"),
            "total_views": statistics.get("viewCount"),
            "total_videos": statistics.get("videoCount"),
            "matching_video_count": len(
                group["matching_videos"]
            ),
            "channel_url": (
                "https://www.youtube.com/channel/"
                + channel_id
            ),
            "matching_videos": group["matching_videos"]
        })

    # Step 5: Show channels with more matching videos first.
    recommendations.sort(
        key=lambda channel: channel["matching_video_count"],
        reverse=True
    )

    return recommendations

def classify_learning_level(video):
    """
    Estimate learning level using video title and description.
    This is a keyword-based estimate, not a guaranteed level.
    """
    title = video.get("title", "").lower()
    description = video.get("description", "").lower()
    text = title + " " + description

    beginner_keywords = [
        "beginner", "beginners", "for dummies",
        "introduction", "intro to", "basics",
        "fundamentals", "getting started",
        "from scratch", "step by step",
        "what is", "explained simply",
        "101", "tutorial for beginners"
    ]

    advanced_keywords = [
        "advanced", "research paper",
        "state of the art", "mathematical proof",
        "mathematical derivation", "optimization theory",
        "research-level", "graduate level",
        "theoretical analysis", "in-depth analysis"
    ]

    intermediate_keywords = [
        "intermediate", "hands-on project",
        "implementation", "practical project",
        "case study", "end-to-end project",
        "real-world project", "applied machine learning",
        "build a project", "complete project",
        "from theory to practice"
    ]

    # Give explicit learning-level labels priority.
    if any(word in title for word in advanced_keywords):
        return "Advanced"

    if any(word in title for word in intermediate_keywords):
        return "Intermediate"

    if any(word in title for word in beginner_keywords):
        return "Beginner"

    # Use the description if the title doesn't provide a clear signal.
    if any(word in text for word in advanced_keywords):
        return "Advanced"

    if any(word in text for word in intermediate_keywords):
        return "Intermediate"

    if any(word in text for word in beginner_keywords):
        return "Beginner"

    return "Unclassified"
      


def filter_videos_by_level(videos, learning_level):
    """
    Keep videos that match the selected learning level.
    """
    if learning_level == "All Levels":
        return videos

    return [
        video
        for video in videos
        if classify_learning_level(video) == learning_level
    ]


