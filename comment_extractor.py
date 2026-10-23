
import os
import time

import pandas as pd
from googleapiclient.errors import HttpError

from youtube_client import create_youtube_client


DATA_DIR = "data"
TARGET_CHANNELS = 18
MAX_COMMENTS_PER_VIDEO = 120
MAX_SEARCH_PAGES = 5


def ensure_data_directory():
    """Create the data directory if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def discover_channels_and_videos(youtube, topic, target_channels=TARGET_CHANNELS):
    """
    Search YouTube for relevant videos and select one video
    from each distinct channel.
    """
    print(f"\nSearching YouTube for: {topic}")

    selected = []
    seen_channels = set()
    page_token = None
    pages_checked = 0

    while len(selected) < target_channels and pages_checked < MAX_SEARCH_PAGES:
        request_args = {
            "part": "snippet",
            "q": topic,
            "type": "video",
            "maxResults": 50,
            "order": "relevance",
        }

        if page_token:
            request_args["pageToken"] = page_token

        response = youtube.search().list(**request_args).execute()
        pages_checked += 1

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId")
            channel_id = snippet.get("channelId")

            if not video_id or not channel_id:
                continue

            # Only one selected video per distinct channel.
            if channel_id in seen_channels:
                continue

            seen_channels.add(channel_id)

            selected.append({
                "video_id": video_id,
                "video_title": snippet.get("title", ""),
                "channel_id": channel_id,
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
            })

            print(
                f"Found channel {len(selected)}/{target_channels}: "
                f"{snippet.get('channelTitle', 'Unknown')}"
            )

            if len(selected) >= target_channels:
                break

        page_token = response.get("nextPageToken")

        if not page_token:
            break

        # Brief pause between paginated requests.
        time.sleep(0.2)

    print(f"\nDistinct channels found: {len(selected)}")
    return selected


def fetch_channel_metadata(youtube, selected_videos):
    """Fetch subscriber counts and basic metadata for selected channels."""
    channel_ids = list({
        item["channel_id"] for item in selected_videos
    })

    channel_records = []

    # YouTube accepts up to 50 channel IDs in one request.
    for start in range(0, len(channel_ids), 50):
        batch = channel_ids[start:start + 50]

        response = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(batch),
            maxResults=50,
        ).execute()

        for channel in response.get("items", []):
            stats = channel.get("statistics", {})
            snippet = channel.get("snippet", {})

            channel_records.append({
                "channel_id": channel.get("id", ""),
                "channel_title": snippet.get("title", ""),
                "channel_description": snippet.get("description", ""),
                "channel_created_at": snippet.get("publishedAt", ""),
                "subscribers": stats.get("subscriberCount"),
                "total_channel_views": stats.get("viewCount"),
                "total_channel_videos": stats.get("videoCount"),
                "subscriber_count_hidden": stats.get(
                    "hiddenSubscriberCount", False
                ),
            })

    return channel_records


def fetch_video_metadata(youtube, selected_videos):
    """Fetch statistics and language/caption metadata for selected videos."""
    video_records = []

    video_ids = [
        item["video_id"] for item in selected_videos
    ]

    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]

        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch),
            maxResults=50,
        ).execute()

        for video in response.get("items", []):
            snippet = video.get("snippet", {})
            stats = video.get("statistics", {})
            details = video.get("contentDetails", {})
            video_id = video.get("id", "")

            video_records.append({
                "video_id": video_id,
                "video_title": snippet.get("title", ""),
                "channel_id": snippet.get("channelId", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "duration": details.get("duration", ""),
                "views": stats.get("viewCount"),
                "likes": stats.get("likeCount"),
                "comments": stats.get("commentCount"),
                "default_video_language": snippet.get(
                    "defaultLanguage", "Unknown"
                ),
                "default_audio_language": snippet.get(
                    "defaultAudioLanguage", "Unknown"
                ),
                "caption_available": details.get("caption", "false"),
                "caption_language": "Not verified",
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
            })

    return video_records


def fetch_comments(youtube, video):
    """Fetch up to 100 top-level comments for one selected video."""
    comments = []
    page_token = None

    while len(comments) < MAX_COMMENTS_PER_VIDEO:
        remaining = MAX_COMMENTS_PER_VIDEO - len(comments)

        request_args = {
            "part": "snippet",
            "videoId": video["video_id"],
            "maxResults": min(100, remaining),
            "textFormat": "plainText",
            "order": "relevance",
        }

        if page_token:
            request_args["pageToken"] = page_token

        try:
            response = youtube.commentThreads().list(
                **request_args
            ).execute()

        except HttpError as error:
            # A video may have disabled comments or restricted access.
            print(
                f"Could not fetch comments for "
                f"'{video['video_title']}': {error}"
            )
            break

        for item in response.get("items", []):
            thread = item.get("snippet", {})
            top_comment = thread.get("topLevelComment", {})
            comment_snippet = top_comment.get("snippet", {})

            comments.append({
                "comment_id": top_comment.get("id", ""),
                "video_id": video["video_id"],
                "video_title": video["video_title"],
                "channel_id": video["channel_id"],
                "channel_title": video["channel_title"],
                "comment_text": comment_snippet.get("textDisplay", ""),
                "comment_published_at": comment_snippet.get(
                    "publishedAt", ""
                ),
                "comment_updated_at": comment_snippet.get(
                    "updatedAt", ""
                ),
                "like_count": comment_snippet.get("likeCount", 0),
                "video_url": video["video_url"],
            })

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return comments


def extract_comments_for_topic(
    topic,
    target_channels=TARGET_CHANNELS,
    max_comments_per_video=MAX_COMMENTS_PER_VIDEO,
):
    """
    Main extraction workflow:
    1. Discover distinct channels and one video per channel.
    2. Save channel and video metadata.
    3. Extract raw top-level comments.
    4. Save the raw comments dataset.
    """
    global MAX_COMMENTS_PER_VIDEO
    MAX_COMMENTS_PER_VIDEO = max_comments_per_video

    ensure_data_directory()
    youtube = create_youtube_client()

    selected_videos = discover_channels_and_videos(
        youtube, topic, target_channels
    )

    if not selected_videos:
        print("No relevant videos were found. Try another topic.")
        return

    print("\nFetching channel metadata...")
    channel_records = fetch_channel_metadata(youtube, selected_videos)

    print("Fetching selected video metadata...")
    video_records = fetch_video_metadata(youtube, selected_videos)

    channel_path = os.path.join(DATA_DIR, "channel_metadata.csv")
    video_path = os.path.join(DATA_DIR, "video_metadata.csv")
    comments_path = os.path.join(DATA_DIR, "raw_comments.csv")

    pd.DataFrame(channel_records).to_csv(
        channel_path, index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(video_records).to_csv(
        video_path, index=False, encoding="utf-8-sig"
    )

    all_comments = []

    print("\nExtracting comments from selected videos...")

    for index, video in enumerate(selected_videos, start=1):
        print(
            f"\nVideo {index}/{len(selected_videos)}: "
            f"{video['video_title']}"
        )

        try:
            video_comments = fetch_comments(youtube, video)
        except HttpError as error:
            print(f"Skipping this video due to API error: {error}")
            video_comments = []

        all_comments.extend(video_comments)

        print(f"Comments collected for this video: {len(video_comments)}")

    comments_df = pd.DataFrame(
        all_comments,
        columns=[
            "comment_id",
            "video_id",
            "video_title",
            "channel_id",
            "channel_title",
            "comment_text",
            "comment_published_at",
            "comment_updated_at",
            "like_count",
            "video_url",
        ],
    )

    comments_df.to_csv(
        comments_path, index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 55)
    print("COMMENT EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"Topic: {topic}")
    print(f"Distinct channels selected: {len(selected_videos)}")
    print(f"Videos with metadata: {len(video_records)}")
    print(f"Total comments collected: {len(comments_df)}")
    print(f"\nComments saved to: {comments_path}")
    print(f"Video metadata saved to: {video_path}")
    print(f"Channel metadata saved to: {channel_path}")
    print("\nRaw comments have not been cleaned or modified.")
