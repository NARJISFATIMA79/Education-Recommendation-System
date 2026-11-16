"""Topic sample: on-topic videos surfaced by the SAME search the original collector runs.

The original discover_channels_and_videos() keeps one video per channel and discards the rest of what the
search returned. This module records the search responses the original code already receives (a transparent
proxy, nothing in the original files changes), optionally fetches a few more result pages, and then calls
videos.list once per 50 videos to get description, tags, duration and statistics.

API cost (YouTube Data API v3): extra search page = 100 quota units; videos.list = 1 unit per request of up to 50 ids.
"""
import math
from contextlib import contextmanager

import pandas as pd

try:
    from googleapiclient.errors import HttpError
except ImportError:                       # pragma: no cover
    HttpError = Exception

SEARCH_UNITS = 100


class _RecordedRequest:
    def __init__(self, request, kwargs, log):
        self._request, self._kwargs, self._log = request, kwargs, log

    def execute(self):
        response = self._request.execute()
        self._log.append({"kwargs": dict(self._kwargs), "response": response})
        return response


class _RecordingResource:
    def __init__(self, resource, log):
        self._resource, self._log = resource, log

    def list(self, **kwargs):
        return _RecordedRequest(self._resource.list(**kwargs), kwargs, self._log)


class RecordingYouTube:
    """Wraps a YouTube client; behaves identically but remembers every search.list response."""

    def __init__(self, real):
        self._real = real
        self.search_log = []

    def search(self):
        return _RecordingResource(self._real.search(), self.search_log)

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextmanager
def record_searches(extractor_module):
    """Temporarily make comment_extractor use a recording client. Restores the original on exit."""
    original_factory = extractor_module.create_youtube_client
    holder = {}

    def factory():
        holder["client"] = RecordingYouTube(original_factory())
        return holder["client"]

    extractor_module.create_youtube_client = factory
    try:
        yield holder
    finally:
        extractor_module.create_youtube_client = original_factory


def _items_from_log(log):
    seen, rows = set(), []
    for entry in log:
        for item in entry["response"].get("items", []):
            vid = item.get("id", {}).get("videoId")
            sn = item.get("snippet", {})
            if not vid or vid in seen or not sn.get("channelId"):
                continue
            seen.add(vid)
            rows.append({"video_id": vid, "channel_id": sn["channelId"]})
    return rows


def collect_market_sample(recorder, selected_video_ids, extra_pages=2, log=print):
    """Returns (DataFrame or None, info dict). Never raises for API problems; the caller reports them."""
    info = {"extra_pages_requested": extra_pages, "extra_pages_fetched": 0, "search_results_seen": 0,
            "videos_in_sample": 0, "quota_units_estimate": 0, "error": None}
    youtube = recorder._real
    try:
        if not recorder.search_log:
            info["error"] = "no search responses were recorded"
            return None, info
        for _ in range(max(0, extra_pages)):
            last = recorder.search_log[-1]
            token = last["response"].get("nextPageToken")
            if not token:
                break
            kwargs = dict(last["kwargs"], pageToken=token)
            recorder.search_log.append({"kwargs": kwargs, "response": youtube.search().list(**kwargs).execute()})
            info["extra_pages_fetched"] += 1
            info["quota_units_estimate"] += SEARCH_UNITS
        ids = [r["video_id"] for r in _items_from_log(recorder.search_log)]
        info["search_results_seen"] = len(ids)
        rows = []
        for start in range(0, len(ids), 50):
            batch = ids[start:start + 50]
            response = youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(batch)).execute()
            info["quota_units_estimate"] += 1
            for v in response.get("items", []):
                sn, st, cd = v.get("snippet", {}), v.get("statistics", {}), v.get("contentDetails", {})
                rows.append({
                    "video_id": v.get("id"), "video_title": sn.get("title"), "channel_id": sn.get("channelId"),
                    "channel_title": sn.get("channelTitle"), "published_at": sn.get("publishedAt"),
                    "description": sn.get("description"), "tags": " ".join(sn.get("tags", []) or []),
                    "category_id": sn.get("categoryId"), "duration": cd.get("duration"),
                    "views": st.get("viewCount"), "likes": st.get("likeCount"),
                    "comments": st.get("commentCount"),
                    "default_audio_language": sn.get("defaultAudioLanguage"),
                    "in_original_selection": v.get("id") in set(selected_video_ids)})
        df = pd.DataFrame(rows).drop_duplicates("video_id")
        info["videos_in_sample"] = len(df)
        return df, info
    except HttpError as exc:
        info["error"] = f"YouTube API error while building the topic sample: {type(exc).__name__}"
        log(info["error"])
        return None, info
