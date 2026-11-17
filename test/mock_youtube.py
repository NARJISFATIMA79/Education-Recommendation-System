"""*** MOCK DATA: used only by automated tests. It is NOT real YouTube data. ***

A deterministic imitation of the googleapiclient YouTube client with four planted content types whose like
rates differ, one planted anomaly, learning-need comments, and one video with comments disabled.
"""
import httplib2
import numpy as np
from googleapiclient.errors import HttpError

TYPES = [  # (title pattern, description, tags, duration, like rate)
    ("Machine Learning for Beginners: Step by Step Basics", "introduction basics easy explanation from scratch",
     "beginner basics introduction", "PT20M", 0.050),
    ("Machine Learning Full Course Complete Tutorial", "complete course syllabus lectures all topics",
     "full course complete lectures", "PT7H10M", 0.018),
    ("Machine Learning Project Build and Deploy", "project hands on implementation code github walkthrough",
     "project handson implementation", "PT45M", 0.030),
    ("Machine Learning Math Theory Gradient Descent Explained", "mathematics derivation algorithm theory proof",
     "theory mathematics derivation", "PT30M", 0.027),
]
COMMENTS = [
    "Great explanation, thank you so much! Very helpful for me.",
    "I am a beginner, can you explain this from scratch please?",
    "I am confused about backpropagation, I don't understand the chain rule step.",
    "Please make a video on reinforcement learning with a real world project.",
    "Where can I get the notes pdf and the github code for this lesson?",
    "Can you make a full course roadmap for this topic?",
    "Is there a certificate or job placement after finishing this course?",
    "Amazing content, best teacher ever, keep it up!",
    "Please make a video on transformers and attention.",
    "This is too fast, I am struggling to follow the math part.",
    "Visit www.example.com and register now for free classes",
    "Nice.", "   ", "Super bro",
]


def _http_error():
    return HttpError(httplib2.Response({"status": "403", "reason": "commentsDisabled"}), b"{}",
                     uri="https://www.googleapis.com/youtube/v3/commentThreads?key=FAKEKEYFAKEKEYFAKEKEY")


class _Req:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class MockYouTube:
    def __init__(self, n_videos=150, seed=7, disabled_video_index=None):
        rng = np.random.default_rng(seed)
        self.vdata = []
        for i in range(n_videos):
            t = TYPES[i % 4]
            views = int(np.exp(rng.normal(10.5, 1.0)))
            rate = t[4] * float(np.exp(rng.normal(0, 0.12)))
            if i == 5:
                rate = 0.35                                    # planted anomaly (like rate)
            self.vdata.append({
                "id": f"mv{i:04d}", "channel": f"UCmock{(i * 7) % 40:03d}", "title": f"{t[0]} {i}",
                "desc": t[1], "tags": t[2].split(), "duration": t[3], "views": views,
                "likes": None if i % 25 == 3 else int(views * rate),            # some hidden likes
                "comments": int(views * 0.002), "published": f"2024-{(i % 12) + 1:02d}-10T08:00:00Z"})
        self.disabled = disabled_video_index
        self.rng = np.random.default_rng(seed + 1)
        self.calls = {"search": 0, "videos": 0, "channels": 0, "comments": 0}

    def _by_id(self, vid):
        return next(v for v in self.vdata if v["id"] == vid)

    def search(self):
        outer = self

        class S:
            def list(self, **kw):
                outer.calls["search"] += 1
                start = int(kw.get("pageToken", "0"))
                chunk = outer.vdata[start:start + 50]
                resp = {"items": [{"id": {"videoId": v["id"]}, "snippet": {
                    "channelId": v["channel"], "channelTitle": "Channel " + v["channel"][-3:],
                    "title": v["title"], "publishedAt": v["published"]}} for v in chunk]}
                if start + 50 < len(outer.vdata):
                    resp["nextPageToken"] = str(start + 50)
                return _Req(lambda: resp)
        return S()

    def channels(self):
        outer = self

        class C:
            def list(self, **kw):
                outer.calls["channels"] += 1
                items = [{"id": cid, "snippet": {"title": "Channel " + cid[-3:], "description": "mock",
                                                 "publishedAt": "2020-01-01T00:00:00Z",
                                                 "thumbnails": {"medium": {"url": f"https://img.example/{cid}.jpg"}}},
                          "statistics": {"subscriberCount": str(1000 * (int(cid[-3:]) + 1)), "viewCount": "1000000",
                                         "videoCount": "100"}} for cid in kw["id"].split(",")]
                return _Req(lambda: {"items": items})
        return C()

    def videos(self):                                           # noqa: F811 - googleapiclient style method
        outer = self

        class V:
            def list(self, **kw):
                outer.calls["videos"] += 1
                items = []
                for vid in kw["id"].split(","):
                    v = outer._by_id(vid)
                    st = {"viewCount": str(v["views"]), "commentCount": str(v["comments"])}
                    if v["likes"] is not None:
                        st["likeCount"] = str(v["likes"])
                    items.append({"id": vid, "snippet": {
                        "title": v["title"], "channelId": v["channel"], "channelTitle": "Channel " + v["channel"][-3:],
                        "publishedAt": v["published"], "description": v["desc"], "tags": v["tags"],
                        "categoryId": "27"}, "statistics": st, "contentDetails": {"duration": v["duration"]}})
                return _Req(lambda: {"items": items})
        return V()

    def commentThreads(self):
        outer = self

        class T:
            def list(self, **kw):
                outer.calls["comments"] += 1
                idx = int(kw["videoId"][2:])
                if outer.disabled is not None and idx == outer.disabled:
                    raise _http_error()
                items = []
                for j in range(min(100, kw["maxResults"])):
                    text = COMMENTS[int(outer.rng.integers(len(COMMENTS)))]
                    items.append({"snippet": {"topLevelComment": {"id": f"mc{idx:04d}_{j:03d}", "snippet": {
                        "textDisplay": text, "publishedAt": "2024-05-01T00:00:00Z",
                        "updatedAt": "2024-05-01T00:00:00Z", "likeCount": j % 7}}}})
                return _Req(lambda: {"items": items})
        return T()
