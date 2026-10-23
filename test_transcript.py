
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

video_id = input("Enter a YouTube video ID: ").strip()

try:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    print("\nTranscript retrieved successfully:\n")

    for snippet in transcript:
        print(snippet.text)

except (TranscriptsDisabled, NoTranscriptFound):
    print(
        "No accessible transcript was found for this video."
    )

except (RequestBlocked, IpBlocked):
    print(
        "YouTube is blocking transcript requests from this network. "
        "Try again later or use another permitted access method."
    )

except VideoUnavailable:
    print(
        "The video is unavailable or cannot be accessed."
    )

except Exception as error:
    print(f"Transcript retrieval failed: {error}")
