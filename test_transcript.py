from youtube_transcript_api import YouTubeTranscriptApi

video_id = input("Enter a YouTube video ID: ").strip()

try:
    api = YouTubeTranscriptApi()

    # Try English first, then Hindi.
    transcript = api.fetch(
        video_id,
        languages=["en", "hi"]
    )

    print("\nTranscript retrieved successfully!")
    print("Language:", transcript.language)
    print("\nFirst 10 transcript segments:\n")

    for snippet in list(transcript)[:10]:
        print(snippet.text)

except Exception as error:
    print(f"\nCould not retrieve transcript: {error}")
