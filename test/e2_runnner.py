"""Run by test_end_to_end.py inside a temporary project copy: patches ONLY the YouTube client factory."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import comment_extractor
from mock_youtube import MockYouTube

mock = MockYouTube(disabled_video_index=int(sys.argv[2]) if len(sys.argv) > 2 else None)
comment_extractor.create_youtube_client = lambda: mock
import enhanced_main
code = enhanced_main.main(["Machine Learning", "--extra-search-pages", "2"])
print("MOCK_CALLS", mock.calls)
sys.exit(code)
