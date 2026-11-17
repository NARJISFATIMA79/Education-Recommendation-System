"""Requirements 1, 6, 7: fresh collection through the ORIGINAL code, then the new analysis.
YouTube is a MOCK here (tests/mock_youtube.py); this does not prove the live API works."""
import hashlib
import os
import shutil
import subprocess
import sys

import pandas as pd

from conftest import ROOT

ORIGINAL_CODE = ["main.py", "comment_extractor.py", "clean_comments.py", "recommender.py", "youtube_client.py",
                 "enhanced_main.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_project(tmp):
    for f in ORIGINAL_CODE:
        shutil.copy(os.path.join(ROOT, f), tmp)
    shutil.copytree(os.path.join(ROOT, "intelligence"), os.path.join(tmp, "intelligence"))
    shutil.copy(os.path.join(ROOT, "tests", "mock_youtube.py"), tmp)
    shutil.copy(os.path.join(ROOT, "tests", "e2e_runner.py"), tmp)


def run(tmp, *extra):
    env = dict(os.environ, YOUTUBE_API_KEY="dummy-not-used", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "e2e_runner.py", tmp, *extra], cwd=tmp, env=env,
                          capture_output=True, text=True)


def test_fresh_collection_original_csvs_and_new_analysis(tmp_path):
    tmp = str(tmp_path); make_project(tmp)
    out = run(tmp, "11")                                           # video index 11 has comments disabled
    assert out.returncode == 0, out.stdout + out.stderr
    d = os.path.join(tmp, "data")
    # original CSV workflow still produces its files with the original columns
    for name in ("channel_metadata.csv", "video_metadata.csv", "raw_comments.csv", "cleaned_comments.csv"):
        assert os.path.exists(os.path.join(d, name)), name
    orig_cols = pd.read_csv(os.path.join(ROOT, "data", "raw_comments.csv"), engine="python").columns
    assert list(pd.read_csv(os.path.join(d, "raw_comments.csv"), engine="python").columns) == list(orig_cols)
    assert list(pd.read_csv(os.path.join(d, "video_metadata.csv"), engine="python").columns) == list(
        pd.read_csv(os.path.join(ROOT, "data", "video_metadata.csv"), engine="python").columns)
    # original collection limits respected
    videos = pd.read_csv(os.path.join(d, "video_metadata.csv"), engine="python")
    assert len(videos) == 15 and videos["channel_id"].nunique() == 15
    # topic sample built from the recorded searches (+2 extra pages) and enriched
    sample = pd.read_csv(os.path.join(d, "market_video_sample.csv"), engine="python")
    assert len(sample) == 150 and sample["description"].notna().all() and sample["in_original_selection"].sum() == 15
    assert sample["likes"].isna().sum() > 0                        # hidden likes stay missing in the sample
    # outputs
    intel = os.path.join(d, "intelligence")
    for name in ("recommendations.csv", "channel_clusters.csv", "video_clusters.csv", "comment_signals.csv",
                 "anomalies.csv", "marketing_insights.csv", "intelligence_summary.json", "intelligence_report.md"):
        assert os.path.exists(os.path.join(intel, name)), name
    # the original engine's ranking equals a direct call on the CSVs just produced
    sys.path.insert(0, tmp)
    from importlib import util
    spec = util.spec_from_file_location("orig_rec", os.path.join(ROOT, "recommender.py"))
    mod = util.module_from_spec(spec); spec.loader.exec_module(mod)
    direct = mod.recommend_channels("Machine Learning", videos,
                                    pd.read_csv(os.path.join(d, "channel_metadata.csv"), engine="python"),
                                    pd.read_csv(os.path.join(d, "cleaned_comments.csv"), engine="python"),
                                    top_n=10 ** 9)
    rec = pd.read_csv(os.path.join(intel, "recommendations.csv"))
    assert list(rec["channel_id"]) == list(direct["channel_id"])
    assert (rec["recommendation_score"] - direct["recommendation_score"].to_numpy()).abs().max() < 1e-12      # CSV text round-trip, not a calculation difference
    # the disabled-comments channel is flagged, not treated as neutral
    ch = pd.read_csv(os.path.join(intel, "channel_clusters.csv"))
    assert (ch["sentiment_status"] != "measured").any()
    assert ch.loc[ch["sentiment_status"] == "not_measured", "sentiment_adjusted"].isna().all()
    # video clustering ran on the 150-video sample and compared content types
    vc = pd.read_csv(os.path.join(intel, "video_clusters.csv"))
    assert vc["cluster"].notna().sum() >= 100
    insights = pd.read_csv(os.path.join(intel, "marketing_insights.csv"))
    assert {"observed_data", "clustering_result", "interpretation", "suggested_action"} <= set(insights.columns)
    assert insights["evidence"].notna().all()
    report = open(os.path.join(intel, "intelligence_report.md"), encoding="utf-8").read()
    assert "Observed data" in report and "does not imply sales" in report


def test_from_csv_mode_reruns_without_touching_collected_files(tmp_path):
    tmp = str(tmp_path); make_project(tmp)
    assert run(tmp).returncode == 0
    d = os.path.join(tmp, "data")
    before = {f: sha(os.path.join(d, f)) for f in ("video_metadata.csv", "channel_metadata.csv", "raw_comments.csv")}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "enhanced_main.py", "Machine Learning", "--from-csv"], cwd=tmp, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert before == {f: sha(os.path.join(d, f)) for f in before}
