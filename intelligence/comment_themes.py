"""Learning-need signals and recurring themes in viewer comments.

Two complementary views (sentiment is a third, separate view and comes from the ORIGINAL VADER code):
  * need signals  : transparent keyword rules -> auditable counts for specific needs (beginner, project...)
  * theme clusters: TF-IDF + SVD + K-Means -> recurring topics discovered without predefined labels
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from intelligence import clustering
from recommender import analyze_comments               # ORIGINAL VADER function, reused unchanged

MIN_DOCS_FOR_THEMES = 60
MIN_TOKENS = 4                 # shorter comments are mostly praise/emoji and carry no theme information
MIN_SIGNAL_COMMENTS = 3        # a need is "repeated" only if at least this many distinct comments mention it
MIN_SIGNAL_VIDEOS = 2          # ... under at least this many different videos
MIN_DEMAND = 2
GAP_MAX_SUPPLY_SHARE = 0.10

NEED_SIGNALS = {
    "beginner_friendly": ("Mentions beginner level, basics or learning from scratch",
                          r"\b(beginners?|newbies?|noobs?|basics?|basic level|from scratch|from zero|starting out|"
                          r"just started|new to|introduction|intro)\b"),
    "practical_projects": ("Mentions projects, hands-on practice, examples or real-world use",
                           r"\b(projects?|hands[- ]on|practicals?|real[- ]world|real[- ]time|case stud(?:y|ies)|"
                           r"implementation|demos?|examples?|use cases?|labs?|assignments?)\b"),
    "difficulty_understanding": ("Expresses confusion or difficulty",
                                 r"\b(confus\w*|difficult\w*|hard to|too hard|struggl\w*|don'?t understand|"
                                 r"did ?n'?t understand|do not understand|not able to|unable to|stuck|"
                                 r"can'?t (?:understand|follow|get)|too fast|lost me)\b"),
    "complete_course_request": ("Asks for a full course, series, roadmap or complete tutorial",
                                r"\b(full course|complete course|whole course|playlist|series|complete tutorial|"
                                r"full tutorial|roadmap|road map|syllabus|curriculum|zero to hero|end to end)\b"),
    "advanced_content": ("Asks for advanced or in-depth material",
                         r"\b(advanced|in[- ]depth|deep dive|next level|expert level|deeper)\b"),
    "learning_resources": ("Asks about books, notes, slides, datasets, code or other resources",
                           r"\b(books?|resources?|notes|pdf|slides|github|datasets?|source code|ppt|"
                           r"study material|materials?|references?)\b"),
    "pricing": ("Mentions price, fees, paid/free access or discounts",
                r"\b(price|pricing|cost|fees?|paid|free of cost|discount|scholarship|affordable)\b"),
    "certification_career": ("Mentions certificates, jobs, careers, admission or enrolment",
                             r"\b(certificat\w*|certification|enrol\w*|admission|diploma|degree|jobs?|placement|"
                             r"internship|salary|career)\b"),
    "content_request": ("Directly asks the creator for more or specific content",
                        r"(?:\b(?:please|plz|pls|kindly)\s+(?:make|upload|do|explain|teach|cover|share|provide|add|bring|"
                        r"create|post|give|start)\b|\b(?:can|could|will) you (?:please )?(?:make|upload|do|explain|teach|"
                        r"cover|share|provide|add|create|post|give)\b|make (?:a )?(?:video|tutorial)|do a video|"
                        r"need (?:a )?(?:video|tutorial)|upload (?:a |the |more |next )?(?:video|tutorial|part))"),
}
_COMPILED = {k: re.compile(p, re.I) for k, (_, p) in NEED_SIGNALS.items()}
_QUESTION_START = re.compile(r"^(how|what|why|when|where|which|who|can|could|is|are|do|does|did|will|should|"
                             r"would|any)\b", re.I)
_URL = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|in|org|net)/\S*)", re.I)
_PROMO = re.compile(r"(whatsapp|telegram|click (?:here|the link)|dm me|inbox me|subscribe to my|visit (?:our|my)|"
                    r"register (?:and|now|here)|link in (?:bio|description))", re.I)
_HINGLISH = set("hai nahi nahin kya aap mujhe bohat boht bahut acha achha achi shukriya samajh karo kar ke ka ki "
                "mein main bhai bhaiya yaar tum hum kaise kaisa kyun lekin aur ko se ho hota hoti".split())
_STOP = set(ENGLISH_STOP_WORDS) | {"video", "videos", "sir", "mam", "madam", "thanks", "thank", "please", "plz",
                                   "pls", "make", "tutorial", "tutorials", "lecture", "course"}


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    t = _URL.sub(" ", text.lower())
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", t)            # timestamps like 12:34
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                             # sooooo -> soo
    t = re.sub(r"[^a-z0-9'+#?\s]", " ", t)                           # drop emoji / punctuation / other scripts
    return re.sub(r"\s+", " ", t).strip()


def is_probably_english(raw):
    """Conservative heuristic: mostly Latin letters and few Roman-Hindi/Urdu function words."""
    if not isinstance(raw, str):
        return False
    letters = [ch for ch in raw if ch.isalpha()]
    if not letters:
        return False
    latin = sum(1 for ch in letters if ch.isascii()) / len(letters)
    toks = re.findall(r"[a-z']+", raw.lower())
    hing = sum(t in _HINGLISH for t in toks) / max(len(toks), 1)
    return latin >= 0.9 and hing <= 0.15


def _tokens(norm):
    return re.findall(r"[a-z0-9+#']+", norm)


def prepare_comments(comments):
    """Flag every comment once; later steps filter on these flags (nothing is silently dropped)."""
    df = comments.copy().reset_index(drop=True)
    for col in ("comment_id", "video_id", "channel_id"):
        if col not in df:
            df[col] = np.nan
    df["raw_text"] = df["comment_text"].astype("string").fillna("")
    df["norm"] = df["raw_text"].map(normalize_text)
    df["n_tokens"] = df["norm"].map(lambda s: len(_tokens(s)))
    df["is_empty"] = df["norm"].eq("")
    df["is_duplicate"] = df.duplicated(subset=["video_id", "norm"], keep="first") & ~df["is_empty"]
    df["is_promotional"] = df["raw_text"].map(lambda s: bool(_URL.search(s) or _PROMO.search(s)))
    df["is_english"] = df["raw_text"].map(is_probably_english)
    df["is_question"] = df["raw_text"].map(lambda s: "?" in s or bool(_QUESTION_START.match(s.strip())))
    df["analysable"] = (~df["is_empty"] & ~df["is_duplicate"] & ~df["is_promotional"] & df["is_english"]
                        & (df["n_tokens"] >= 3))
    df["informative"] = df["analysable"] & (df["n_tokens"] >= MIN_TOKENS) & \
        df["norm"].map(lambda s: sum(t not in _STOP for t in _tokens(s)) >= 2)
    for name, rx in _COMPILED.items():
        df[f"sig_{name}"] = df["norm"].map(lambda s: bool(rx.search(s))) & df["analysable"]
    return df


def _excerpt(text, n=160):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def signal_summary(df):
    rows, denom = [], int(df["analysable"].sum())
    for name, (desc, _) in NEED_SIGNALS.items():
        hit = df[df[f"sig_{name}"]]
        n_videos = hit["video_id"].nunique()
        repeated = len(hit) >= MIN_SIGNAL_COMMENTS and n_videos >= MIN_SIGNAL_VIDEOS
        ex = hit.assign(l=hit["raw_text"].str.len()).query("l >= 25").sort_values("l").head(3)["raw_text"]
        rows.append({"signal": name, "description": desc, "comments": len(hit),
                     "share_of_analysable": (len(hit) / denom) if denom else np.nan,
                     "videos": n_videos, "channels": hit["channel_id"].nunique(),
                     "repeated": repeated, "examples": " || ".join(_excerpt(x, 140) for x in ex)})
    return pd.DataFrame(rows).sort_values("comments", ascending=False).reset_index(drop=True)


def question_terms(df, top=10):
    q = df[df["analysable"] & df["is_question"]]
    counter = Counter()
    for norm in q["norm"]:
        toks = [t for t in _tokens(norm) if t not in _STOP and len(t) > 2]
        counter.update(set(toks))
        counter.update(set(" ".join(p) for p in zip(toks, toks[1:])))
    rows = [{"term": t, "question_comments": c} for t, c in counter.most_common(60) if c >= 3][:top]
    return pd.DataFrame(rows, columns=["term", "question_comments"]), len(q)


_DEMAND_PATTERNS = [
    re.compile(r"\b(?:video|videos|tutorial|tutorials|course|courses|lecture|lectures|class|session|series|content|"
               r"playlist)\s+(?:on|about|for|of|regarding|related to)\s+((?:[a-z0-9+#.]+\s?){1,4})"),
    re.compile(r"\b(?:please|plz|pls)\s+(?:explain|teach|cover|discuss|make|do|upload)\s+(?:us\s+|me\s+)?"
               r"(?:a\s+)?(?:video\s+|tutorial\s+)?(?:on|about)\s+((?:[a-z0-9+#.]+\s?){1,4})"),
]


def demand_vs_supply(df, supply_texts):
    """Topics viewers ask for ('video on X') versus how many sampled videos mention X."""
    hits = Counter()
    where = {}
    for _, r in df[df["analysable"]].iterrows():
        for rx in _DEMAND_PATTERNS:
            for m in rx.finditer(r["norm"]):
                toks = [t for t in _tokens(m.group(1)) if t not in _STOP][:3]
                if not toks:
                    continue
                phrase = " ".join(toks)
                hits[phrase] += 1
                where.setdefault(phrase, set()).add(r["video_id"])
    supply = [str(t).lower() for t in supply_texts if isinstance(t, str) and t.strip()]
    rows = []
    for phrase, n in hits.most_common():
        toks = phrase.split()
        covering = sum(all(re.search(rf"\b{re.escape(t)}\b", s) for t in toks) for s in supply)
        share = covering / len(supply) if supply else np.nan
        rows.append({"requested_topic": phrase, "demand_comments": n, "demand_videos": len(where[phrase]),
                     "supply_videos": covering, "supply_sample_size": len(supply), "supply_share": share,
                     "possible_gap": bool(supply and n >= MIN_DEMAND and share <= GAP_MAX_SUPPLY_SHARE)})
    return pd.DataFrame(rows, columns=["requested_topic", "demand_comments", "demand_videos", "supply_videos",
                                       "supply_sample_size", "supply_share", "possible_gap"])


def difficulty_terms(df, top=6):
    """Most frequent content words in comments that express difficulty understanding."""
    hit = df[df["sig_difficulty_understanding"]]
    if len(hit) < MIN_SIGNAL_COMMENTS:
        return ""
    counter = Counter()
    for norm in hit["norm"]:
        counter.update({t for t in _tokens(norm) if t not in _STOP and len(t) > 3})
    return ", ".join(t for t, c in counter.most_common(top) if c >= 2)


def run_comment_analysis(cleaned_comments, supply_texts=()):
    """Full comment analysis. Returns a dict of DataFrames and diagnostics."""
    df = prepare_comments(cleaned_comments)
    sentiment = analyze_comments(df.assign(comment_text=df["raw_text"].where(~df["is_empty"], None)))
    df["sentiment_label"] = np.where(df["is_empty"], None, sentiment["sentiment"].to_numpy())
    df["sentiment_compound"] = np.where(df["is_empty"], np.nan, sentiment["sentiment_score"].to_numpy())
    counts = {"total": len(df), "empty": int(df["is_empty"].sum()), "duplicates": int(df["is_duplicate"].sum()),
              "promotional": int(df["is_promotional"].sum()),
              "non_english_or_mixed": int((~df["is_english"] & ~df["is_empty"]).sum()),
              "analysable": int(df["analysable"].sum()), "informative": int(df["informative"].sum())}
    signals = signal_summary(df)
    qterms, n_questions = question_terms(df)
    counts["question_comments"] = n_questions
    demand = demand_vs_supply(df, supply_texts)

    pool = df[df["informative"]]
    info, fit = clustering.cluster_texts(pool["norm"], min_docs=MIN_DOCS_FOR_THEMES, k_max=8,
                                         min_cluster_size=max(5, len(pool) // 40))
    df["theme_cluster"] = np.nan
    themes = pd.DataFrame()
    if fit is not None:
        df.loc[pool.index, "theme_cluster"] = fit["labels"]
        rows = []
        pool_idx = pool.index.to_numpy()
        for c, meta in fit["clusters"].items():
            members = df[df["theme_cluster"] == c]
            reps = [_excerpt(df.loc[pool_idx[i], "raw_text"]) for i in meta["representative_idx"]]
            row = {"cluster": c, "n_comments": len(members), "share_of_informative": len(members) / len(pool),
                   "top_terms": ", ".join(meta["top_terms"][:6]),
                   "mean_vader_compound": members["sentiment_compound"].mean(),
                   "question_share": members["is_question"].mean(),
                   "representative_comments": " || ".join(reps)}
            for name in NEED_SIGNALS:
                row[f"share_{name}"] = members[f"sig_{name}"].mean()
            rows.append(row)
        themes = pd.DataFrame(rows).sort_values("n_comments", ascending=False).reset_index(drop=True)
    return {"comments": df, "counts": counts, "signals": signals, "question_terms": qterms,
            "demand_supply": demand, "theme_info": info, "themes": themes,
            "difficulty_terms": difficulty_terms(df)}
