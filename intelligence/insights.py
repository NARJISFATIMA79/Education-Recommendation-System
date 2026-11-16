"""Evidence-linked educational marketing insights.

Every insight separates four layers:
  observed      - measurements taken from the collected data
  clustering    - what the clustering / anomaly step found (or that it found nothing reliable)
  interpretation- what this may mean *within this sample*
  action        - a suggested test (never a prediction of sales, conversion or ROI)
An insight is produced only when its rule's evidence threshold is met. Weak or absent evidence yields an
explicit "no reliable pattern" insight instead of advice.
"""
import numpy as np
import pandas as pd
from scipy import stats

MIN_GROUP = 8                  # videos needed in a cluster before it is compared
MIN_LIKES_PER_GROUP = 5
LIFT = 1.2                     # a cluster must beat the other videos by 20% to be called higher
ALPHA = 0.05

NEED_ACTIONS = {
    "beginner_friendly": "Test a beginner-oriented promotion (for example an 'introduction from zero' lesson) "
                         "and compare its response with a general promotion.",
    "practical_projects": "Test a promotion built around a hands-on project or worked example and compare it with "
                          "a lecture-style promotion.",
    "difficulty_understanding": "Prepare FAQ or explainer content for the concepts viewers find confusing "
                                "(see the terms listed) and test it as promotional material.",
    "complete_course_request": "Test a roadmap / full-course overview as a promotion for a structured offering.",
    "advanced_content": "Test an advanced-topics teaser to see whether an advanced track attracts response.",
    "learning_resources": "Test offering notes, code or datasets as a lead-in to a course promotion.",
    "pricing": "State pricing and access conditions clearly in promotional content, because viewers mention cost.",
    "certification_career": "Address certification and career outcomes explicitly (only claims you can support), "
                            "because viewers mention them.",
}
NEED_TITLES = {
    "beginner_friendly": "Viewers mention beginner level, basics or learning from scratch",
    "practical_projects": "Viewers mention projects, hands-on work or examples",
    "difficulty_understanding": "Viewers report difficulty understanding",
    "complete_course_request": "Viewers ask for complete courses, series or roadmaps",
    "advanced_content": "Viewers ask for advanced material",
    "learning_resources": "Viewers ask about books, notes, code or datasets",
    "pricing": "Viewers mention price or cost",
    "certification_career": "Viewers mention certificates, jobs or careers",
}


def _pct(x, digits=1):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def _ev(metric, value, n, source):
    if isinstance(value, (np.floating, float)):
        value = float(value)
    return {"metric": metric, "value": value, "n": int(n) if n is not None else None, "source": source}


class Insights:
    def __init__(self):
        self.items = []

    def add(self, category, title, observed, clustering, interpretation, action, strength, evidence, caveats):
        self.items.append({
            "insight_id": f"I{len(self.items) + 1:02d}", "category": category, "title": title,
            "observed_data": observed, "clustering_result": clustering, "interpretation": interpretation,
            "suggested_action": action, "evidence_strength": strength, "evidence": evidence, "caveats": caveats})

    def frame(self):
        df = pd.DataFrame(self.items)
        if len(df):
            df["evidence"] = df["evidence"].map(
                lambda e: " | ".join(f"{x['metric']}={x['value']} (n={x['n']}; {x['source']})" for x in e))
        return df


# ------------------------------------------------------------------ content intelligence
def _compare_clusters(v, metric, label, ins, info, profile, source):
    groups = {c: g[metric].dropna() for c, g in v.groupby("cluster")}
    groups = {c: x for c, x in groups.items() if len(x) >= MIN_LIKES_PER_GROUP}
    if len(groups) < 2:
        return
    h, p = stats.kruskal(*groups.values())
    n_total = sum(len(x) for x in groups.values())
    medians = {c: float(x.median()) for c, x in groups.items()}
    best = max(medians, key=medians.get)
    rest = pd.concat([x for c, x in groups.items() if c != best])
    ratio = medians[best] / rest.median() if rest.median() > 0 else np.nan
    terms = profile.loc[profile["cluster"] == best, "top_terms"].iloc[0] if "top_terms" in profile else ""
    clus = (f"{info['k']} content clusters (silhouette {info['silhouette_best']:.2f}, permutation p "
            f"{info['null_p_value']:.2f}, stability ARI {info['stability_ari']:.2f}); Kruskal-Wallis p = {p:.3f}.")
    evidence = [_ev(f"median {label} cluster {int(best)}", medians[best], len(groups[best]), source),
                _ev(f"median {label} other clusters", float(rest.median()), len(rest), source),
                _ev("Kruskal-Wallis p", float(p), n_total, source)]
    if p < ALPHA and ratio >= LIFT:
        strength = "moderate" if (p < 0.01 and n_total >= 50) else "low"
        ins.add("content_intelligence", f"One content type shows a higher {label}",
                f"Videos in content cluster {int(best)} (terms: {terms}) have a median {label} of "
                f"{medians[best]:.4g} versus {float(rest.median()):.4g} for the other sampled videos "
                f"(n = {len(groups[best])} vs {len(rest)}).",
                clus,
                f"In this sample, this content type attracts a higher {label}. This describes viewer response on "
                "YouTube; it does not show that it leads to enrolments or sales.",
                f"Test a promotional video in this content type (title pattern: {terms}) against a different "
                "format and compare response on your own channel.",
                strength, evidence,
                "Observational data: topic, channel size and video age are not controlled.")
    else:
        ins.add("content_intelligence", f"No reliable difference in {label} between content types",
                f"Median {label} by cluster: " + ", ".join(f"cluster {int(c)}: {m:.4g}" for c, m in medians.items()) + ".",
                clus, "The sample does not show that any one content type responds better. Do not choose a format "
                      "on the basis of this metric.",
                "Collect a larger or more targeted sample before prioritising a format.", "low", evidence,
                "A non-significant result is not proof that no difference exists.")


def content_insights(ins, video_table, info, profile):
    src = "video_clusters.csv"
    if info is None or info.get("status") not in ("ok",):
        reason = (info or {}).get("reason", "no topic sample was available")
        ins.add("content_intelligence", "Content-type analysis was not performed",
                f"Videos available for analysis: {len(video_table) if video_table is not None else 0}.",
                f"Not performed: {reason}.", "No claim about which content types perform better can be made "
                                              "from this run.",
                "Re-run with the extra search pages enabled (--extra-search-pages) or on a broader topic.",
                "not established", [_ev("videos", len(video_table) if video_table is not None else 0, None, "market_video_sample.csv")],
                "Clustering on very small samples would produce arbitrary groups.")
        return
    v = video_table[video_table["cluster"].notna()]
    _compare_clusters(v, "like_rate", "like rate", ins, info, profile, src)
    _compare_clusters(v, "log_views_per_day", "log views per day", ins, info, profile, src)
    both = v[["log_duration_min", "like_rate"]].dropna()
    if len(both) >= 20:
        rho, p = stats.spearmanr(both["log_duration_min"], both["like_rate"])
        ev = [_ev("Spearman rho (duration vs like rate)", float(rho), len(both), "video_clusters.csv"),
              _ev("p-value", float(p), len(both), "video_clusters.csv")]
        if p < ALPHA and abs(rho) >= 0.3:
            direction = "shorter" if rho < 0 else "longer"
            ins.add("content_intelligence", f"{direction.capitalize()} videos tend to have a higher like rate",
                    f"Spearman correlation between duration and like rate is {rho:.2f} (p = {p:.3f}, n = {len(both)}).",
                    "Not a clustering result (rank correlation).",
                    f"In this sample {direction} videos are associated with a higher like rate. Correlation, not cause.",
                    f"Test {direction} formats for promotional content and compare response.", "low", ev,
                    "Longer videos may attract different viewers; the effect of duration is not isolated.")
        else:
            ins.add("content_intelligence", "No clear relationship between video length and like rate",
                    f"Spearman correlation {rho:.2f} (p = {p:.3f}, n = {len(both)}).", "Not a clustering result.",
                    "Duration alone does not explain response in this sample.",
                    "Do not choose video length on the basis of this sample.", "low", ev, "")


# ------------------------------------------------------------------ channel segments
def channel_insights(ins, channel_table, info, profile, top_n=3):
    n = len(channel_table)
    if info.get("status") != "ok":
        ins.add("content_intelligence", "No distinct channel segments could be established",
                f"{n} candidate channels; {info.get('n_clustered', 0)} had all required features.",
                f"Not established: {info.get('reason')}.",
                "Channels are described individually in the ranking, not as segments.",
                "Do not treat channels as belonging to distinct types on the basis of this run.", "not established",
                [_ev("channels clustered", info.get("n_clustered", 0), n, "channel_clusters.csv")],
                "With a small number of channels, K-Means can only find exploratory groupings.")
        return
    top = channel_table.nsmallest(top_n, "recommendation_rank")
    for _, row in profile.iterrows():
        members = channel_table[channel_table["cluster"] == row["cluster"]]
        names = ", ".join(members.sort_values("recommendation_rank")["channel_title"].head(6))
        top_in = int((top["cluster"] == row["cluster"]).sum())
        ins.add("content_intelligence", f"Channel segment {int(row['cluster'])}: {row['descriptors']}",
                f"{int(row['n'])} channels, e.g. {names}. Median like rate "
                f"{_pct(row.get('median_like_rate'), 2)}, median mean views {row.get('median_mean_views', np.nan):,.0f}.",
                f"Cluster {int(row['cluster'])} of {info['k']} (silhouette {info['silhouette_best']:.2f}, "
                f"permutation p {info['null_p_value']:.2f}, stability ARI {info['stability_ari']:.2f}).",
                f"A descriptive grouping of {n} candidate channels; {top_in} of the top {top_n} recommended "
                "channels belong to it.",
                "Use as a benchmark group when comparing a channel with similar channels.",
                "exploratory", [_ev("channels in cluster", row["n"], n, "channel_cluster_profiles.csv")],
                "Small sample: the grouping may change with different data. Cluster descriptions come from the "
                "feature profile, not from predefined labels.")
    unclustered = channel_table[channel_table["cluster"].isna()]
    if len(unclustered):
        ins.add("data_quality", "Some channels were not clustered",
                "; ".join(f"{r.channel_title} ({r.sentiment_status})" for r in unclustered.itertuples()),
                "Excluded because a required feature was missing (for example too few analysed comments).",
                "Their sentiment is NOT measured and is not treated as neutral.",
                "Review these channels individually.", "not established",
                [_ev("channels not clustered", len(unclustered), n, "channel_clusters.csv")], "")


# ------------------------------------------------------------------ audience needs
def audience_insights(ins, comment_result, difficulty_terms):
    sig, c = comment_result["signals"], comment_result["counts"]
    denom = c["analysable"]
    if denom < 30:
        ins.add("audience_needs", "Too few analysable comments for audience-need analysis",
                f"{denom} analysable comments of {c['total']} collected.", "Not performed.",
                "No claim about repeated viewer needs can be made.", "Collect comments from more videos.", "not established",
                [_ev("analysable comments", denom, c["total"], "comment_signals.csv")], "")
        return
    emitted = 0
    for _, r in sig.iterrows():
        name = r["signal"]
        if name == "content_request" or name not in NEED_ACTIONS or not r["repeated"]:
            continue
        strength = "moderate" if (r["comments"] >= 10 and r["videos"] >= 3) else "low"
        extra = f" Frequent terms in these comments: {difficulty_terms}." \
            if name == "difficulty_understanding" and difficulty_terms else ""
        ins.add("audience_needs", NEED_TITLES[name],
                f"{int(r['comments'])} of {denom} analysable comments ({_pct(r['share_of_analysable'])}) "
                f"across {int(r['videos'])} videos and {int(r['channels'])} channels match this need. "
                f"Examples: {r['examples']}.{extra}",
                "Keyword rule match (transparent), not a cluster.",
                "Several viewers on different videos express this need; it is a recurring need in this sample, "
                "not a measured share of the whole audience.",
                NEED_ACTIONS[name], strength,
                [_ev("matching comments", r["comments"], denom, "comment_signals.csv"),
                 _ev("videos", r["videos"], None, "comment_signals.csv")],
                "Comments are the top relevance-ranked, top-level comments; English-language comments only.")
        emitted += 1
    if emitted == 0:
        ins.add("audience_needs", "No learning need is repeated often enough to report",
                f"Among {denom} analysable comments no need signal reaches the minimum of "
                f"{3} comments on {2} videos.", "Keyword rules; no cluster claim.",
                "Comments in this sample are mostly praise or short reactions.",
                "Do not infer audience needs from this run.", "not established",
                [_ev("analysable comments", denom, c["total"], "comment_signals.csv")], "")


def theme_insights(ins, comment_result):
    info, themes = comment_result["theme_info"], comment_result["themes"]
    if info.get("status") != "ok" or themes.empty:
        ins.add("audience_needs", "No reliable comment themes were found",
                f"{comment_result['counts']['informative']} informative comments were available.",
                f"Not established: {info.get('reason', 'insufficient data')}.",
                "Recurring topics cannot be claimed from this run.", "Rely on the keyword-signal counts instead.",
                "not established", [_ev("informative comments", comment_result["counts"]["informative"], None, "comment_themes.csv")],
                "")
        return
    for _, t in themes.head(3).iterrows():
        ins.add("audience_needs", f"Comment theme {int(t['cluster'])}: {t['top_terms']}",
                f"{int(t['n_comments'])} comments ({_pct(t['share_of_informative'])} of informative comments); "
                f"mean VADER sentiment {t['mean_vader_compound']:.2f}; {_pct(t['question_share'])} are questions.",
                f"Text cluster of {info['k']} (silhouette {info['silhouette_best']:.2f}, permutation p "
                f"{info['null_p_value']:.2f}, stability ARI {info['stability_ari']:.2f}).",
                "A recurring way viewers talk about the topic; see representative comments in comment_themes.csv.",
                "Use the representative comments to understand wording viewers use.", "exploratory",
                [_ev("comments in theme", t["n_comments"], comment_result["counts"]["informative"], "comment_themes.csv")],
                "Text clusters on short comments are weak; praise-dominated themes carry no marketing message.")


def gap_insights(ins, comment_result):
    d = comment_result["demand_supply"]
    if d.empty:
        ins.add("market_opportunity", "No explicit topic requests were found in the comments",
                "No comment matched a 'video/course on <topic>' request pattern.", "Not applicable.",
                "No supply-versus-demand comparison is possible from this run.", "None.", "not established", [], "")
        return
    gaps = d[d["possible_gap"]]
    if gaps.empty:
        ins.add("market_opportunity", "Requested topics are already covered by sampled videos",
                f"{len(d)} requested topics were found; none are both requested at least 2 times and rare "
                "in the sampled videos.", "Not a clustering result.", "No underserved topic is indicated by this sample.",
                "None.", "low", [_ev("requested topics", len(d), None, "demand_vs_supply.csv")], "")
        return
    for _, g in gaps.head(5).iterrows():
        ins.add("market_opportunity", f"Possible underserved topic: '{g['requested_topic']}'",
                f"Requested in {int(g['demand_comments'])} comments on {int(g['demand_videos'])} video(s); "
                f"mentioned in {int(g['supply_videos'])} of {int(g['supply_sample_size'])} sampled videos "
                f"({_pct(g['supply_share'])}).", "Not a clustering result (request-phrase match).",
                "A hypothesis: viewers ask for this and the sampled videos rarely cover it. The sample is small and "
                "not the whole market.",
                "Search YouTube and course platforms for this topic directly before investing in content; if "
                "supply is genuinely thin, test a short piece of content on it.", "exploratory",
                [_ev("demand comments", g["demand_comments"], None, "demand_vs_supply.csv"),
                 _ev("supply share", g["supply_share"], g["supply_sample_size"], "demand_vs_supply.csv")],
                "Few requests; wording differences can hide existing coverage.")


# ------------------------------------------------------------------ anomalies and benchmarks
def anomaly_insights(ins, anomalies, diag):
    if anomalies.empty:
        ins.add("anomalies", "No unusual channels or videos were detected",
                f"Tested {diag.get('channel_n', 0)} channels and {diag.get('video_n', 0)} videos.",
                "Robust z-score rule (|z| > 3.5).", "Nothing stands out relative to peers in this sample.",
                "None.", "not established", [], "")
        return
    for level in ("channel", "video"):
        sub = anomalies[anomalies["level"] == level]
        if sub.empty:
            continue
        top = sub.head(5)
        items = "; ".join(f"{r.title} ({r.direction} {r.feature}, z = {r.modified_z:.1f})"
                          if pd.notna(r.modified_z) else f"{r.title} ({r.direction})" for r in top.itertuples())
        ins.add("anomalies", f"Unusual {level}s to investigate",
                f"{len(sub)} flagged observations; most extreme: {items}.", "Anomaly detection (not clustering).",
                "These differ from their peers; the reason is unknown. They may reflect a successful format, a "
                "popular creator, a viral event, or data quirks.",
                f"Review these {level}s manually before drawing conclusions; use a low-engagement outlier as a "
                "warning sign to check, not as something to copy.", "exploratory",
                [_ev("flagged observations", len(sub), None, "anomalies.csv")],
                "Age, language, topic popularity and external promotion are not controlled.")


def recommendation_link_insight(ins, channel_table):
    top = channel_table.nsmallest(3, "recommendation_rank")
    if top.empty:
        return
    med = channel_table["like_rate"].median()
    parts = []
    for r in top.itertuples():
        cl = "not clustered" if pd.isna(r.cluster) else f"segment {int(r.cluster)}"
        parts.append(f"#{int(r.recommendation_rank)} {r.channel_title} (like rate {_pct(r.like_rate, 2)}, {cl})")
    ins.add("recommendation_link", "Where the top-ranked channels sit",
            "; ".join(parts) + f". Median like rate across candidates: {_pct(med, 2)}.",
            "Segments come from channel clustering where available.",
            "The original recommendation score (65% sentiment, 25% engagement, 10% views) is unchanged; this only "
            "adds context about the channels it ranks.",
            "Use the top-ranked channels as examples of what the audience responds well to, then check the "
            "audience-need insights for what to add.", "exploratory",
            [_ev("channels ranked", len(channel_table), None, "recommendations.csv")], "")


def limitations_insight(ins, meta):
    ins.add("data_quality", "What this analysis can and cannot say",
            f"Candidate channels: {meta['n_channels']}; sampled on-topic videos: {meta['n_videos']}; comments "
            f"collected: {meta['n_comments']} (analysable: {meta['n_analysable']}); channels with unmeasured "
            f"sentiment: {meta['n_unmeasured']}.",
            "See the diagnostics in intelligence_summary.json for every skipped analysis and why.",
            "YouTube public data contains no watch time, impressions, click-through, demographics, purchases or "
            "ROI. High engagement does not imply sales.",
            "Treat every action as a hypothesis to test with your own campaign data.", "not established",
            [_ev("candidate channels", meta["n_channels"], None, "recommendations.csv")],
            "Sample = what YouTube's relevance ranking returned for one query at one time.")


def generate(ctx):
    """ctx: dict produced by intelligence.pipeline. Returns the insights DataFrame."""
    ins = Insights()
    content_insights(ins, ctx.get("video_table"), ctx.get("video_info"), ctx.get("video_profile", pd.DataFrame()))
    channel_insights(ins, ctx["channel_table"], ctx["channel_info"], ctx["channel_profile"])
    comments = ctx["comments"]
    sig = comments["signals"]
    diff_terms = ""
    if "difficulty_terms" in comments:
        diff_terms = comments["difficulty_terms"]
    audience_insights(ins, comments, diff_terms)
    theme_insights(ins, comments)
    gap_insights(ins, comments)
    anomaly_insights(ins, ctx["anomalies"], ctx["anomaly_diag"])
    recommendation_link_insight(ins, ctx["channel_table"])
    limitations_insight(ins, ctx["meta"])
    return ins.frame()
