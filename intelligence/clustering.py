"""Clustering utilities shared by channel, video and comment analyses.

Methodological choices (all reported in the output):
  * numeric features: drop low-coverage / constant features, use complete cases only, z-score, clip at +/-3
  * k chosen by silhouette among k >= 2 with no tiny clusters; elbow (inertia) is computed and reported
  * if the best silhouette is too low, the result is flagged 'no_structure' and must not support claims
  * stability = mean adjusted Rand index over subsamples; agreement with Ward hierarchical clustering
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

SEED = 42
N_INIT = 25
MIN_COVERAGE = 0.8
Z_CLIP = 3.0

# Kaufman & Rousseeuw guideline for dense numeric data; much lower values are normal for sparse text.
NUMERIC_THRESHOLDS = [(0.70, "strong"), (0.50, "reasonable"), (0.25, "weak")]
TEXT_THRESHOLDS = [(0.20, "strong"), (0.10, "reasonable"), (0.03, "weak")]


def structure_label(silhouette, text=False):
    for threshold, label in (TEXT_THRESHOLDS if text else NUMERIC_THRESHOLDS):
        if silhouette > threshold:
            return label
    return "none"


def usable_features(df, features, min_coverage=MIN_COVERAGE):
    """Keep features with enough coverage and some variation. Returns (kept, {dropped: reason})."""
    kept, dropped = [], {}
    for f in features:
        if f not in df:
            dropped[f] = "not available"
            continue
        coverage = df[f].notna().mean()
        if coverage < min_coverage:
            dropped[f] = f"coverage {coverage:.0%} below {min_coverage:.0%}"
        elif df[f].dropna().nunique() < 2 or df[f].std(ddof=0) < 1e-12:
            dropped[f] = "constant (no information)"
        else:
            kept.append(f)
    return kept, dropped


def elbow_k(ks, inertias):
    """Largest distance from the straight line joining the first and last (k, inertia) points."""
    if len(ks) < 3:
        return None
    x = np.array(ks, float)
    y = np.array(inertias, float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
    # distance of each point from the line (0,1)-(1,0) after normalisation
    dist = np.abs(xn + yn - 1) / np.sqrt(2)
    return int(x[int(np.argmax(dist))])


def _fit_kmeans(X, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)


def _stability(X, labels, k, n_boot=20, frac=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(X)
    size = max(k + 1, int(round(frac * n)))
    for _ in range(n_boot):
        idx = rng.choice(n, size=size, replace=False)
        sub = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1_000_000))).fit(X[idx])
        scores.append(adjusted_rand_score(labels[idx], sub.labels_))
    return float(np.mean(scores))


NULL_RUNS = 30
NULL_ALPHA = 0.05


def null_best_silhouettes(X, k_max, min_cluster_size, runs=NULL_RUNS, renormalize=False, seed=SEED):
    """Best silhouette that the SAME procedure achieves on column-permuted copies of X.

    Permuting each column independently keeps every feature's distribution but removes joint structure,
    so this is the silhouette one expects from data with no real groups.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        if renormalize:
            Xp = normalize(Xp)
        best = -1.0
        for k in range(2, k_max + 1):
            m = KMeans(n_clusters=k, n_init=5, random_state=0).fit(Xp)
            if np.bincount(m.labels_, minlength=k).min() < min_cluster_size:
                continue
            best = max(best, float(silhouette_score(Xp, m.labels_)))
        out.append(best)
    return np.array(out)


def choose_and_fit(X, k_max, min_cluster_size, text=False):
    """Evaluate k = 1..k_max, pick the best silhouette among valid k >= 2. Returns result dict."""
    n = len(X)
    ks, inertias, sil, valid = [], [], {}, {}
    for k in range(1, k_max + 1):
        model = _fit_kmeans(X, k)
        ks.append(k)
        inertias.append(float(model.inertia_))
        if k == 1:
            continue
        sizes = np.bincount(model.labels_, minlength=k)
        if sizes.min() < min_cluster_size:
            sil[k] = None                          # rejected: a cluster is too small to interpret
            continue
        sil[k] = float(silhouette_score(X, model.labels_))
        valid[k] = model
    result = {"n": n, "ks": ks, "inertia": inertias, "silhouette": sil, "elbow_k": elbow_k(ks, inertias)}
    if not valid:
        result.update(status="skipped", reason="no k >= 2 gives clusters of at least "
                                               f"{min_cluster_size} members", k=None)
        return result
    best_k = max(valid, key=lambda k: (round(sil[k], 12), -k))          # ties -> fewer clusters
    model = valid[best_k]
    labels = model.labels_
    structure = structure_label(sil[best_k], text=text)
    null = null_best_silhouettes(X, k_max, min_cluster_size, renormalize=text)
    p_value = float((1 + np.sum(null >= sil[best_k])) / (len(null) + 1))
    if structure != "none" and p_value >= NULL_ALPHA:
        structure = "none"                       # not distinguishable from structureless data
    ward = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    result.update(
        k=best_k, labels=labels, silhouette_best=sil[best_k], structure=structure,
        null_p_value=p_value, null_silhouette_95th=float(np.percentile(null, 95)),
        status="ok" if structure != "none" else "no_structure",
        reason=None if structure != "none" else
        (f"best silhouette {sil[best_k]:.2f} (permutation p = {p_value:.2f}) is not clearly better than "
         "structureless data: no substantial cluster structure"),
        stability_ari=_stability(X, labels, best_k),
        hierarchical_ari=float(adjusted_rand_score(labels, ward)),
        elbow_agrees=(result["elbow_k"] == best_k),
        centers=model.cluster_centers_)
    return result


def cluster_numeric(df, features, id_cols, min_rows, min_cluster_size, k_cap=6, label="rows"):
    """Cluster complete cases of `df` on `features`. Returns (result, assigned_df)."""
    info = {"label": label, "n_input": len(df), "features_requested": list(features)}
    if len(df) < min_rows:
        info.update(status="skipped", reason=f"only {len(df)} {label}; at least {min_rows} are needed",
                    features_used=[], features_dropped={})
        return info, df.assign(cluster=np.nan)
    kept, dropped = usable_features(df, features)
    info.update(features_used=kept, features_dropped=dropped)
    if len(kept) < 2:
        info.update(status="skipped", reason="fewer than 2 usable features")
        return info, df.assign(cluster=np.nan)
    complete = df[kept].notna().all(axis=1)
    sub = df[complete]
    info["n_clustered"] = int(complete.sum())
    info["excluded_incomplete"] = df.loc[~complete, id_cols[0]].tolist() if id_cols else []
    if len(sub) < min_rows:
        info.update(status="skipped", reason=f"only {len(sub)} {label} have all required features; "
                                             f"at least {min_rows} are needed")
        return info, df.assign(cluster=np.nan)
    scaler = StandardScaler()
    X = np.clip(scaler.fit_transform(sub[kept].to_numpy(float)), -Z_CLIP, Z_CLIP)
    k_max = max(2, min(k_cap, len(sub) // 3))
    fit = choose_and_fit(X, k_max, min_cluster_size)
    info.update({k: v for k, v in fit.items() if k not in ("labels", "centers")})
    assigned = df.copy()
    assigned["cluster"] = np.nan
    if fit.get("labels") is not None and fit["status"] == "ok":      # unsupported clusters are never labelled
        assigned.loc[sub.index, "cluster"] = fit["labels"]
        info["centroid_z"] = pd.DataFrame(
            np.vstack([X[fit["labels"] == c].mean(axis=0) for c in range(fit["k"])]),
            columns=kept).round(3).to_dict(orient="index")
    return info, assigned


def describe_clusters(assigned, info, feature_labels, profile_cols):
    """Per-cluster size, median of profile columns (original units) and data-driven descriptors."""
    if "centroid_z" not in info:
        return pd.DataFrame()
    rows = []
    z = pd.DataFrame(info["centroid_z"]).T if isinstance(info["centroid_z"], dict) else None
    z = pd.DataFrame.from_dict(info["centroid_z"], orient="index")
    for c in sorted(assigned["cluster"].dropna().unique()):
        members = assigned[assigned["cluster"] == c]
        desc = []
        for f in z.columns:
            val = z.loc[int(c), f]
            if abs(val) >= 0.5:
                desc.append(f"{'higher' if val > 0 else 'lower'} {feature_labels.get(f, f)}")
        row = {"cluster": int(c), "n": len(members),
               "descriptors": "; ".join(desc) if desc else "close to the overall average on every feature"}
        for col in profile_cols:
            if col in members:
                row[f"median_{col}"] = members[col].median()
        for f in z.columns:
            row[f"z_{f}"] = z.loc[int(c), f]
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ text clustering
def cluster_texts(texts, min_docs, k_max=8, min_cluster_size=5, svd_dims=30, min_df=2, max_df=0.6,
                  ngram_range=(1, 2), stop_words="english"):
    """TF-IDF -> truncated SVD -> L2 normalisation -> K-Means (silhouette on the embedding)."""
    texts = list(texts)
    info = {"n_docs": len(texts)}
    if len(texts) < min_docs:
        info.update(status="skipped", reason=f"only {len(texts)} documents; at least {min_docs} are needed")
        return info, None
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, stop_words=stop_words,
                          sublinear_tf=True)
    try:
        tfidf = vec.fit_transform(texts)
    except ValueError:
        info.update(status="skipped", reason="no usable vocabulary after preprocessing")
        return info, None
    if tfidf.shape[1] < 10:
        info.update(status="skipped", reason=f"vocabulary too small ({tfidf.shape[1]} terms)")
        return info, None
    dims = max(2, min(svd_dims, tfidf.shape[1] - 1, len(texts) - 1))
    emb = normalize(TruncatedSVD(n_components=dims, random_state=SEED).fit_transform(tfidf))
    k_cap = max(2, min(k_max, len(texts) // max(min_cluster_size, 1)))
    fit = choose_and_fit(emb, k_cap, min_cluster_size, text=True)
    info.update({k: v for k, v in fit.items() if k not in ("labels", "centers")})
    info["vocabulary_size"] = int(tfidf.shape[1])
    if fit.get("labels") is None or fit["status"] != "ok":
        return info, None
    labels = fit["labels"]
    terms = np.array(vec.get_feature_names_out())
    clusters = {}
    for c in range(fit["k"]):
        mask = labels == c
        centroid = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top = terms[np.argsort(centroid)[::-1][:8]].tolist()
        center = emb[mask].mean(axis=0)
        order = np.argsort(-(emb[mask] @ center))
        clusters[c] = {"size": int(mask.sum()), "top_terms": top,
                       "representative_idx": np.where(mask)[0][order[:3]].tolist()}
    return info, {"labels": labels, "clusters": clusters}
a