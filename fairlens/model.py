"""Model artifacts: TF-IDF + KNN + dataset keyword bank + category profiles."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from . import config

# generic resume words that are frequent but say nothing about a job role
KEYWORD_STOP = set("""
experience exprience month months year years company details description project projects responsibilities
responsibility skill skills work worked working team client clients using used use good knowledge strong ability
etc less than present current jan feb mar apr may jun jul aug sep oct nov dec name maharashtra pune mumbai
bengaluru bangalore hyderabad chennai delhi india ltd limited pvt private technologies technology solutions
services education detail college university school board percentage cgpa university pune student studied
duration role roles responsible various related including include includes like also well based new
time like year month level data business management system systems process processes application applications
kpit aissms savitribai phule hsc ssc mid co ms db js tech given existing national public inter positive attitude
accept accepted enthusiast enthusiastic learner ready errors free proven accurate established interpersonal english
challenges challenge player duties suggesting minimize measures command parties paper sessions raw open extract
bot pivot opening periodic payment return queries query tapping failures high secondary secondary school
done make making ensure ensuring handling handled maintain maintaining provide provided providing support supported
good excellent hard team player skills lead leading led
measure party tap failure duty suggest session error challenge currently selected last existing given
implementation language meet office distance word cause leader january known prompt incorporated integrated
last formats format institute computer proficiency tool tools
""".split())


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)


def make_knn(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance", algorithm="brute")


def build_keyword_bank(texts, labels, classes, top_n=config.TOP_KEYWORDS):
    """For each category pick the terms that most commonly appear in that category's
    resumes AND are rare in other categories (coverage x log-lift)."""
    labels = np.asarray(labels)
    cv = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True,
                         token_pattern=r"\b[a-z][a-z0-9]+\b")
    B = cv.fit_transform(texts).tocsr()
    terms = cv.get_feature_names_out()
    bad = np.array([any(w in KEYWORD_STOP for w in t.split()) for t in terms])
    bank = {}
    for c in classes:
        m = labels == c
        cov_c = np.asarray(B[m].mean(axis=0)).ravel()
        cov_o = np.asarray(B[~m].mean(axis=0)).ravel()
        lift = (cov_c + 0.01) / (cov_o + 0.01)
        score = cov_c * np.log(np.maximum(lift, 1.0001))
        for min_cov, min_lift in [(0.4, 4), (0.3, 3), (0.2, 2), (0.1, 1.5), (0.0, 1.0)]:
            ok = (~bad) & (cov_c >= min_cov) & (lift >= min_lift)
            if ok.sum() >= top_n:
                break
        order = np.argsort(-(score * ok))
        chosen = []
        for i in order:
            if not ok[i] or len(chosen) >= top_n:
                continue
            t = terms[i]
            # skip a unigram already contained in a chosen bigram (and vice-versa)
            tw = set(t.split())
            if any((t in s.split()) or (s in t.split()) or set(s.split()) == tw
                   for s in (terms[j] for j in chosen)):
                continue
            chosen.append(i)
        w = cov_c[chosen]
        w = w / w.sum()
        bank[c] = [{"term": terms[i], "weight": float(wi), "coverage": float(cov_c[i])}
                   for i, wi in zip(chosen, w)]
    return bank


@dataclass
class Artifacts:
    vectorizer: TfidfVectorizer
    knn: KNeighborsClassifier
    classes: list
    centroids: np.ndarray
    bank: dict
    kw_p90: np.ndarray
    sim_p90: np.ndarray
    surface: dict
    threshold: float = config.SHORTLIST_THRESHOLD
    gate: float = config.CATEGORY_GATE
    meta: dict = field(default_factory=dict)

    # ---- keyword matching
    def _kw_coverage(self, clean_texts):
        padded = [f" {t} " for t in clean_texts]
        cov = np.zeros((len(padded), len(self.classes)))
        for j, c in enumerate(self.classes):
            for kw in self.bank[c]:
                needle = f" {kw['term']} "
                for i, p in enumerate(padded):
                    if needle in p:
                        cov[i, j] += kw["weight"]
        return cov

    def components(self, clean_texts):
        """Return raw+normalised keyword, similarity and KNN-probability matrices (n x C)."""
        X = self.vectorizer.transform(clean_texts)
        sim = np.asarray(X @ self.centroids.T)
        proba = self.knn.predict_proba(X)
        kw = self._kw_coverage(clean_texts)
        return {
            "kw": np.clip(kw / self.kw_p90[None, :], 0, 1),
            "sim": np.clip(sim / self.sim_p90[None, :], 0, 1),
            "proba": proba, "kw_raw": kw, "sim_raw": sim,
        }

    def matched_missing(self, clean_text, category):
        padded = f" {clean_text} "
        hit, miss = [], []
        for kw in self.bank[category]:
            (hit if f" {kw['term']} " in padded else miss).append(kw["term"])
        return hit, miss

    # ---- persistence
    def save(self):
        config.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self, config.MODEL_FILE, compress=3)
        readable = {c: [{"keyword": k["term"], "weight": round(k["weight"], 4),
                         "coverage": round(k["coverage"], 3)} for k in v] for c, v in self.bank.items()}
        config.KEYWORD_FILE.write_text(json.dumps(readable, indent=2))

    @staticmethod
    def load() -> "Artifacts":
        return joblib.load(config.MODEL_FILE)


def fit_artifacts(train_clean, train_y, k: int, surface_counter: Counter | None = None) -> Artifacts:
    train_y = np.asarray(train_y)
    vec = make_vectorizer()
    X = vec.fit_transform(train_clean)
    knn = make_knn(k).fit(X, train_y)
    classes = list(knn.classes_)
    cent = np.vstack([np.asarray(X[train_y == c].mean(axis=0)).ravel() for c in classes])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    bank = build_keyword_bank(list(train_clean), train_y, classes)
    art = Artifacts(vec, knn, classes, cent, bank, np.ones(len(classes)), np.ones(len(classes)), {})
    # calibration: 90th percentile of in-category scores = "a very good resume of this category"
    comp = art.components(list(train_clean))
    for j, c in enumerate(classes):
        m = train_y == c
        art.kw_p90[j] = max(np.percentile(comp["kw_raw"][m, j], 90), 0.10)
        art.sim_p90[j] = max(np.percentile(comp["sim_raw"][m, j], 90), 0.05)
    if surface_counter:
        best = {}
        for (lemma, w), n in surface_counter.items():
            if lemma not in best or n > best[lemma][1]:
                best[lemma] = (w, n)
        art.surface = {l: w for l, (w, _) in best.items() if l != w}
    return art
