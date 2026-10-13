"""Training + evaluation helpers (used by train_model.py and the Colab notebook)."""
from __future__ import annotations
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from . import config, extract
from .engine import final_score, score_extras
from .model import fit_artifacts, make_knn, make_vectorizer
from .preprocess import preprocess_many

warnings.filterwarnings("ignore")


def load_dataset(path=config.DATA_PATH, dedupe=True) -> pd.DataFrame:
    df = pd.read_csv(path).dropna()
    df["Category"] = df["Category"].str.strip()
    return df.drop_duplicates().reset_index(drop=True) if dedupe else df.reset_index(drop=True)


def preprocess_dataset(df):
    surface = Counter()
    clean, _, _ = preprocess_many(df["Resume"].tolist(), surface)
    out = df.copy()
    out["clean"] = clean
    return out, surface


def split(df):
    return train_test_split(df, test_size=config.TEST_SIZE, stratify=df["Category"],
                            random_state=config.RANDOM_STATE)


def tune_knn(train_clean, y_train, ks=(1, 3, 5, 7, 9), folds=3):
    X = make_vectorizer().fit_transform(train_clean)
    grid = GridSearchCV(make_knn(3), {"n_neighbors": list(ks)}, scoring="accuracy",
                        cv=StratifiedKFold(folds, shuffle=True, random_state=config.RANDOM_STATE))
    grid.fit(X, y_train)
    table = pd.DataFrame({"K": grid.cv_results_["param_n_neighbors"].astype(int),
                          "cv_accuracy": grid.cv_results_["mean_test_score"]})
    return int(grid.best_params_["n_neighbors"]), table


def evaluate_classifier(art, test_clean, y_test):
    X = art.vectorizer.transform(test_clean)
    pred = art.knn.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_test, pred), "precision": p, "recall": r, "f1": f,
        "report": classification_report(y_test, pred, zero_division=0, output_dict=True),
        "confusion": confusion_matrix(y_test, pred, labels=art.classes), "pred": pred,
    }


def cv_classifier(df, k, folds=5):
    accs, f1s = [], []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=config.RANDOM_STATE).split(df, df["Category"]):
        art = fit_artifacts(df.clean.iloc[tr], df.Category.iloc[tr], k)
        pred = art.knn.predict(art.vectorizer.transform(df.clean.iloc[te]))
        accs.append(accuracy_score(df.Category.iloc[te], pred))
        f1s.append(precision_recall_fscore_support(df.Category.iloc[te], pred, average="weighted", zero_division=0)[2])
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)), "folds": folds}


# --------------------------------------------------------------- pipeline CV
VARIANTS = ["Skill (keywords) only", "Similarity only", "KNN only", "Skill + Similarity",
            "FairLens score (5 factors)", "FairLens score + category gate"]


def pipeline_cv(df, k, folds=5, w=config.WEIGHTS, thresholds=range(30, 86, 5), gates=(0.0, 0.1, 0.2, 0.34, 0.5)):
    """Out-of-fold evaluation of the whole screening pipeline.
    For every fold and every job category c: the held-out resumes are screened for job 'c'
    and a resume is relevant iff its true category is c."""
    ex = np.array([score_extras(extract.profile(t)) for t in df.Resume])    # n x 3
    ranks = {v: [] for v in VARIANTS}      # reciprocal rank, hit@1, hit@3 per (fold, category)
    sl = {}                                 # (t, g) -> [tp, fp, fn]
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=config.RANDOM_STATE).split(df, df["Category"]):
        art = fit_artifacts(df.clean.iloc[tr], df.Category.iloc[tr], k)
        comp = art.components(list(df.clean.iloc[te]))
        y = df.Category.iloc[te].to_numpy()
        for j, c in enumerate(art.classes):
            rel = y == c
            if not rel.any():
                continue
            kw, sim, pr = comp["kw"][:, j], comp["sim"][:, j], comp["proba"][:, j]
            full = final_score(kw, sim, ex[te, 0], ex[te, 1], ex[te, 2], w)
            gated = np.where(pr >= art.gate, full, 0)
            scores = {VARIANTS[0]: kw, VARIANTS[1]: sim, VARIANTS[2]: pr,
                      VARIANTS[3]: 0.6 * kw + 0.4 * sim, VARIANTS[4]: full, VARIANTS[5]: gated}
            for v, s in scores.items():
                order = np.argsort(-s, kind="stable")
                first = int(np.where(rel[order])[0][0]) + 1
                ranks[v].append((1 / first, first == 1, first <= 3))
            for t in thresholds:
                for g in gates:
                    pred = (full >= t) & (pr >= g)
                    cell = sl.setdefault((t, g), [0, 0, 0])
                    cell[0] += int((pred & rel).sum()); cell[1] += int((pred & ~rel).sum()); cell[2] += int((~pred & rel).sum())
    rank_tab = pd.DataFrame({v: {"MRR": np.mean([r[0] for r in rs]), "Hit@1": np.mean([r[1] for r in rs]),
                                 "Hit@3": np.mean([r[2] for r in rs])} for v, rs in ranks.items()}).T
    rows = []
    for (t, g), (tp, fp, fn) in sl.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"threshold": t, "gate": g, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return rank_tab, pd.DataFrame(rows)


def title_test(resolver, cases):
    ok = 0
    out = []
    for title, exp in cases:
        got = resolver.resolve(title)
        cat = got["category"] or got.get("best_guess")
        out.append((title, exp, got["category"], round(got["confidence"], 2), cat == exp))
        ok += cat == exp
    return ok / len(cases), pd.DataFrame(out, columns=["title", "expected", "resolved", "confidence", "correct"])


TITLE_CASES = [
    ("Senior Data Scientist", "Data Science"), ("Machine Learning Engineer", "Data Science"),
    ("HR Executive", "HR"), ("Talent Acquisition Specialist", "HR"), ("Corporate Lawyer", "Advocate"),
    ("Senior Java Programmer", "Java Developer"), ("Java Full Stack Developer", "Java Developer"),
    ("Python Backend Developer", "Python Developer"), ("Django Developer", "Python Developer"),
    ("Hadoop Big Data Developer", "Hadoop"), ("Big Data Engineer", "Hadoop"),
    ("Selenium Automation Tester", "Automation Testing"), ("QA Automation Engineer", "Automation Testing"),
    ("Manual QA Tester", "Testing"), ("Software Test Engineer", "Testing"),
    ("Front End Web Designer", "Web Designing"), ("UI/UX Designer", "Web Designing"),
    ("Site Reliability Engineer", "DevOps Engineer"), ("DevOps Engineer", "DevOps Engineer"),
    ("Network Security Engineer", "Network Security Engineer"), ("Cyber Security Analyst", "Network Security Engineer"),
    ("SAP ABAP Consultant", "SAP Developer"), ("ETL Developer", "ETL Developer"), ("Informatica Developer", "ETL Developer"),
    ("Oracle DBA", "Database"), ("SQL Database Developer", "Database"), (".NET Developer", "DotNet Developer"),
    ("C# ASP.NET Developer", "DotNet Developer"), ("Solidity Smart Contract Developer", "Blockchain"),
    ("Blockchain Engineer", "Blockchain"), ("Civil Site Engineer", "Civil Engineer"),
    ("Mechanical Design Engineer", "Mechanical Engineer"), ("Electrical Engineer", "Electrical Engineering"),
    ("Sales Manager", "Sales"), ("Business Development Executive", "Sales"), ("Operations Manager", "Operations Manager"),
    ("Project Manager", "PMO"), ("Business Analyst", "Business Analyst"), ("Personal Fitness Trainer", "Health and fitness"),
    ("Fine Arts Teacher", "Arts"),
]
