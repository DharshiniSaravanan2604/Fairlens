"""Screening engine: job title + resumes -> scored, ranked, explained DataFrame."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config, extract
from .model import Artifacts
from .preprocess import display_term, preprocess_many
from .titles import TitleResolver


def score_extras(prof: dict, min_years=config.DEFAULT_MIN_EXPERIENCE_YEARS,
                 min_edu=config.DEFAULT_MIN_EDUCATION_LEVEL, pref_certs=config.DEFAULT_PREFERRED_CERTS):
    exp = min(1.0, prof["experience_years"] / min_years) if min_years > 0 else 1.0
    lvl = prof["education_level"]
    edu = 1.0 if lvl >= min_edu else (0.5 if lvl == min_edu - 1 and lvl > 0 else 0.0)
    cert = min(1.0, prof["certifications"] / pref_certs) if pref_certs > 0 else 1.0
    return exp, edu, cert


def final_score(kw, sim, exp, edu, cert, w=config.WEIGHTS):
    return 100 * (w["skill"] * kw + w["similarity"] * sim + w["experience"] * exp
                  + w["education"] * edu + w["certification"] * cert)


class ScreeningEngine:
    def __init__(self, art: Artifacts | None = None):
        self.art = art or Artifacts.load()
        self.resolver = TitleResolver(self.art)

    def resolve_title(self, title):
        return self.resolver.resolve(title)

    def keywords_for(self, category):
        return [display_term(k["term"], self.art.surface) for k in self.art.bank[category]]

    def screen(self, category: str, docs: list[dict], threshold=None, min_years=config.DEFAULT_MIN_EXPERIENCE_YEARS,
               min_edu=config.DEFAULT_MIN_EDUCATION_LEVEL, pref_certs=config.DEFAULT_PREFERRED_CERTS) -> pd.DataFrame:
        art = self.art
        threshold = art.threshold if threshold is None else threshold
        j = art.classes.index(category)
        clean, masked, nmask = preprocess_many([d["text"] for d in docs])
        comp = art.components(clean)
        rows = []
        for i, d in enumerate(docs):
            prof = extract.profile(d["text"])
            exp, edu, cert = score_extras(prof, min_years, min_edu, pref_certs)
            kw, sim, p = comp["kw"][i, j], comp["sim"][i, j], comp["proba"][i]
            fs = final_score(kw, sim, exp, edu, cert)
            hit, miss = art.matched_missing(clean[i], category)
            pj = int(np.argmax(p))
            rows.append({
                "file_name": d["name"], "final_score": round(fs, 1),
                "skill_score": round(kw, 3), "similarity_score": round(sim, 3),
                "experience_score": round(exp, 3), "education_score": round(edu, 3),
                "certification_score": round(cert, 3),
                "category_probability": round(float(p[j]), 3),
                "predicted_category": art.classes[pj], "predicted_confidence": round(float(p[pj]), 3),
                "experience_years": prof["experience_years"], "education": prof["education"],
                "certifications": prof["certifications"],
                "matched_keywords": list(dict.fromkeys(display_term(t, art.surface) for t in hit)),
                "missing_keywords": list(dict.fromkeys(display_term(t, art.surface) for t in miss)),
                "identifiers_masked": nmask[i],
                "category_ok": bool(p[j] >= art.gate),
                "shortlisted": bool(fs >= threshold and p[j] >= art.gate),
            })
        # candidates the KNN classifier associates with the role are ranked first, then by final score
        df = pd.DataFrame(rows).sort_values(["category_ok", "final_score", "skill_score"],
                                            ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df["decision"] = np.where(df["shortlisted"], "Shortlisted", "Not shortlisted")
        return df
