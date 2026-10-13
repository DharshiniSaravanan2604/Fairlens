"""Map a free-text job title to one of the dataset's job categories."""
from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ALIASES = {
 "Data Science": ["data scientist", "data science", "machine learning engineer", "ml engineer", "ai engineer", "deep learning engineer", "nlp engineer", "data analyst"],
 "HR": ["hr", "human resources", "hr executive", "hr manager", "recruiter", "talent acquisition", "hr generalist", "hr business partner"],
 "Advocate": ["advocate", "lawyer", "legal counsel", "attorney", "legal advisor", "legal associate"],
 "Arts": ["arts", "artist", "fine arts", "art teacher", "creative artist", "illustrator", "arts graduate"],
 "Web Designing": ["web designer", "web designing", "front end developer", "frontend developer", "ui designer", "ui ux designer", "web developer"],
 "Mechanical Engineer": ["mechanical engineer", "mechanical engineering", "design engineer", "production engineer", "maintenance engineer"],
 "Sales": ["sales", "sales executive", "sales manager", "business development executive", "sales associate", "account executive"],
 "Health and fitness": ["health and fitness", "fitness trainer", "personal trainer", "nutritionist", "dietician", "yoga instructor", "gym trainer"],
 "Civil Engineer": ["civil engineer", "civil engineering", "site engineer", "structural engineer", "construction engineer"],
 "Java Developer": ["java developer", "java", "java engineer", "j2ee developer", "spring developer", "java programmer"],
 "Business Analyst": ["business analyst", "ba", "systems analyst", "product analyst", "business systems analyst"],
 "SAP Developer": ["sap developer", "sap", "sap consultant", "abap developer", "sap abap", "sap fico consultant"],
 "Automation Testing": ["automation testing", "automation tester", "test automation engineer", "qa automation engineer", "selenium tester", "sdet"],
 "Electrical Engineering": ["electrical engineer", "electrical engineering", "electrical", "electronics engineer", "power engineer"],
 "Operations Manager": ["operations manager", "operations", "operations executive", "operations head", "supply chain manager"],
 "Python Developer": ["python developer", "python", "python engineer", "django developer", "flask developer", "python programmer"],
 "DevOps Engineer": ["devops engineer", "devops", "site reliability engineer", "sre", "cloud engineer", "build and release engineer"],
 "Network Security Engineer": ["network security engineer", "network engineer", "security engineer", "cyber security analyst", "cybersecurity engineer", "information security analyst", "firewall engineer"],
 "PMO": ["pmo", "project management office", "project manager", "program manager", "pmo analyst", "project coordinator"],
 "Database": ["database", "database administrator", "dba", "sql developer", "database developer", "oracle dba"],
 "Hadoop": ["hadoop", "hadoop developer", "big data engineer", "big data developer", "spark developer"],
 "ETL Developer": ["etl developer", "etl", "data engineer", "informatica developer", "data warehouse developer", "bi developer"],
 "DotNet Developer": ["dotnet developer", "net developer", "dot net developer", "asp net developer", "c sharp developer", "csharp developer"],
 "Blockchain": ["blockchain", "blockchain developer", "smart contract developer", "solidity developer", "ethereum developer", "web3 developer"],
 "Testing": ["testing", "software tester", "manual tester", "qa engineer", "quality analyst", "test engineer", "qa tester"],
}
_SENIORITY = re.compile(r"\b(senior|sr|junior|jr|lead|principal|associate|trainee|intern|fresher|entry level|i|ii|iii)\b")


def norm_title(t: str) -> str:
    t = t.lower().replace("c#", "csharp ").replace(".net", " net ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = _SENIORITY.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


class TitleResolver:
    def __init__(self, art):
        self.art = art
        self.phrases, self.owner = [], []
        for cat in art.classes:
            for p in {cat.lower(), *ALIASES.get(cat, [])}:
                self.phrases.append(norm_title(p))
                self.owner.append(cat)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(self.phrases)
        self.P = self.vec.transform(self.phrases)

    def resolve(self, title: str, top: int = 3) -> dict:
        t = norm_title(title)
        if not t:
            return {"category": None, "confidence": 0.0, "alternatives": []}
        cats = self.art.classes
        score = dict.fromkeys(cats, 0.0)
        # 1) exact alias / contains alias
        for p, c in zip(self.phrases, self.owner):
            if t == p:
                score[c] = max(score[c], 1.0)
            elif len(p) > 2 and (f" {p} " in f" {t} "):
                score[c] = max(score[c], 0.6 + 0.4 * len(p) / max(len(t), 1))
        # 2) fuzzy character n-gram similarity
        sims = (self.vec.transform([t]) @ self.P.T).toarray().ravel()
        for s, c in zip(sims, self.owner):
            score[c] = max(score[c], 0.9 * s)
        # 3) data-driven: title words against the learned keyword banks
        words = set(t.split())
        for c in cats:
            kws = {w for k in self.art.bank[c] for w in k["term"].split()}
            overlap = len(words & kws) / max(len(words), 1)
            score[c] = max(score[c], 0.7 * overlap)
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        best, conf = ranked[0]
        return {"category": best if conf >= 0.35 else None, "best_guess": best,
                "confidence": float(min(conf, 1.0)),
                "alternatives": [(c, float(s)) for c, s in ranked[1:top + 1]]}
