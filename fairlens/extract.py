"""Rule-based information extraction: experience, education, certifications."""
from __future__ import annotations
import re

EDU_PATTERNS = [
    (5, "PhD",        re.compile(r"\b(?:Ph\.?\s?D|Doctorate)\b", re.I)),
    (4, "Master's",   re.compile(r"\b(?:M\.?\s?Tech|M\.E\.?|M\.?Sc\.?|MBA|MCA|M\.?Com\.?|M\.A\.?|Masters?)\b|\bME\b")),
    (3, "Bachelor's", re.compile(r"\b(?:B\.?\s?Tech|B\.E\.?|B\.?Sc\.?|BCA|B\.?Com\.?|BBA|B\.A\.?|Bachelors?|Graduate|Graduation)\b|\bBE\b|\bB\.?Tech\b")),
    (2, "Diploma",    re.compile(r"\bdiploma\b", re.I)),
    (1, "12th / HSC", re.compile(r"\b(?:12th|HSC|higher secondary|intermediate)\b", re.I)),
]
EDU_NAMES = {5: "PhD", 4: "Master's", 3: "Bachelor's", 2: "Diploma", 1: "12th / HSC", 0: "Not found"}

CERT_PATTERNS = re.compile(
    r"\b(?:PMP|CCNA|CCNP|CISSP|CEH|ITIL|PRINCE2|OCJP|OCA|OCP|Six Sigma|Scrum Master)\b|certif\w*", re.I)

_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I)
_ENTRY = re.compile(r"(?:experience|exprience)\s*[-:\u2013]?\s*(?:less than\s*)?(\d+(?:\.\d+)?)\s*(months?|years?)", re.I)
_RANGE = re.compile(r"((?:19|20)\d{2})\s*(?:-|\u2013|to)\s*((?:19|20)\d{2}|present|current|till date|date)", re.I)


def extract_experience(text: str, this_year: int = 2026) -> float:
    stated = [float(m.group(1)) for m in _YEARS.finditer(text) if float(m.group(1)) <= 40
              and re.search(r"exp", text[max(0, m.start() - 60): m.end() + 60], re.I)]
    entries = 0.0
    for m in _ENTRY.finditer(text):
        v = float(m.group(1))
        entries += v / 12 if m.group(2).lower().startswith("month") else v
    spans = []
    for m in _RANGE.finditer(text):
        a = int(m.group(1))
        b = this_year if not m.group(2)[0].isdigit() else int(m.group(2))
        if 0 <= b - a <= 40 and a >= 1990:
            spans.append((a, b))
    merged, total = [], 0
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    total = sum(b - a for a, b in merged)
    # education date ranges are also year ranges; keep the estimate conservative
    ranges = max(0.0, total - 4) if total > 4 else 0.0
    return round(min(40.0, max(max(stated, default=0.0), entries, ranges)), 1)


def extract_education(text: str) -> tuple[int, str]:
    best = 0
    for level, _name, pat in EDU_PATTERNS:
        if pat.search(text):
            best = max(best, level)
    return best, EDU_NAMES[best]


def extract_certifications(text: str) -> int:
    found = {m.group(0).lower() for m in CERT_PATTERNS.finditer(text)}
    lines = {ln.strip().lower()[:80] for ln in text.splitlines() if re.search(r"certif", ln, re.I)}
    return min(10, max(len(found - {"certification", "certifications", "certificate", "certificates"}), len(lines)))


def profile(text: str) -> dict:
    lvl, name = extract_education(text)
    return {"experience_years": extract_experience(text), "education_level": lvl,
            "education": name, "certifications": extract_certifications(text)}
