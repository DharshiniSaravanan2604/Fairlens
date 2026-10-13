"""Text pre-processing: identifier masking, cleaning, stop-word removal, lemmatisation.

The SAME functions are used for the training dataset and for uploaded resumes so
that the model sees identical features at training time and at screening time.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

# ----------------------------------------------------------------- resources
@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    try:
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("english"))
    except LookupError:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return frozenset(stopwords.words("english"))
        except Exception:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            return frozenset(ENGLISH_STOP_WORDS)


@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser"])
    except OSError:
        print("[FairLens] spaCy model missing -> run: python -m spacy download en_core_web_sm")
        return spacy.blank("en")


# ------------------------------------------------------- identifier masking
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github)\.com/\S+", re.I)
PHONE = re.compile(
    r"(?:\+\d{1,3}[\s-]?)?\b\d{10}\b|(?:\+\d{1,3}[\s-]?)?\b\d{5}[\s-]\d{5}\b|\b\d{3}[\s-]\d{3}[\s-]\d{4}\b")
DOB = re.compile(r"(?:date of birth|d\.?o\.?b\.?)\s*[:\-]?\s*[\w /.\-]{6,20}", re.I)
GENDER = re.compile(r"\b(?:gender|sex)\s*[:\-]?\s*(?:male|female|other)\b|\b(?:male|female)\b", re.I)
ADDRESS = re.compile(r"(?:permanent\s+)?address\s*[:\-].*", re.I)
NAME_LINE = re.compile(r"^[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){0,3}$")


def mask_identifiers(text: str) -> tuple[str, int]:
    """Remove e-mail, phone, URL, DOB, gender, address and the candidate's name.
    Returns (masked_text, number_of_items_masked)."""
    count = 0
    for pat in (EMAIL, URL, PHONE, DOB, ADDRESS, GENDER):
        text, n = pat.subn(" ", text)
        count += n

    # candidate name: (1) first line that looks like a name, (2) spaCy PERSON in the header
    names = set()
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if first and len(first) < 40 and NAME_LINE.match(first) and not any(ch.isdigit() for ch in first):
        names.add(first)
    nlp = get_nlp()
    if "ner" in nlp.pipe_names:
        for ent in nlp(text[:400]).ents:
            if ent.label_ == "PERSON" and len(ent.text) > 2:
                names.add(ent.text.strip())
    for nm in sorted(names, key=len, reverse=True):
        text, n = re.subn(re.escape(nm), " ", text, flags=re.I)
        count += n
    return text, count


# ------------------------------------------------------------------ cleaning
TERM_FIXES = [
    (re.compile(r"\bc\+\+", re.I), " cplusplus "),
    (re.compile(r"\bc#", re.I), " csharp "),
    (re.compile(r"\bvb\.net\b", re.I), " vbnet "),
    (re.compile(r"\basp\.net\b", re.I), " aspnet "),
    (re.compile(r"\.net\b", re.I), " dotnet "),
    (re.compile(r"\bnode\.?js\b", re.I), " nodejs "),
    (re.compile(r"\bms[\s-]?excel\b", re.I), " excel "),
    (re.compile(r"\bpl/sql\b", re.I), " plsql "),
]
DISPLAY_FIX = {"cplusplus": "C++", "csharp": "C#", "vbnet": "VB.NET", "aspnet": "ASP.NET",
               "dotnet": ".NET", "nodejs": "Node.js", "plsql": "PL/SQL"}
ACRONYMS = {"sql", "etl", "sap", "hr", "aws", "api", "html", "css", "xml", "json", "ui", "ux", "qa",
            "sdlc", "erp", "crm", "bi", "oop", "oops", "jsp", "ajax", "rest", "jquery", "ci", "cd",
            "vpn", "tcp", "ip", "lan", "wan", "dns", "ccna", "pmo", "mvc", "hdfs", "hive", "ide",
            "gst", "cad", "plc", "scada", "abap", "fico", "mm", "sd", "bw", "hana"}


def _fix_terms(text: str) -> str:
    text = text.encode("ascii", "ignore").decode()      # drop mojibake bullets etc.
    for pat, rep in TERM_FIXES:
        text = pat.sub(rep, text)
    return text


_TOKEN_OK = re.compile(r"[a-z][a-z0-9]+")


def normalize_many(texts: list[str], surface: Counter | None = None) -> list[str]:
    """Lower-case, strip noise, remove stop-words and lemmatise. Texts must already be masked."""
    stop = get_stopwords()
    nlp = get_nlp()
    disable = ["ner"] if "ner" in nlp.pipe_names else []
    out = []
    for doc in nlp.pipe([_fix_terms(t) for t in texts], batch_size=8, disable=disable):
        toks = []
        for t in doc:
            w = t.text.lower()
            if not _TOKEN_OK.fullmatch(w) or w in stop:
                continue
            lemma = (t.lemma_ or w).lower()
            if not _TOKEN_OK.fullmatch(lemma):
                lemma = w
            if lemma in stop:
                continue
            toks.append(lemma)
            if surface is not None:
                surface[(lemma, w)] += 1
        out.append(" ".join(toks))
    return out


def preprocess_many(raw_texts: list[str], surface: Counter | None = None):
    """raw text -> (clean_texts, masked_texts, n_masked)."""
    masked, counts = [], []
    for t in raw_texts:
        m, n = mask_identifiers(t)
        masked.append(m)
        counts.append(n)
    return normalize_many(masked, surface), masked, counts


def display_term(term: str, surface_map: dict | None = None) -> str:
    """Turn a lemma n-gram back into a readable form (panda -> pandas, csharp -> C#)."""
    words = []
    for w in term.split():
        w = (surface_map or {}).get(w, w)
        if w in DISPLAY_FIX:
            w = DISPLAY_FIX[w]
        elif w in ACRONYMS:
            w = w.upper()
        words.append(w)
    return " ".join(words)
