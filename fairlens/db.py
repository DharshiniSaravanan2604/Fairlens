"""SQLite storage: hr_users, jobs, candidates, scores, reports (Chapter 5.4)."""
import hashlib, hmac, json, os, sqlite3
from contextlib import contextmanager
from datetime import datetime

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS hr_users(user_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE, password_hash TEXT);
CREATE TABLE IF NOT EXISTS jobs(job_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, title TEXT,
    matched_category TEXT, skills TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS candidates(candidate_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, file_name TEXT,
    predicted_category TEXT, extracted_text TEXT);
CREATE TABLE IF NOT EXISTS scores(score_id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER, skill_score REAL,
    similarity_score REAL, experience_score REAL, education_score REAL, certification_score REAL, final_score REAL,
    details TEXT);
CREATE TABLE IF NOT EXISTS reports(report_id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER, report_path TEXT, created_at TEXT);
"""


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)


def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + "$" + h.hex()


def register(name, email, password):
    if len(password) < 6:
        return False, "Password must have at least 6 characters."
    try:
        with conn() as c:
            c.execute("INSERT INTO hr_users(name,email,password_hash) VALUES(?,?,?)",
                      (name.strip(), email.strip().lower(), _hash(password)))
        return True, "Account created. You can log in now."
    except sqlite3.IntegrityError:
        return False, "This e-mail is already registered."


def login(email, password):
    with conn() as c:
        row = c.execute("SELECT * FROM hr_users WHERE email=?", (email.strip().lower(),)).fetchone()
    if not row:
        return None
    salt_hex, h = row["password_hash"].split("$")
    ok = hmac.compare_digest(_hash(password, bytes.fromhex(salt_hex)).split("$")[1], h)
    return dict(row) if ok else None


def save_screening(user_id, title, category, keywords, df, texts):
    """Persist one screening run. Returns job_id."""
    with conn() as c:
        job_id = c.execute("INSERT INTO jobs(user_id,title,matched_category,skills,created_at) VALUES(?,?,?,?,?)",
                           (user_id, title, category, json.dumps(keywords), datetime.now().isoformat(timespec="seconds"))).lastrowid
        for _, r in df.iterrows():
            cid = c.execute("INSERT INTO candidates(job_id,file_name,predicted_category,extracted_text) VALUES(?,?,?,?)",
                            (job_id, r.file_name, r.predicted_category, texts.get(r.file_name, ""))).lastrowid
            details = {k: (v if not hasattr(v, "item") else v.item()) for k, v in r.to_dict().items()}
            c.execute("INSERT INTO scores(candidate_id,skill_score,similarity_score,experience_score,education_score,"
                      "certification_score,final_score,details) VALUES(?,?,?,?,?,?,?,?)",
                      (cid, r.skill_score, r.similarity_score, r.experience_score, r.education_score,
                       r.certification_score, r.final_score, json.dumps(details, default=str)))
    return job_id


def list_jobs(user_id):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT job_id,title,matched_category,created_at,(SELECT COUNT(*) FROM candidates WHERE job_id=jobs.job_id) AS n "
            "FROM jobs WHERE user_id=? ORDER BY job_id DESC LIMIT 20", (user_id,))]


def load_job(job_id):
    import pandas as pd
    with conn() as c:
        job = dict(c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        rows = c.execute("SELECT s.details FROM scores s JOIN candidates c ON c.candidate_id=s.candidate_id "
                         "WHERE c.job_id=?", (job_id,)).fetchall()
    df = pd.DataFrame([json.loads(r["details"]) for r in rows]).sort_values("rank").reset_index(drop=True)
    return job, df


def record_report(file_name, job_id, path):
    with conn() as c:
        row = c.execute("SELECT candidate_id FROM candidates WHERE job_id=? AND file_name=?", (job_id, file_name)).fetchone()
        if row:
            c.execute("INSERT INTO reports(candidate_id,report_path,created_at) VALUES(?,?,?)",
                      (row["candidate_id"], str(path), datetime.now().isoformat(timespec="seconds")))
