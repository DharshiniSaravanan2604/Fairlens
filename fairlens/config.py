"""Central configuration for FairLens."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "UpdatedResumeDataSet.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "fairlens_model.joblib"
KEYWORD_FILE = MODEL_DIR / "keyword_bank.json"
METRICS_FILE = MODEL_DIR / "metrics.json"
DB_PATH = ROOT / "fairlens.db"
REPORT_DIR = ROOT / "reports_out"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Weighted scoring model (Chapter 6.7 of the report). Must sum to 1.
WEIGHTS = {
    "skill": 0.35,          # dataset keywords found in resume
    "similarity": 0.25,     # TF-IDF cosine similarity with the category profile
    "experience": 0.20,     # years of experience vs. required (default below)
    "education": 0.12,      # highest qualification vs. required (default below)
    "certification": 0.08,  # certifications found vs. preferred (default below)
}

# Defaults used because the HR user only types a job title.
DEFAULT_MIN_EXPERIENCE_YEARS = 2.0
DEFAULT_MIN_EDUCATION_LEVEL = 3          # 3 = Bachelor's
DEFAULT_PREFERRED_CERTS = 2

TOP_KEYWORDS = 20            # keywords searched per job category
SHORTLIST_THRESHOLD = 55.0   # overwritten by the value tuned in training
CATEGORY_GATE = 0.20         # min KNN probability for the job category
