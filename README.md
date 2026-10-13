# FairLens - AI-Powered Resume Screening

The HR user enters **only a job title** and uploads **PDF resumes**. The model maps the title to a job category
learned from the dataset, searches every resume for that category's dataset-derived keywords, scores and ranks the
candidates, shortlists them and writes an explainable PDF report for each.

## Folder map (what to show your mentor)
| Item | File / folder |
|---|---|
| Dataset | `data/UpdatedResumeDataSet.csv` (962 rows, 166 unique, 25 categories) |
| ML training code, step by step with plots (Colab) | `notebooks/FairLens_Training_Colab.ipynb` |
| ML code (modules) | `fairlens/preprocess.py`, `model.py` (TF-IDF, KNN, keyword bank), `titles.py`, `engine.py`, `training.py` |
| One-command training | `train_model.py` -> writes `models/` |
| Learned keywords per job (readable) | `models/keyword_bank.json` |
| Metrics | `models/metrics.json`, plots in `analysis/` |
| Front end (Streamlit) | `app.py`, `fairlens/db.py`, `fairlens/report.py`, `fairlens/parser.py` |
| Demo resumes (held-out test split) | `sample_resumes/` |
| Timing | `benchmark.py` -> `models/benchmark.json` |

## Run in VS Code
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
python train_model.py              # optional: models/ is already included
streamlit run app.py
```
Open http://localhost:8501 -> *Create account* -> log in -> type `Data Scientist` -> upload the PDFs from `sample_resumes/` -> *Screen resumes*.

## Run the training in Google Colab
1. Zip this folder as `FairLens.zip`. 2. Open `notebooks/FairLens_Training_Colab.ipynb` in Colab.
3. *Runtime > Run all*, upload the zip when asked. 4. At the end `fairlens_models.zip` is downloaded; unzip it into `FairLens/models/`.

## How the score works
`Final = 100 x (0.35 Skill + 0.25 Similarity + 0.20 Experience + 0.12 Education + 0.08 Certifications)`
* **Skill** - share of the 20 dataset keywords of the job category found in the resume (weighted by how common each is in that category).
* **Similarity** - TF-IDF cosine similarity between the resume and the category profile (mean vector of that category's training resumes).
* **Experience / Education / Certifications** - extracted from the resume with rules; because only a title is entered, defaults are used
  (2 years, Bachelor's, 2 certifications; the minimum experience can be changed under *Advanced settings*).
* **KNN category check** - a candidate is shortlisted only if `score >= cut-off` and the KNN probability for the job category `>= gate`
  (cut-off and gate are tuned by cross-validation and stored in the model). Candidates the classifier associates with the job are ranked first.
* Names, e-mails, phones, URLs, DOB, gender and addresses are masked before scoring.

## Honest notes
* The CSV contains 796 duplicate rows; they are removed before splitting. Keeping them gives ~99.5 % accuracy that is pure leakage.
* On the unique data KNN reaches ~76 % hold-out / cross-validated accuracy; classes are tiny (3-13 resumes).
* Demo resumes were generated from the held-out test split with fictional names/e-mails. Use real resumes for real screening.
* The tool supports the recruiter; it must not replace the human decision.
