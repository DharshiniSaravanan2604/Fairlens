# Edits needed in the report text (because the user now enters only a job title)

1. **FR2 / 2.3 / 4.8 "Define job requirements":** the HR user enters only the *job title*. The required skills are the 20 keywords
   learned from the dataset for the matched job category (shown on screen). Experience, education and certification requirements use
   defaults (2 years, Bachelor's degree, 2 certifications) that are used only inside the weighted score.
2. **4.7 Dataset:** "Resume Dataset (UpdatedResumeDataSet.csv): 962 rows, 166 unique resumes, 25 job categories (Resume text + Category).
   Duplicates were removed before splitting."
3. **5.4 Database:** `jobs` fields = job_id, user_id, title, matched_category, skills (keywords as JSON), created_at;
   `scores` has an additional `details` column (JSON with matched/missing keywords).
4. **6.6 KNN:** TF-IDF unigrams+bigrams, cosine distance, distance-weighted voting, K = 9 chosen by 3-fold CV on the training split.
5. **6.7 Scoring:** weights unchanged (35/25/20/12/8). Add: "A candidate is shortlisted when the final score reaches the cut-off (30) and the KNN
   probability of the advertised category is at least 0.34; candidates the classifier associates with the job are ranked first."
   Skill and similarity values are divided by the 90th percentile of the same value for in-category training resumes and capped at 1.
6. **7.1 Experimental setup:** run `python benchmark.py` on YOUR laptop and replace the 7.4 times with your numbers
   (the filled report uses the numbers measured in the development environment: one CPU core, 3 GB RAM).
7. **7.5 Discussion:** the sentence "Combining text similarity with explicit checks ... prevents a resume from ranking highly only because it repeats the words
   of the job description" is partly supported: the ablation shows similarity alone and KNN alone rank slightly better than the 5-factor score, and the
   factors that do not depend on the job only help once the KNN category check is applied.
