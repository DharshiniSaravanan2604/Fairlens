"""Train the FairLens model end-to-end and write models/*.  Usage:  python train_model.py"""
import json, time
import numpy as np
from fairlens import config
from fairlens.model import fit_artifacts
from fairlens.titles import TitleResolver
from fairlens import training as T

t0 = time.time()
raw = T.load_dataset(dedupe=False)
df = T.load_dataset()
print(f"Dataset rows: {len(raw)}  unique: {len(df)}  categories: {df.Category.nunique()}")
df, surface = T.preprocess_dataset(df)
train, test = T.split(df)
print(f"train={len(train)}  test={len(test)}")

k, ktab = T.tune_knn(train.clean, train.Category)
print(ktab.to_string(index=False)); print("Selected K =", k)

art = fit_artifacts(train.clean, train.Category, k, surface)
res = T.evaluate_classifier(art, test.clean, test.Category)
print({m: round(res[m] * 100, 2) for m in ["accuracy", "precision", "recall", "f1"]})
cv = T.cv_classifier(df, k); print("5-fold CV:", cv)

rank_tab, sl = T.pipeline_cv(df, k)
print(rank_tab.round(3).to_string())
best = sl.sort_values(["f1", "precision"], ascending=False).iloc[0]
print("Best shortlist rule:\n", best.to_dict())

# final model: fitted on the training split for honest test numbers, then the threshold is stored
art.threshold, art.gate = float(best.threshold), float(best.gate)
art.meta = {"k": k, "train_size": len(train), "test_size": len(test)}
art.save()
acc_title, ttab = T.title_test(TitleResolver(art), T.TITLE_CASES)
print("Title resolver accuracy:", acc_title)

metrics = {"unique_resumes": len(df), "raw_rows": len(raw), "categories": df.Category.nunique(), "k": k,
           "k_table": ktab.to_dict("records"),
           "holdout": {m: res[m] for m in ["accuracy", "precision", "recall", "f1"]}, "cv": cv,
           "ranking": rank_tab.to_dict("index"), "shortlist_best": best.to_dict(),
           "title_accuracy": acc_title, "train_seconds": round(time.time() - t0, 1)}
config.METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=float))
print("Saved model to", config.MODEL_FILE)
