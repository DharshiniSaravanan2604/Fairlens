"""Measure screening time for 50 / 100 / 500 PDF resumes.  Usage: python benchmark.py
Run this on YOUR laptop and quote those numbers in the report (Chapter 7.4)."""
import json, platform, random, tempfile, time
from pathlib import Path
from fairlens import training as T
from fairlens.engine import ScreeningEngine
from fairlens.parser import pdf_to_text
from make_sample_resumes import pdf, NAMES

MANUAL_MIN_PER_RESUME = 5     # assumption used for the "manual" column of the report
if __name__ == "__main__":
    random.seed(1)
    resumes = T.load_dataset().Resume.tolist()
    eng = ScreeningEngine()
    out = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(), "runs": []}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for n in (50, 100, 500):
            files = []
            for i in range(n):
                f = d / f"r{n}_{i}.pdf"
                pdf(f, random.choice(NAMES), resumes[i % len(resumes)])
                files.append(f)
            t0 = time.perf_counter()
            docs = [{"name": f.name, "text": pdf_to_text(f.read_bytes())} for f in files]
            t_parse = time.perf_counter() - t0
            eng.screen("Data Science", docs)
            total = time.perf_counter() - t0
            manual_h = n * MANUAL_MIN_PER_RESUME / 60
            out["runs"].append({"resumes": n, "parse_s": round(t_parse, 1), "total_s": round(total, 1),
                                "total_min": round(total / 60, 2), "manual_hours": round(manual_h, 1),
                                "manual_days_8h": round(manual_h / 8, 1)})
            print(out["runs"][-1])
    Path("models/benchmark.json").write_text(json.dumps(out, indent=2))
