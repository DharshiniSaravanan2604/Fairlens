"""Create demo resume PDFs from the HELD-OUT test split (resumes the model never saw in training).
Fictional names/e-mails/phones are added so that identifier masking can be demonstrated."""
import html, random
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from fairlens import training as T, config

OUT = Path(__file__).parent / "sample_resumes"
NAMES = ["Priya Raman", "Karthik Subramanian", "Meera Nair", "Arjun Menon", "Divya Lakshmi", "Rahul Verma",
         "Sneha Iyer", "Vikram Reddy", "Anitha Kumar", "Suresh Babu", "Nisha Patel", "Harish Chandran",
         "Lavanya Murali", "Gokul Krishnan", "Pooja Sharma", "Naveen Raj"]
WANT = {"Data Science": 3, "Python Developer": 2, "Java Developer": 2, "HR": 2, "ETL Developer": 1,
        "Hadoop": 1, "Web Designing": 1, "Testing": 1, "Database": 1, "Sales": 1, "DevOps Engineer": 1}


def pdf(path, name, body):
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Title"], fontSize=18, alignment=0, spaceAfter=2)
    b = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)
    first = name.split()[0].lower()
    el = [Paragraph(name, h),
          Paragraph(f"{first}.{name.split()[-1].lower()}@example.com | +91 9{random.randint(100000000, 999999999)} | Gender: {random.choice(['Male', 'Female'])}", b),
          Spacer(1, 6)]
    for line in body.replace("\r", "").split("\n"):
        if line.strip():
            el.append(Paragraph(html.escape(line.strip()), b))
    doc.build(el)


if __name__ == "__main__":
    random.seed(7)
    df = T.load_dataset()
    _, test = T.split(df)
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.pdf"):
        f.unlink()
    names = NAMES[:]
    random.shuffle(names)
    n = 0
    for cat, k in WANT.items():
        for _, row in test[test.Category == cat].head(k).iterrows():
            n += 1
            pdf(OUT / f"Resume_{n:02d}.pdf", names.pop(), row.Resume)
    print("Created", n, "sample resumes in", OUT)
    print(test.Category.value_counts().to_string())
