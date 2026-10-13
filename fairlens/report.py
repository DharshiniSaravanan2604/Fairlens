"""Explainable PDF screening report (ReportLab)."""
from __future__ import annotations
import io
from datetime import datetime

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config

NAVY, TEAL, GREEN, RED, GREY = (colors.HexColor(x) for x in ("#1f2a44", "#0f766e", "#15803d", "#b91c1c", "#64748b"))


def summary_text(r, title, category, n_total):
    n_hit, n_all = len(r["matched_keywords"]), len(r["matched_keywords"]) + len(r["missing_keywords"])
    s = (f"This resume was screened for the role <b>{title}</b> (dataset category: {category}). "
         f"It contains {n_hit} of the {n_all} key terms that characterise {category} resumes in the training dataset "
         f"and ranks <b>#{r['rank']} of {n_total}</b> with a final score of <b>{r['final_score']:.1f}/100</b>. ")
    s += (f"The KNN classifier places the resume in <b>{r['predicted_category']}</b> "
          f"(confidence {r['predicted_confidence']*100:.0f}%); the probability for the advertised category is "
          f"{r['category_probability']*100:.0f}%. ")
    gaps = r["missing_keywords"][:5]
    if gaps:
        s += "The main gaps are the absence of: " + ", ".join(gaps) + ". "
    s += "The recommendation is " + ("to <b>shortlist</b> the candidate." if r["shortlisted"] else "<b>not to shortlist</b> the candidate on this evidence.")
    return s


def build_pdf(r: dict, title: str, category: str, n_total: int, threshold: float, gate: float = config.CATEGORY_GATE) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm,
                            title="FairLens Screening Report", author="FairLens")
    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Heading2"], textColor=NAVY, spaceBefore=10, spaceAfter=4)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13.5)
    S = ParagraphStyle("S", parent=B, fontSize=8, textColor=GREY)
    el = []
    banner = Table([[Paragraph("<font color='white' size=17><b>FairLens</b></font><br/><font color='#cbd5e1' size=9>"
                               "Individual Screening Report</font>", B),
                     Paragraph(f"<font color='#cbd5e1' size=8>Generated {datetime.now():%d %b %Y, %H:%M}</font>",
                               ParagraphStyle("R", parent=B, alignment=2))]], colWidths=[110*mm, 64*mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 10),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    el += [banner, Spacer(1, 8)]

    ok = r["shortlisted"]
    head = Table([[Paragraph(f"<b>Role:</b> {title}<br/><b>Dataset category:</b> {category}<br/>"
                             f"<b>Candidate file:</b> {r['file_name']}<br/>"
                             f"<font size=8 color='#64748b'>Personal identifiers masked before scoring "
                             f"({r['identifiers_masked']} items)</font>", B),
                     Paragraph(f"<font size=26 color='#1f2a44'><b>{r['final_score']:.1f}</b></font>"
                               f"<font size=10 color='#64748b'> / 100</font><br/><b>Rank #{r['rank']}</b> of {n_total}<br/>"
                               f"<font color='{'#15803d' if ok else '#b91c1c'}'><b>"
                               f"{'SHORTLISTED' if ok else 'NOT SHORTLISTED'}</b></font>",
                               ParagraphStyle("C", parent=B, alignment=1, leading=20))]],
                 colWidths=[105*mm, 69*mm])
    head.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
                              ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f1f5f9")),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 8),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    el += [head, Paragraph("Why this candidate matches the role", H), Paragraph(summary_text(r, title, category, n_total), B)]

    # factor table + bar drawing
    w = config.WEIGHTS
    factors = [("Skill match (dataset keywords)", r["skill_score"], w["skill"]),
               ("Text similarity (TF-IDF cosine)", r["similarity_score"], w["similarity"]),
               ("Experience", r["experience_score"], w["experience"]),
               ("Education", r["education_score"], w["education"]),
               ("Certifications", r["certification_score"], w["certification"])]
    rows = [["Factor", "Score (0-1)", "Weight", "Points", ""]]
    for name, v, wt in factors:
        d = Drawing(60*mm, 6)
        d.add(Rect(0, 0, 60*mm, 6, fillColor=colors.HexColor("#e2e8f0"), strokeColor=None))
        d.add(Rect(0, 0, 60*mm * max(0.0, min(1.0, v)), 6, fillColor=TEAL, strokeColor=None))
        rows.append([name, f"{v:.2f}", f"{wt*100:.0f}%", f"{100*wt*v:.1f}", d])
    rows.append(["Final score", "", "100%", f"{r['final_score']:.1f}", ""])
    t = Table(rows, colWidths=[56*mm, 20*mm, 16*mm, 16*mm, 66*mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                           ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#cbd5e1")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (3, -1), "CENTER"),
                           ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f1f5f9"))]))
    el += [Paragraph("Score breakdown", H), t]

    kw = Table([[Paragraph(f"<b><font color='#15803d'>Matched keywords ({len(r['matched_keywords'])})</font></b><br/>"
                           + (", ".join(r["matched_keywords"]) or "none"), B),
                 Paragraph(f"<b><font color='#b91c1c'>Missing keywords ({len(r['missing_keywords'])})</font></b><br/>"
                           + (", ".join(r["missing_keywords"]) or "none"), B)]], colWidths=[87*mm, 87*mm])
    kw.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (0, 0), 0.5, GREEN),
                            ("BOX", (1, 0), (1, 0), 0.5, RED), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    el += [Paragraph("Keyword evidence", H), kw]

    info = Table([["Predicted job category (KNN)", f"{r['predicted_category']} ({r['predicted_confidence']*100:.0f}%)"],
                  ["Probability of advertised category", f"{r['category_probability']*100:.0f}%"],
                  ["Estimated experience", f"{r['experience_years']} years"],
                  ["Highest qualification found", r["education"]],
                  ["Certifications found", str(r["certifications"])],
                  ["Shortlist rule", f"score >= {threshold:.0f} and category probability >= {gate:.2f}"]],
                 colWidths=[70*mm, 104*mm])
    info.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                              ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1"))]))
    el += [Paragraph("Extracted profile", H), info, Spacer(1, 10),
           Paragraph("This report supports, and does not replace, the judgement of the hiring team. Scores are produced by the "
                     "same weighted rules for every resume; the final hiring decision must be taken by a human.", S)]
    doc.build(el)
    return buf.getvalue()
