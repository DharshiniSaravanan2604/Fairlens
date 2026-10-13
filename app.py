"""FairLens - Streamlit front end.   Run:  streamlit run app.py"""
import io
import zipfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fairlens import config, db
from fairlens.engine import ScreeningEngine
from fairlens.parser import ParseError, pdf_to_text
from fairlens.report import build_pdf, summary_text

st.set_page_config(page_title="FairLens - AI Resume Screening", page_icon="🔍", layout="wide")
db.init_db()

st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1250px}
.fl-banner{background:linear-gradient(120deg,#1f2a44,#0f766e);color:#fff;border-radius:14px;padding:20px 26px;margin-bottom:18px}
.fl-banner h1{color:#fff;margin:0;font-size:1.9rem}.fl-banner p{margin:4px 0 0;color:#d1e7e4}
.fl-step{font-weight:700;color:#0f766e;letter-spacing:.04em;font-size:.8rem;text-transform:uppercase;margin-bottom:2px}
.fl-chip{display:inline-block;background:#e6f4f1;color:#0f766e;border:1px solid #b8dfd8;border-radius:999px;padding:2px 11px;margin:3px 4px 0 0;font-size:.82rem}
.fl-chip.miss{background:#fdecec;color:#b91c1c;border-color:#f5c2c2}
.fl-match{background:#fff;border:1px solid #d6dee8;border-left:5px solid #0f766e;border-radius:10px;padding:12px 16px;margin:8px 0 4px}
.fl-login{max-width:430px;margin:6vh auto 0}
div[data-testid="stMetric"]{background:#fff;border:1px solid #d6dee8;border-radius:12px;padding:12px 16px}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading FairLens model...")
def get_engine():
    return ScreeningEngine()


def chips(items, cls=""):
    return "".join(f"<span class='fl-chip {cls}'>{i}</span>" for i in items)


# ------------------------------------------------------------------ login
def login_page():
    st.markdown("<div class='fl-login'>", unsafe_allow_html=True)
    st.markdown("<div class='fl-banner'><h1>🔍 FairLens</h1><p>AI-powered resume screening for HR teams</p></div>",
                unsafe_allow_html=True)
    t_login, t_reg = st.tabs(["Log in", "Create account"])
    with t_login:
        with st.form("login"):
            email = st.text_input("E-mail", placeholder="hr@company.com")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", use_container_width=True):
                user = db.login(email, pw)
                if user:
                    st.session_state.user = user
                    st.rerun()
                st.error("Invalid e-mail or password.")
    with t_reg:
        with st.form("register"):
            name = st.text_input("Full name")
            email = st.text_input("Work e-mail")
            pw = st.text_input("Password (min. 6 characters)", type="password")
            if st.form_submit_button("Create account", use_container_width=True):
                ok, msg = db.register(name, email, pw)
                (st.success if ok else st.error)(msg)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------ results views
def results_view(res, engine):
    df = res["df"].copy()
    df["shortlisted"] = (df.final_score >= res["threshold"]) & (df.category_probability >= engine.art.gate)
    df["decision"] = df.shortlisted.map({True: "Shortlisted", False: "Not shortlisted"})
    n_short = int(df.shortlisted.sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resumes screened", len(df))
    c2.metric("Shortlisted", n_short)
    c3.metric("Top score", f"{df.final_score.max():.1f}")
    c4.metric("Average score", f"{df.final_score.mean():.1f}")
    if res.get("errors"):
        with st.expander(f"⚠️ {len(res['errors'])} file(s) could not be read"):
            for f, e in res["errors"]:
                st.write(f"**{f}** - {e}")

    t_rank, t_cmp, t_rep = st.tabs(["🏆 Candidate Ranking", "⚖️ Compare Candidates", "📄 Screening Reports"])

    with t_rank:
        view = df[["rank", "file_name", "final_score", "skill_score", "similarity_score", "category_probability",
                   "predicted_category", "experience_years", "decision"]].copy()
        for c in ["skill_score", "similarity_score", "category_probability"]:
            view[c] = (view[c] * 100).round(0)
        st.dataframe(
            view, hide_index=True, use_container_width=True, height=min(35 * len(view) + 38, 600),
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "file_name": "Candidate",
                "final_score": st.column_config.ProgressColumn("Final score", min_value=0, max_value=100, format="%.1f"),
                "skill_score": st.column_config.NumberColumn("Keyword %", format="%d"),
                "similarity_score": st.column_config.NumberColumn("Similarity %", format="%d"),
                "category_probability": st.column_config.NumberColumn("KNN prob. %", format="%d"),
                "predicted_category": "KNN category",
                "experience_years": st.column_config.NumberColumn("Exp. (yrs)", format="%.1f"),
                "decision": "Decision"})
        fig = px.bar(df.sort_values("rank", ascending=False), x="final_score", y="file_name", orientation="h", color="decision",
                     color_discrete_map={"Shortlisted": "#0f766e", "Not shortlisted": "#94a3b8"},
                     labels={"final_score": "Final score (out of 100)", "file_name": ""}, text="final_score")
        fig.add_vline(x=res["threshold"], line_dash="dash", line_color="#b91c1c",
                      annotation_text=f"cut-off {res['threshold']:.0f}")
        fig.update_layout(height=max(320, 34 * len(df) + 120), legend_title="", margin=dict(l=0, r=10, t=30, b=0),
                          plot_bgcolor="white")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇️ Download ranking (CSV)", view.to_csv(index=False).encode(), "fairlens_ranking.csv", "text/csv")

    with t_cmp:
        names = df.file_name.tolist()
        pick = st.multiselect("Select candidates to compare (2-6)", names, default=names[:min(5, len(names))], max_selections=6)
        sub = df[df.file_name.isin(pick)].sort_values("rank")
        if len(sub) < 2:
            st.info("Select at least two candidates.")
        else:
            labels = {"skill_score": "Keyword match", "similarity_score": "Text similarity", "experience_score": "Experience",
                      "education_score": "Education", "certification_score": "Certifications"}
            long = sub.melt(id_vars="file_name", value_vars=list(labels), var_name="factor", value_name="score")
            long["factor"] = long.factor.map(labels)
            long["score"] *= 100
            a, b = st.columns(2)
            f1 = px.bar(long, x="file_name", y="score", color="factor", barmode="group",
                        labels={"file_name": "", "score": "Factor score (0-100)", "factor": ""},
                        color_discrete_sequence=["#0f766e", "#1f2a44", "#d97706", "#7c3aed", "#94a3b8"],
                        title="Factor scores side by side")
            f1.update_layout(plot_bgcolor="white", height=420, legend=dict(orientation="h", y=-0.25))
            a.plotly_chart(f1, use_container_width=True)
            f2 = go.Figure()
            cats = list(labels.values())
            for _, r in sub.iterrows():
                vals = [r[k] * 100 for k in labels]
                f2.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=r.file_name, fill="toself", opacity=.45))
            f2.update_layout(title="Profile radar", height=420, polar=dict(radialaxis=dict(range=[0, 100])),
                             legend=dict(orientation="h", y=-0.15))
            b.plotly_chart(f2, use_container_width=True)
            kws = res["keywords"]
            z = [[1 if k in r.matched_keywords else 0 for k in kws] for _, r in sub.iterrows()]
            f3 = go.Figure(go.Heatmap(z=z, x=kws, y=sub.file_name, showscale=False, xgap=2, ygap=2,
                                      colorscale=[[0, "#fde2e2"], [1, "#0f766e"]]))
            f3.update_layout(title="Keyword evidence (green = found in resume, pink = missing)", height=120 + 52 * len(sub),
                             xaxis=dict(tickangle=-45), margin=dict(b=130))
            st.plotly_chart(f3, use_container_width=True)

    with t_rep:
        who = st.selectbox("Candidate", df.file_name.tolist(), format_func=lambda n: f"#{int(df.loc[df.file_name == n, 'rank'].iloc[0])}  {n}")
        r = df[df.file_name == who].iloc[0].to_dict()
        st.markdown(f"<div class='fl-match'>{summary_text(r, res['title'], res['category'], len(df))}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown("**Matched keywords**<br>" + (chips(r["matched_keywords"]) or "none"), unsafe_allow_html=True)
        c2.markdown("**Missing keywords**<br>" + (chips(r["missing_keywords"], "miss") or "none"), unsafe_allow_html=True)
        pdf = build_pdf(r, res["title"], res["category"], len(df), res["threshold"], engine.art.gate)
        config.REPORT_DIR.mkdir(exist_ok=True)
        path = config.REPORT_DIR / f"job{res['job_id']}_{who.rsplit('.', 1)[0]}_report.pdf"
        path.write_bytes(pdf)
        db.record_report(who, res["job_id"], path)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download this report (PDF)", pdf, path.name, "application/pdf", type="primary")
        shortlisted = df[df.shortlisted]
        if len(shortlisted):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for _, rr in shortlisted.iterrows():
                    z.writestr(f"{int(rr['rank']):02d}_{rr.file_name.rsplit('.', 1)[0]}_report.pdf",
                               build_pdf(rr.to_dict(), res["title"], res["category"], len(df), res["threshold"], engine.art.gate))
            d2.download_button(f"⬇️ All {len(shortlisted)} shortlisted reports (ZIP)", buf.getvalue(), "shortlisted_reports.zip")


# -------------------------------------------------------------------- main
def main_page():
    user = st.session_state.user
    if not config.MODEL_FILE.exists():
        st.error("Model not found. Run `python train_model.py` first (or copy the `models/` folder from Colab).")
        st.stop()
    engine = get_engine()

    with st.sidebar:
        st.markdown(f"### 👤 {user['name']}")
        st.caption(user["email"])
        with st.expander("⚙️ Advanced settings (optional)"):
            cutoff = st.slider("Shortlist cut-off score", 0, 100, int(engine.art.threshold),
                               help="A candidate is shortlisted if the final score reaches this value and the KNN classifier agrees with the job category.")
            min_years = st.number_input("Minimum experience (years)", 0.0, 20.0, config.DEFAULT_MIN_EXPERIENCE_YEARS, 0.5)
        st.markdown("**Previous screenings**")
        jobs = db.list_jobs(user["user_id"])
        if jobs:
            sel = st.selectbox("Open", jobs, format_func=lambda j: f"{j['title']} · {j['n']} resumes · {j['created_at'][:10]}",
                               label_visibility="collapsed")
            if st.button("Open selected", use_container_width=True):
                job, df = db.load_job(sel["job_id"])
                st.session_state.result = {"job_id": job["job_id"], "title": job["title"], "category": job["matched_category"],
                                           "df": df, "keywords": __import__("json").loads(job["skills"]),
                                           "threshold": cutoff, "errors": []}
                st.rerun()
        else:
            st.caption("None yet.")
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.markdown("<div class='fl-banner'><h1>🔍 FairLens</h1><p>Enter a job title, upload resumes - get a ranked, "
                "explained shortlist in minutes.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='fl-step'>Step 1 · Job title</div>", unsafe_allow_html=True)
    title = st.text_input("Job title", placeholder="e.g. Data Scientist, Java Developer, HR Executive",
                          label_visibility="collapsed")
    category = None
    if title.strip():
        m = engine.resolve_title(title)
        if m["category"]:
            category = m["category"]
        else:
            st.warning(f"No confident match for “{title}”. Pick the closest job family:")
            category = st.selectbox("Closest job family", engine.art.classes,
                                    index=engine.art.classes.index(m["best_guess"]))
        kws = engine.keywords_for(category)
        kws = list(dict.fromkeys(kws))
        st.markdown(f"<div class='fl-match'>Matched to dataset job family <b>{category}</b> "
                    f"(match confidence {m['confidence']*100:.0f}%).<br>The model will search every resume for these "
                    f"{len(kws)} keywords learned from the dataset:<br>{chips(kws)}</div>", unsafe_allow_html=True)

    st.markdown("<br><div class='fl-step'>Step 2 · Upload resumes (PDF)</div>", unsafe_allow_html=True)
    files = st.file_uploader("Resumes", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if files:
        st.caption(f"{len(files)} resume(s) ready.")

    go_btn = st.button("🔎 Screen resumes", type="primary", disabled=not (category and files))
    if go_btn:
        docs, errors, texts, seen = [], [], {}, set()
        bar = st.progress(0.0, text="Reading resumes...")
        for i, f in enumerate(files):
            name = f.name if f.name not in seen else f"{i}_{f.name}"
            seen.add(name)
            try:
                text = pdf_to_text(f.getvalue())
                docs.append({"name": name, "text": text})
                texts[name] = text
            except ParseError as e:
                errors.append((f.name, str(e)))
            bar.progress((i + 1) / len(files), text=f"Reading resumes... {i + 1}/{len(files)}")
        bar.empty()
        if not docs:
            st.error("None of the uploaded PDFs could be read.")
        else:
            with st.spinner("Scoring candidates..."):
                df = engine.screen(category, docs, threshold=cutoff, min_years=min_years)
            kws_all = list(dict.fromkeys(engine.keywords_for(category)))
            job_id = db.save_screening(user["user_id"], title.strip(), category, kws_all, df, texts)
            st.session_state.result = {"job_id": job_id, "title": title.strip(), "category": category, "df": df,
                                       "keywords": kws_all, "threshold": cutoff, "errors": errors}

    res = st.session_state.get("result")
    if res:
        res["threshold"] = cutoff
        st.markdown("---")
        st.subheader(f"Results for “{res['title']}”")
        results_view(res, engine)


if "user" not in st.session_state:
    login_page()
else:
    main_page()
