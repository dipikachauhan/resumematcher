import streamlit as st
import re
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Resume Matcher", page_icon="🧠", layout="wide")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f8fafc; }
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 60%, #38bdf8 100%);
    border-radius: 16px; padding: 2.5rem 2rem; color: white; margin-bottom: 2rem;
}
.hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { font-size: 1.05rem; opacity: 0.85; margin-top: 0.4rem; }
.score-card {
    background: white; border-radius: 14px; padding: 1.4rem 1rem;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-top: 4px solid #2563eb;
}
.score-card .lbl { font-size: 0.75rem; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: .05em; }
.score-card .val { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin: 4px 0; }
.score-card .sub { font-size: 0.8rem; color: #94a3b8; }
.sec { font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
    border-left: 4px solid #2563eb; padding-left: 10px; margin: 1.5rem 0 0.8rem; }
.chip { display: inline-block; border-radius: 20px; padding: 3px 12px;
    font-size: 0.8rem; font-weight: 600; margin: 3px; }
.chip-blue  { background:#dbeafe; color:#1d4ed8; }
.chip-green { background:#dcfce7; color:#15803d; }
.chip-red   { background:#fee2e2; color:#b91c1c; }
.gap-card { border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.4rem 0; font-size: 0.88rem; }
.gap-critical { background:#fff1f2; border-left:4px solid #ef4444; }
.gap-moderate { background:#fffbeb; border-left:4px solid #f59e0b; }
.gap-minor    { background:#f0fdf4; border-left:4px solid #22c55e; }
.box { background:white; border-radius:14px; padding:1.4rem;
    box-shadow:0 2px 12px rgba(0,0,0,0.07); margin-bottom:1rem; }
.stButton>button {
    background: linear-gradient(135deg,#1e3a5f,#2563eb); color:white;
    border:none; border-radius:10px; padding:0.65rem 2rem;
    font-size:1rem; font-weight:600; width:100%;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models (cached — downloads once, reused forever) ───────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from sentence_transformers import SentenceTransformer
    import spacy
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return sbert, nlp

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 AI Resume Matching System</h1>
  <p>Semantic NLP · Skill Extraction · Weighted Scoring · Gap Analysis</p>
</div>
""", unsafe_allow_html=True)

# ─── Skill DB (from your Colab + extras) ─────────────────────────────────────
SKILL_DB = [
    "python", "java", "sql", "machine learning", "deep learning",
    "nlp", "data analysis", "pandas", "numpy", "tensorflow",
    "pytorch", "excel", "tableau", "power bi", "aws", "gcp",
    "docker", "kubernetes", "git",
    # extras
    "javascript", "typescript", "c++", "c#", "go", "rust", "ruby",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "computer vision", "data science", "statistics", "data engineering",
    "scikit-learn", "keras", "xgboost", "matplotlib", "seaborn", "looker",
    "azure", "terraform", "ansible", "github", "gitlab", "ci/cd",
    "jenkins", "github actions", "react", "angular", "vue", "node.js",
    "django", "flask", "fastapi", "spark", "hadoop", "kafka", "airflow",
    "agile", "scrum", "jira", "leadership", "project management",
    "restful api", "graphql", "microservices", "linux", "bash",
]

# ─── Core Functions — exact logic from your Colab ────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    return text

def extract_text_from_pdf(uploaded_file):
    import pdfplumber
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILL_DB:
        if skill in text:
            found_skills.append(skill)
    return list(set(found_skills))

def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())
    if matches:
        years = [int(m[0]) for m in matches]
        return max(years)
    return 0

def semantic_similarity(jd_text, resume_text):
    from sentence_transformers import util
    sbert, _ = load_models()
    emb1 = sbert.encode(jd_text,     convert_to_tensor=True)
    emb2 = sbert.encode(resume_text, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score)

def skill_score(jd_skills, resume_skills):
    if not jd_skills:
        return 1
    return len(set(jd_skills) & set(resume_skills)) / len(jd_skills)

def experience_score(jd_exp, resume_exp):
    if jd_exp == 0:
        return 1
    return min(resume_exp / jd_exp, 1.0)

def final_score(skill_s, exp_s, semantic_s):
    weights = {"skills": 0.4, "experience": 0.3, "semantic": 0.3}
    score = (
        skill_s    * weights["skills"]     +
        exp_s      * weights["experience"] +
        semantic_s * weights["semantic"]
    )
    return round(score * 100, 2)

def gap_analysis(jd_skills, resume_skills, jd_exp, resume_exp):
    missing_skills = list(set(jd_skills) - set(resume_skills))
    gaps = {"critical": [], "moderate": [], "minor": []}

    priority = {"python", "java", "machine learning", "sql", "aws",
                "deep learning", "docker", "kubernetes", "tensorflow", "pytorch"}

    for skill in missing_skills:
        if skill in priority:
            gaps["critical"].append(f"Missing skill: **{skill.title()}**")
        else:
            gaps["moderate"].append(f"Missing skill: **{skill.title()}**")

    if resume_exp < jd_exp:
        diff = jd_exp - resume_exp
        gaps["critical"].append(f"Experience short by **{diff} year{'s' if diff > 1 else ''}**")

    return gaps

def analyze_resume(jd_text, resume_text):
    jd_clean     = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    jd_skills     = extract_skills(jd_clean)
    resume_skills = extract_skills(resume_clean)
    jd_exp        = extract_experience(jd_clean)
    resume_exp    = extract_experience(resume_clean)

    s_score   = skill_score(jd_skills, resume_skills)
    e_score   = experience_score(jd_exp, resume_exp)
    sem_score = semantic_similarity(jd_clean, resume_clean)
    total     = final_score(s_score, e_score, sem_score)
    gaps      = gap_analysis(jd_skills, resume_skills, jd_exp, resume_exp)

    return {
        "match_score":      total,
        "skill_score":      round(s_score   * 100, 2),
        "experience_score": round(e_score   * 100, 2),
        "semantic_score":   round(sem_score * 100, 2),
        "jd_skills":        jd_skills,
        "resume_skills":    resume_skills,
        "jd_exp":           jd_exp,
        "resume_exp":       resume_exp,
        "gaps":             gaps,
        "matched":          list(set(jd_skills) & set(resume_skills)),
        "missing":          list(set(jd_skills) - set(resume_skills)),
        "extra":            list(set(resume_skills) - set(jd_skills)),
    }

# ─── Helpers ──────────────────────────────────────────────────────────────────
def score_color(s):
    return "#22c55e" if s >= 75 else "#f59e0b" if s >= 50 else "#ef4444"

def score_label(s):
    return "🟢 Strong Match" if s >= 75 else "🟡 Moderate Match" if s >= 50 else "🔴 Weak Match"

# ─── UI: Inputs ───────────────────────────────────────────────────────────────
col_jd, col_res = st.columns(2, gap="large")

with col_jd:
    st.markdown("### 📋 Job Description")
    jd_mode = st.radio("", ["Paste Text", "Upload PDF"], key="jd_mode", horizontal=True)
    jd_text = ""
    if jd_mode == "Paste Text":
        jd_text = st.text_area("Paste JD here", height=280,
            placeholder="We are looking for a Data Scientist with 3+ years of experience.\nSkills required: Python, Machine Learning, SQL, AWS, Deep Learning.",
            key="jd_paste")
    else:
        jd_file = st.file_uploader("Upload JD PDF", type=["pdf"], key="jd_file")
        if jd_file:
            jd_text = extract_text_from_pdf(jd_file)
            st.success(f"✅ {jd_file.name} loaded ({len(jd_text)} chars)")
            with st.expander("Preview extracted text"):
                st.text(jd_text[:600] + ("..." if len(jd_text) > 600 else ""))

with col_res:
    st.markdown("### 📄 Resume")
    res_mode = st.radio("", ["Paste Text", "Upload PDF"], key="res_mode", horizontal=True)
    resume_text = ""
    if res_mode == "Paste Text":
        resume_text = st.text_area("Paste Resume here", height=280,
            placeholder="I have 2 years of experience working as a Data Analyst.\nSkilled in Python, SQL, Pandas, and Machine Learning.",
            key="res_paste")
    else:
        res_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="res_file")
        if res_file:
            resume_text = extract_text_from_pdf(res_file)
            st.success(f"✅ {res_file.name} loaded ({len(resume_text)} chars)")
            with st.expander("Preview extracted text"):
                st.text(resume_text[:600] + ("..." if len(resume_text) > 600 else ""))

# ─── Analyse Button ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, bcol, _ = st.columns([2, 2, 2])
with bcol:
    run = st.button("🔍  Analyse Match")

# ─── Results ──────────────────────────────────────────────────────────────────
if run:
    if not jd_text.strip() or not resume_text.strip():
        st.error("⚠️ Please provide both a Job Description and a Resume.")
    else:
        with st.spinner("⏳ Loading AI models (first run downloads ~80MB — only once)..."):
            load_models()

        with st.spinner("🔍 Running NLP pipeline..."):
            result = analyze_resume(jd_text, resume_text)

        total = result["match_score"]
        color = score_color(total)

        # ── Score Cards ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Match Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="score-card" style="border-top-color:{color}">
            <div class="lbl">Overall Score</div>
            <div class="val" style="color:{color}">{total}<span style="font-size:1rem">/100</span></div>
            <div class="sub">{score_label(total)}</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="score-card"><div class="lbl">Skills Match</div>
            <div class="val">{result['skill_score']}%</div>
            <div class="sub">{len(result['matched'])}/{len(result['jd_skills'])} skills matched</div></div>""",
            unsafe_allow_html=True)
        c3.markdown(f"""<div class="score-card"><div class="lbl">Experience</div>
            <div class="val">{result['experience_score']}%</div>
            <div class="sub">{result['resume_exp']} yr / {result['jd_exp']} yr required</div></div>""",
            unsafe_allow_html=True)
        c4.markdown(f"""<div class="score-card"><div class="lbl">Semantic Similarity</div>
            <div class="val">{result['semantic_score']}%</div>
            <div class="sub">SBERT all-MiniLM-L6-v2</div></div>""", unsafe_allow_html=True)

        # ── Progress bars + Skills ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Score Breakdown</div>', unsafe_allow_html=True)
            st.progress(result['skill_score']      / 100, text=f"Skills Match — {result['skill_score']}%")
            st.progress(result['experience_score'] / 100, text=f"Experience — {result['experience_score']}%")
            st.progress(result['semantic_score']   / 100, text=f"Semantic Similarity — {result['semantic_score']}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Skills Overview</div>', unsafe_allow_html=True)
            if result["matched"]:
                st.markdown("**✅ Matched Skills**")
                st.markdown(" ".join(f'<span class="chip chip-green">{s}</span>'
                    for s in sorted(result["matched"])), unsafe_allow_html=True)
            if result["missing"]:
                st.markdown("**❌ Missing Skills**")
                st.markdown(" ".join(f'<span class="chip chip-red">{s}</span>'
                    for s in sorted(result["missing"])), unsafe_allow_html=True)
            if result["extra"]:
                st.markdown("**⭐ Bonus / Extra Skills**")
                st.markdown(" ".join(f'<span class="chip chip-blue">{s}</span>'
                    for s in sorted(result["extra"])), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Gap Analysis ──────────────────────────────────────────────────────
        st.markdown('<div class="sec">🔍 Gap Analysis</div>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3, gap="medium")

        with g1:
            st.markdown("#### 🔴 Critical")
            if result["gaps"]["critical"]:
                for g in result["gaps"]["critical"]:
                    st.markdown(f'<div class="gap-card gap-critical">🔴 {g}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="gap-card gap-minor">✅ No critical gaps!</div>', unsafe_allow_html=True)

        with g2:
            st.markdown("#### 🟡 Moderate")
            if result["gaps"]["moderate"]:
                for g in result["gaps"]["moderate"]:
                    st.markdown(f'<div class="gap-card gap-moderate">🟡 {g}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="gap-card gap-minor">✅ No moderate gaps!</div>', unsafe_allow_html=True)

        with g3:
            st.markdown("#### 🟢 Minor")
            if result["gaps"]["minor"]:
                for g in result["gaps"]["minor"]:
                    st.markdown(f'<div class="gap-card gap-minor">🟢 {g}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="gap-card gap-minor">✅ No minor gaps!</div>', unsafe_allow_html=True)

        # ── Recommendation ────────────────────────────────────────────────────
        st.markdown('<div class="sec">💡 Recruiter Recommendation</div>', unsafe_allow_html=True)
        if total >= 75:
            bg  = "#f0fdf4"
            msg = f"✅ **Shortlist for interview.** Strong match ({total}/100). " + \
                  (f"Probe: {result['gaps']['critical'][0]}" if result['gaps']['critical']
                   else "Candidate meets all key requirements.")
        elif total >= 50:
            bg  = "#fffbeb"
            msg = f"⚠️ **Consider with probing questions.** Moderate match ({total}/100). " + \
                  f"Missing: {', '.join(result['missing'][:3]) or 'some skills'}."
        else:
            bg  = "#fff1f2"
            msg = f"❌ **Not recommended.** Low match ({total}/100). Significant skill/experience gaps."

        st.markdown(f"""<div style="background:{bg};border-radius:12px;padding:1.2rem 1.5rem;
            font-size:0.97rem;border:1px solid #e2e8f0;">{msg}</div>""", unsafe_allow_html=True)

        # ── Raw output — mirrors your Colab print statements exactly ──────────
        with st.expander("🖥️ Raw Output (same as your Colab terminal)"):
            st.code(f"""Match Score: {result['match_score']}

Skill Score: {result['skill_score']}
Experience Score: {result['experience_score']}
Semantic Score: {result['semantic_score']}

JD Skills: {result['jd_skills']}
Resume Skills: {result['resume_skills']}

Gaps:
critical : {result['gaps']['critical']}
moderate : {result['gaps']['moderate']}
minor    : {result['gaps']['minor']}""", language="text")

        # ── Download Report ───────────────────────────────────────────────────
        report = f"""AI RESUME MATCH REPORT
======================
Match Score      : {result['match_score']} / 100
Verdict          : {score_label(total)}

Skill Score      : {result['skill_score']}%
Experience Score : {result['experience_score']}%  ({result['resume_exp']} yr vs {result['jd_exp']} yr required)
Semantic Score   : {result['semantic_score']}%

JD Skills     : {', '.join(result['jd_skills'])}
Resume Skills : {', '.join(result['resume_skills'])}

Matched  : {', '.join(result['matched']) or 'None'}
Missing  : {', '.join(result['missing']) or 'None'}
Bonus    : {', '.join(result['extra'])   or 'None'}

Gaps:
  Critical : {chr(10).join(result['gaps']['critical']) or 'None'}
  Moderate : {chr(10).join(result['gaps']['moderate']) or 'None'}
  Minor    : {chr(10).join(result['gaps']['minor'])    or 'None'}
"""
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("📥 Download Report (.txt)", data=report,
            file_name="resume_match_report.txt", mime="text/plain")