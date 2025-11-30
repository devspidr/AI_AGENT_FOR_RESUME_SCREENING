












# app/app.py
import sys
import os
import io
import re
import html
import collections
import datetime
import streamlit as st

# Ensure project root is on sys.path when running from inside app/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic.agent import SimpleAgent
from agentic.config import TOP_K
from agentic.utils import clean_text, pdf_bytes_to_text, extract_keywords_simple
from agentic import config as _config

# try to import reportlab for PDF generation
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# pdf handling
try:
    import pdfplumber  # noqa: F401
    _PDFPLUMBER_AVAILABLE = True
except Exception:
    _PDFPLUMBER_AVAILABLE = False

st.set_page_config(page_title="Resume↔JD Matcher", layout="wide")
st.title("Resume ↔ Job Description Matcher")

st.sidebar.header("Inputs")
jd_source = st.sidebar.text_area(
    "Paste JD text here (leave blank to upload file)", height=200
)
jd_file = st.sidebar.file_uploader(
    "Or upload JD file (txt/pdf)",
    type=["txt", "pdf"]
)

resume_files = st.sidebar.file_uploader(
    "Upload resumes (txt/pdf) — multiple allowed",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.write("Weights (embedding vs keywords)")
weight_emb = st.sidebar.slider("Embedding weight", 0.0, 1.0, 0.7, 0.05)

_config.WEIGHT_EMBEDDING = float(weight_emb)
_config.WEIGHT_KEYWORD = float(1.0 - weight_emb)

st.sidebar.markdown("## Debug (uploader)")
if st.sidebar.checkbox("Show uploaded debug info", value=False):
    st.sidebar.write("count:", len(resume_files) if resume_files else 0)
    names_dbg = [getattr(f, "name", None) for f in (resume_files or [])]
    st.sidebar.write("names:", names_dbg)

# ---------------------------
# Helpers: name extraction & concise, improved skills/project extraction
# ---------------------------

EMAIL_RE = re.compile(r"([a-zA-Z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")
LOCATION_RE = re.compile(
    r'\b(Bengaluru|Bangalore|Delhi|Hyderabad|Chennai|Mumbai|Pune|Kolkata|Trivandrum|Bengaluru,|Bengaluru\.|Bengaluru:)\b',
    flags=re.I,
)
EDU_RE = re.compile(r'\b(B\.?E\.?|BTech|M\.?E\.?|MTech|B\.?Sc|M\.?Sc|MBA|Ph\.?D|Diploma|BCA|BCom)\b', flags=re.I)
PROJECT_TITLE_RE = re.compile(r'([A-Z][A-Za-z0-9 &\-\:]{4,120})(?:\s+–|\s+-|\s+\(|\s+:)', flags=re.M)

# curated tech vocabulary snippets (used to bias skills)
_TECH_KEYWORDS = {
    "python", "java", "c", "c++", "c#", "javascript", "react", "reactjs", "node", "flask", "django",
    "sql", "postgres", "mysql", "mongodb", "pandas", "numpy", "sklearn", "scikit", "tensorflow", "pytorch",
    "docker", "kubernetes", "aws", "gcp", "azure", "git", "linux", "html", "css", "bootstrap", "matplotlib",
    "seaborn", "fastapi", "streamlit", "opencv", "nlp", "tensorflow", "lightgbm", "xgboost", "spark",
    "hadoop", "rest", "api", "graphql", "bash", "shell", "typescript", "redux", "chroma", "faiss"
}

_STOPWORDS = {
    "and", "or", "the", "to", "for", "with", "in", "on", "by", "of", "a", "an", "is", "are", "be",
    "as", "that", "this", "it", "at", "from", "i", "we", "you", "their", "have", "has", "will", "can",
    "skill", "skills", "experienced", "experience", "work", "projects", "project", "years", "year"
}

def to_title_name(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return " ".join([w.capitalize() for w in s.split() if w])

def extract_name_from_text(text: str) -> str:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 1. check for Name: pattern
    for ln in lines[:12]:
        m = re.search(r'^(?:name[:\-\s]{1,}|candidate[:\-\s]{1,})(.+)$', ln, flags=re.I)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'[\|\-\,\•\*\:\(\)]', ' ', name).strip()
            if 1 < len(name.split()) <= 6:
                return to_title_name(name)
    # 2. first line that looks like name (2-4 words title case)
    for ln in lines[:6]:
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or "linkedin" in ln.lower() or "github" in ln.lower():
            continue
        candidate = re.sub(r'[^A-Za-z\s\-]', '', ln).strip()
        words = candidate.split()
        if 1 < len(words) <= 4 and all(re.match(r'^[A-Z][a-z\-]+$', w) for w in words):
            return " ".join(words)
    # 3. derive from email
    for ln in lines[:12]:
        m = EMAIL_RE.search(ln)
        if m:
            local = m.group(1)
            local = re.sub(r'[\._\-]+', ' ', local)
            local = re.sub(r'\d+', '', local).strip()
            if local:
                return to_title_name(local)
    # 4. linkedin url last segment
    for ln in lines[:12]:
        if "linkedin.com/in/" in ln.lower() or "linkedin.com/pub/" in ln.lower():
            seg = ln.strip().rstrip('/').split('/')[-1]
            seg = seg.replace('-', ' ').replace('_', ' ')
            return to_title_name(seg)
    # fallback: first 3 words
    if lines:
        first = re.sub(r'[^A-Za-z\s]', ' ', lines[0]).strip()
        parts = first.split()
        if parts:
            return to_title_name(" ".join(parts[:3]))
    return None

def extract_location(text: str) -> str:
    if not text:
        return None
    m = LOCATION_RE.search(text)
    if m:
        return m.group(0).strip()
    m2 = re.search(r'Location[:\-\s]+([A-Za-z\s]{3,40})', text, flags=re.I)
    if m2:
        return m2.group(1).strip()
    return None

def extract_education_one_line(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:120]:
        if EDU_RE.search(ln):
            return ln[:120].strip()
    for ln in lines[:120]:
        if re.search(r'\b(University|Institute|College|School|VTU|Sir M\.?)\b', ln, flags=re.I):
            return ln[:120].strip()
    return ""

def extract_years_experience(text: str) -> str:
    if not text:
        return ""
    m = re.findall(r'(\d{1,2})(?:\+)?\s*(?:years|year|yrs|yr)\b', text.lower())
    years = max([int(x) for x in m]) if m else 0
    ranges = re.findall(r'(\b20\d{2}\b)[\s\-–to]{1,5}(\b20\d{2}\b)', text)
    yrs_from_ranges = []
    for a, b in ranges:
        try:
            diff = abs(int(b) - int(a))
            if 0 <= diff < 80:
                yrs_from_ranges.append(diff)
        except Exception:
            pass
    if yrs_from_ranges:
        years = max(years, max(yrs_from_ranges))
    return f"{years} yrs" if years else ""

def _clean_token(tok: str) -> str:
    tok = tok.strip().lower()
    tok = re.sub(r'http\S+|\S+@\S+|\+?\d[\d\s\-\(\)]{7,}\d', '', tok)  # remove emails/phones/urls
    tok = re.sub(r'[^a-z0-9\+\#\.\-]', ' ', tok)
    tok = tok.strip()
    return tok

def extract_top_skills(text: str, max_skills: int = 6) -> list:
    if not text:
        return []
    txt = text
    m = re.search(r'(skills|technical skills|technologies|tools)\s*[:\-\n]\s*(.+?)(?:\n{2,}|\Z)', txt, flags=re.I | re.S)
    tokens = []
    if m:
        body = m.group(2)
        parts = re.split(r'[\n\r]+|[,;•\|··\-•]', body)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if ',' in p or len(p.split()) <= 6:
                subtoks = re.split(r'[,/]', p)
                for st in subtoks:
                    tok = _clean_token(st)
                    if not tok:
                        continue
                    for candidate in tok.split():
                        if candidate and candidate not in _STOPWORDS and len(candidate) > 1:
                            tokens.append(candidate)
            else:
                for candidate in re.findall(r'[A-Za-z\+\#\.]{2,}', p):
                    ct = _clean_token(candidate)
                    if ct and ct not in _STOPWORDS:
                        tokens.append(ct)
    else:
        kws = extract_keywords_simple(txt)
        for k in kws:
            ck = _clean_token(k)
            if ck and ck not in _STOPWORDS and len(ck) > 1:
                tokens.append(ck.lower())

    prioritized = []
    other = []
    seen = set()
    for t in tokens:
        t0 = t.lower()
        if t0 in seen:
            continue
        seen.add(t0)
        if any(t0 == tk or t0.startswith(tk) or t0.endswith(tk) for tk in _TECH_KEYWORDS):
            prioritized.append(t0)
        else:
            other.append(t0)
    final = prioritized + other
    final = [f for f in final if f not in _STOPWORDS and len(f) > 1]
    return [to_title_name(f) for f in final[:max_skills]]

def extract_project_titles(text: str, max_projects: int = 2) -> list:
    if not text:
        return []
    txt = text
    projects = []
    m = re.search(r'(projects|personal projects|academic projects)\s*[:\-\n]\s*(.+?)(?:\n{2,}|\Z)', txt, flags=re.I | re.S)
    if m:
        body = m.group(2)
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        for ln in lines:
            title = re.split(r'\s+–\s+|\s+-\s+|\s+:\s+', ln, maxsplit=1)[0].strip()
            title = re.sub(r'\s{2,}', ' ', title)
            title = re.sub(r'^[\-\•\*]+', '', title).strip()
            if title and len(title) > 6 and len(title) < 120:
                projects.append(title)
            if len(projects) >= max_projects:
                break
    if not projects:
        snippet = txt[:4000]
        lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
        for ln in lines:
            if len(ln) < 120 and re.search(r'\b(project|system|app|bot|assistant|model|application)\b', ln, flags=re.I):
                title = re.split(r'\s+–\s+|\s+-\s+|\s+:\s+', ln, maxsplit=1)[0].strip()
                if title and len(title) > 6:
                    projects.append(title)
            if len(projects) >= max_projects:
                break
    cleaned = []
    seen = set()
    for p in projects:
        p0 = re.sub(r'[^A-Za-z0-9 \-\&\:\,\.]', ' ', p).strip()
        if p0.lower() not in seen and p0:
            cleaned.append(p0)
            seen.add(p0.lower())
        if len(cleaned) >= max_projects:
            break
    return cleaned

def build_concise_summary(text: str) -> str:
    if not text:
        return ""
    txt = clean_text(text)
    location = extract_location(txt) or ""
    years = extract_years_experience(txt) or ""
    education = extract_education_one_line(txt) or ""
    skills = extract_top_skills(txt, max_skills=6)
    projects = extract_project_titles(txt, max_projects=2)

    header_items = [h for h in [location, years, education] if h]
    header = " | ".join(header_items)

    parts = []
    if header:
        parts.append(header)
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    if projects:
        parts.append("Projects: " + "; ".join(projects))
    return "\n\n".join(parts)

# ---------------------------
# App main helpers
# ---------------------------

def extract_text_from_upload(f):
    name = getattr(f, "name", "") or ""
    raw = f.read()
    if name.lower().endswith(".pdf") or getattr(f, "type", "") == "application/pdf":
        return pdf_bytes_to_text(raw)
    else:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            try:
                return raw.decode(errors="ignore")
            except Exception:
                return ""

# ---------------------------
# PDF / HTML report generation
# ---------------------------

def generate_pdf_bytes(jd_text: str, results: list, uploaded_names: list) -> bytes:
    """
    Generate a PDF report using reportlab and return bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=14, leading=16, spaceAfter=6)
    small = ParagraphStyle("Small", parent=normal, fontSize=9, leading=11)
    elems = []

    title_text = f"Resume ↔ JD Matching Report — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elems.append(Paragraph(title_text, heading))
    elems.append(Spacer(1, 6))

    # JD snippet
    elems.append(Paragraph("<b>Job Description (snippet)</b>", styles["Heading3"]))
    jd_snip = clean_text(jd_text or "")[:1000]
    elems.append(Paragraph(jd_snip.replace("\n", "<br/>"), small))
    elems.append(Spacer(1, 8))

    elems.append(Paragraph(f"<b>Uploaded resumes</b>: {', '.join(uploaded_names)}", normal))
    elems.append(Spacer(1, 8))

    # Results
    elems.append(Paragraph(f"<b>Ranked results ({len(results)})</b>", styles["Heading3"]))
    elems.append(Spacer(1, 6))

    for idx, r in enumerate(results, start=1):
        name = r.get("name") or ""
        resume_obj = r.get("resume") or {}
        if not name:
            if isinstance(resume_obj, dict):
                name = (resume_obj.get("meta", {}) or {}).get("name") or resume_obj.get("name") or ""
        if not name:
            name = extract_name_from_text(r.get("raw_resume") or r.get("text_preview") or "") or "Unknown Candidate"

        score_val = r.get("score", 0.0) or 0.0
        elems.append(Paragraph(f"<b>{idx}. {html.escape(name)}</b> — Score: {score_val:.4f}", styles["Heading4"]))
        conc = build_concise_summary(r.get("raw_resume") or r.get("text_preview") or "")
        if conc:
            # split into paragraphs
            for block in conc.split("\n\n"):
                elems.append(Paragraph(html.escape(block), small))
        else:
            elems.append(Paragraph(html.escape((r.get("text_preview") or "")[:400]), small))
        # keywords/skills
        kw = r.get("keywords", []) or []
        if kw:
            elems.append(Paragraph("<b>Top keywords:</b> " + html.escape(", ".join(kw[:20])), small))
        # explanation or scores
        explanation = r.get("explanation") or ""
        if isinstance(explanation, dict):
            expl_text = "; ".join([f"{k}: {v}" for k, v in explanation.items()])
            elems.append(Paragraph("<b>Breakdown:</b> " + html.escape(expl_text), small))
        else:
            if explanation:
                elems.append(Paragraph("<b>Explanation:</b>", small))
                elems.append(Paragraph(html.escape(explanation[:800]).replace("\n", "<br/>"), small))
        elems.append(Spacer(1, 8))
        # page break every few candidates if desired
        if idx % 5 == 0:
            elems.append(PageBreak())

    doc.build(elems)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

def generate_html_report(jd_text: str, results: list, uploaded_names: list) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Resume-JD Report</title>",
        "<style>body{font-family:Arial, sans-serif; background:#fff; color:#111} .card{border:1px solid #ddd;padding:12px;margin:12px 0;border-radius:6px} h2{color:#0b66c2}</style>",
        "</head><body>"
    ]
    html_parts.append(f"<h2>Resume ↔ JD Matching Report</h2><p>{now}</p>")
    html_parts.append(f"<h3>Uploaded resumes</h3><p>{', '.join(uploaded_names)}</p>")
    html_parts.append("<h3>Job Description (snippet)</h3>")
    html_parts.append(f"<pre style='white-space:pre-wrap'>{html.escape((jd_text or '')[:1000])}</pre>")
    html_parts.append("<h3>Ranked Results</h3>")
    for idx, r in enumerate(results, start=1):
        name = r.get("name") or ""
        resume_obj = r.get("resume") or {}
        if not name:
            if isinstance(resume_obj, dict):
                name = (resume_obj.get("meta", {}) or {}).get("name") or resume_obj.get("name") or ""
        if not name:
            name = extract_name_from_text(r.get("raw_resume") or r.get("text_preview") or "") or "Unknown Candidate"
        score_val = r.get("score", 0.0) or 0.0
        html_parts.append(f"<div class='card'><h4>{idx}. {html.escape(name)} — Score: {score_val:.4f}</h4>")
        conc = build_concise_summary(r.get("raw_resume") or r.get("text_preview") or "")
        if conc:
            for block in conc.split("\n\n"):
                html_parts.append(f"<p>{html.escape(block)}</p>")
        else:
            html_parts.append(f"<pre style='white-space:pre-wrap'>{html.escape((r.get('text_preview') or '')[:400])}</pre>")
        kw = r.get("keywords", []) or []
        if kw:
            html_parts.append(f"<p><b>Top keywords:</b> {html.escape(', '.join(kw[:20]))}</p>")
        explanation = r.get("explanation") or ""
        if isinstance(explanation, dict):
            expl_text = "; ".join([f"{k}: {v}" for k, v in explanation.items()])
            html_parts.append(f"<p><b>Breakdown:</b> {html.escape(expl_text)}</p>")
        else:
            if explanation:
                html_parts.append(f"<pre style='white-space:pre-wrap'>{html.escape(explanation[:1000])}</pre>")
        html_parts.append("</div>")
    html_parts.append("</body></html>")
    return "\n".join(html_parts)

# ---------------------------
# Run button
# ---------------------------

results = None
uploaded_names = []
jd_text_final = ""

if st.button("Run Matching"):
    # Prepare JD text
    jd_text = ""
    if jd_file is not None:
        jd_text = extract_text_from_upload(jd_file)
    else:
        jd_text = jd_source

    if not jd_text or len(jd_text.strip()) < 10:
        st.error("Please provide a JD either by pasting text or uploading a txt/pdf file.")
        st.stop()

    if not resume_files:
        st.error("Please upload at least one resume (txt/pdf).")
        st.stop()

    # Read uploaded resumes
    resumes_texts = []
    uploaded_names = []
    for f in resume_files:
        uploaded_names.append(getattr(f, "name", None))
        resumes_texts.append(extract_text_from_upload(f))

    st.info(f"Uploaded {len(resumes_texts)} resumes: {uploaded_names}")

    # Run agent
    agent = SimpleAgent()
    try:
        results = agent.run(jd_text_or_path=jd_text, resumes_text_or_paths=resumes_texts, from_file=False)
    except Exception as e:
        st.error("Error while running the matching pipeline.")
        st.exception(e)
        st.stop()

    jd_text_final = jd_text  # keep for reporting

    st.success(f"Ranked {len(results)} resumes")
    topk = min(TOP_K, len(results))

    for i, r in enumerate(results[:topk], start=1):
        # Resolve name
        name = r.get("name") or None
        resume_obj = r.get("resume") or {}
        if not name and isinstance(resume_obj, dict):
            name = (resume_obj.get("meta", {}) or {}).get("name") or resume_obj.get("name")
        if not name:
            name = extract_name_from_text(r.get("raw_resume") or r.get("text_preview") or "") or "Unknown Candidate"

        # Build concise summary
        raw_text = r.get("raw_resume") or r.get("text_preview") or ""
        concise = build_concise_summary(raw_text)

        score_val = r.get("score", 0.0) or 0.0
        safe_name = html.escape(name)

        # Card
        st.markdown(f"""
        <div style="
            padding: 12px;
            margin: 10px 0;
            border-radius:8px;
            border:1px solid #2b2b2b;
            background-color:#0b1220;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="color:#4CAF50; margin:0;">Rank #{i}</h3>
                    <h4 style="color:#FFDD57; margin:4px 0 0 0;">{safe_name}</h4>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:18px; color:#2196F3;">Score: {score_val:.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Concise summary display
        st.markdown("### 📄 Quick Summary")
        if concise:
            for block in concise.split("\n\n"):
                st.markdown(f"- {html.escape(block)}")
        else:
            st.markdown("- No concise summary available. Showing short preview:")
            st.code(clean_text(raw_text[:800]), language="text")

        # Keywords / Skills
        st.markdown("### 🧩 Top Skills / Keywords")
        skills = extract_top_skills(raw_text, max_skills=8)
        if skills:
            st.write(", ".join([to_title_name(s) for s in skills]))
        else:
            kws = extract_keywords_simple(raw_text)
            filt = [k for k in kws if k.lower() not in _STOPWORDS and len(k) > 2]
            st.write(", ".join([to_title_name(k) for k in filt[:8]]) or "No skills detected.")

        # Scoring breakdown
        with st.expander("🔍 Detailed Scoring Breakdown"):
            explanation = r.get("explanation", "") or ""
            if isinstance(explanation, dict):
                for k, v in explanation.items():
                    st.write(f"**{k}**: {v}")
            else:
                if explanation.strip():
                    st.markdown(explanation.replace("\n", "  \n"))
                else:
                    scores = r.get("scores", {}) or {}
                    if isinstance(scores, dict) and scores:
                        for k, v in scores.items():
                            st.write(f"**{k}**: {v}")
                    else:
                        st.write("No detailed breakdown available.")

        st.markdown("<hr>", unsafe_allow_html=True)

    # After rendering results, show download buttons
    if results:
        st.markdown("## 📥 Download Report")
        if REPORTLAB_AVAILABLE:
            try:
                pdf_bytes = generate_pdf_bytes(jd_text_final, results, uploaded_names)
                st.download_button(
                    label="Download PDF report",
                    data=pdf_bytes,
                    file_name="resume_jd_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error("Failed to generate PDF via reportlab; offering HTML fallback.")
                st.exception(e)
                html_text = generate_html_report(jd_text_final, results, uploaded_names)
                st.download_button(
                    label="Download HTML report",
                    data=html_text.encode("utf-8"),
                    file_name="resume_jd_report.html",
                    mime="text/html"
                )
        else:
            # fallback: HTML
            html_text = generate_html_report(jd_text_final, results, uploaded_names)
            st.info("reportlab not installed — offering HTML report download instead.")
            st.download_button(
                label="Download HTML report",
                data=html_text.encode("utf-8"),
                file_name="resume_jd_report.html",
                mime="text/html"
            )
