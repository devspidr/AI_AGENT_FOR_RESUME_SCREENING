# 📄 Resume ↔ Job Description Matcher  
### Agentic AI · Streamlit · Embeddings · Keyword Scoring · PDF Support

A fast, lightweight, recruiter-friendly **Resume Screening Agent** built using  
**Python + Streamlit + Embeddings + Intelligent Parsing**.

Upload a **Job Description (JD)** + multiple **Resumes (PDF/TXT)** →  
the system parses, extracts candidate information, computes similarity scores,  
and displays ranked candidates with a beautiful UI.

This project is ideal for:

- 🏆 Hackathons  
- 🎓 AI/ML student portfolio  
- 🧩 HR prototype tools  
- 🚀 Resume screening demos  
- 🤖 Agentic AI learning projects  

---

# 🚀 Features

## 🧠 Agentic Workflow (Lightweight DAG Flow)

The engine follows a structured pipeline:

parse_jd → parse_resume_text → extract_features → compute_scores → rank → explanation


### ✔️ Perception (Data Extraction)
- Clean extraction from **PDF/TXT**
- JD → keywords, expectations, responsibilities, skills  
- Resume → name, location, skills, projects, education, summary

### ✔️ Reasoning  
- Weighted composite scoring using:
  - Embedding similarity  
  - Keyword overlap  
  - Section understanding  
  - Resume structure quality  

### ✔️ Action  
- Rank all candidates  
- Display summaries  
- Extract meaningful insights  
- Generate formatted candidate cards  
- Provide download-ready PDF reports

### ✔️ Learning  
- System includes utility layers:
  - Text cleaning  
  - Keyword extraction  
  - Name extraction  
  - Resume summarization  
  - JD similarity recalibration  

---

# 📊 Scoring System

Each resume receives multiple computed signals:

| Score Type | Description |
|-----------|-------------|
| **EmbeddingScore** | Semantic match between JD and resume |
| **KeywordScore** | Fuzzy keyword overlap detection |
| **SkillScore** | Skills extracted from resume vs JD |
| **CompositeScore** | Weighted combined score for ranking |

The scoring weights can be modified via slider in the UI:

- Weight on Embeddings  
- Weight on Keywords  

---

# 📁 Project Structure
```
RESUME+JD/
├─ agentic/
│ ├─ agent.py # Main agent entry
│ ├─ config.py # Weights, global constants
│ ├─ embedding_manager.py # Embeddings + cosine similarity
│ ├─ parsers.py # JD + Resume parser wrapper
│ ├─ jd_parser.py # JD structural extraction
│ ├─ resume_parser.py # Resume cleanup and section extraction
│ ├─ prompts.py # Lightweight reasoning templates
│ ├─ scoring.py # Composite scoring logic
│ ├─ utils.py # Cleaning, PDF extraction, keyword helper
│ ├─ graph.py # Lightweight DAG-like flow (no full LangGraph needed)
│
├─ app/
│ ├─ app.py # Streamlit UI
│
├─ data/
│ ├─ logs/ # run logs (optional)
│ ├─ sample_jds/ # sample files for testing
│
├─ .env # environment variables (ignored by git)
├─ requirements.txt # pip dependencies
└─ README.md # this file

```

---

# 🔧 Installation & Setup

## 1️⃣ Clone this repository

```bash
git clone https://github.com/<your-github-username>/RESUME-JD-Matcher.git
cd RESUME-JD-Matcher
```

2️⃣ Create & activate a Virtual Environment

Windows
```
python -m venv .venv
.\.venv\Scripts\activate
```

macOS / Linux
```
python3 -m venv .venv
source .venv/bin/activate
```
3️⃣ Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
4️⃣ Environment Variables

Create .env file in project root:
```
OPENAI_API_KEY=your_key_here
PYTHONPATH=./
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

⚠️ No quotes should be used.

5️⃣ Run the Streamlit app
```
cd app
streamlit run app.py

```
Open browser:
```
👉 http://localhost:8501
```
## 🧪 Usage Guide

### **Step 1 — Provide Job Description**
You can either:
- Paste the JD text into the sidebar text area  
**or**
- Upload `JD.pdf` / `JD.txt`

The system automatically extracts:
- Expected skills  
- Experience criteria  
- Responsibilities  
- Role keywords  

---

### **Step 2 — Upload Multiple Resumes**
Upload multiple `PDF` / `TXT` resumes.

The system extracts:
- Candidate name  
- Skills  
- Projects  
- Education  
- Summary  
- Contact information  
- Work-related content  

---

### **Step 3 — Adjust Weight Slider**
You can control:
- Embedding weight  
- Keyword weight  

**Final Score = weighted combination of all signals**

---

### **Step 4 — Run the Matcher**
When you run the matcher, the system performs:
1. JD parsing  
2. Resume parsing  
3. Skills extraction  
4. Semantic similarity computation  
5. Composite scoring  
6. Candidate ranking  
7. Summary generation  

---

## 🧾 Output Details

Each ranked candidate card includes:

### 🟩 **Rank + Name + Score**
### 📝 **Quick Summary**
- Location  
- Education  
- Years of experience  
- Top skills  
- Top project titles  

### 🧩 **Extracted Skills**
Curated list of relevant skills using tech-aware keyword detection.

### 🔍 **Scoring Breakdown**
- Embedding Score  
- Keyword Score  
- Composite Score  
- Section-based signals  

### 📄 **Resume Snippet**
Clean preview of important resume sections.

### 📥 **PDF Download (Optional)**
Includes:
- Candidate summary  
- Skills & project insights  
- JD alignment  
- Scoring table  

### 🗃 **Logs (Optional)**
If enabled, the `/data/logs/` folder stores debug and processing logs.

---

## 🏆 Credits

**Built by:** *Soundar Balaji J*  
*CSE • AI/ML • NLP • Agentic AI • Resume Intelligence Systems*

**GitHub:** https://github.com/devspidr
