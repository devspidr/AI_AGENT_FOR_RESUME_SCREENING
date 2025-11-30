# Resume Screening Agent (Streamlit + LangChain + LangGraph + OpenAI)

This project is an **agentic resume screening system**:

- Upload a **Job Description (JD)** and multiple **resumes** (PDF/DOCX).
- The agent:
  - Uses **LangChain + OpenAI** to parse the JD into structured JSON:
    - Role title, must-have / nice-to-have skills, experience, outcomes, risk flags
  - Parses resumes into sections and basic PII (name, email, phone).
  - Builds embeddings via **OpenAIEmbeddings** and uses FAISS for semantic similarity.
  - Computes rich scores:
    - SkillScore, SemanticScore, ExperienceScore, OutcomeScore, RiskScore,
      JDMatchScore, CompositeScore
  - Ranks candidates and shows:
    - Full ranking and **blind-mode ranking** (PII redacted before scoring)
    - Must-have skill coverage and missing skills
    - LLM-generated action (Shortlist / Review / Escalate) and rationale
  - Logs each run in `data/logs/runs.jsonl` (for audit / replay).
  - Provides a **bias & fairness** narrative via the LLM.

All of this is driven by a single Streamlit UI. Under the hood, a **LangGraph graph**
(orchestrator) coordinates steps: JD parsing → resume parsing → embedding →
scoring → rationales.

## Tech Stack

- **Frontend**: Streamlit
- **Agentic Orchestration**: LangGraph
- **LLM & Embeddings**: LangChain + OpenAI (ChatOpenAI, OpenAIEmbeddings)
- **Vector Search**: FAISS (via langchain-community)
- **Storage**: Filesystem (JSONL logs) + persisted uploads

## Setup

### 1. Create virtualenv and install deps

```bash
git clone <this-repo> resume-agent
cd resume-agent

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
