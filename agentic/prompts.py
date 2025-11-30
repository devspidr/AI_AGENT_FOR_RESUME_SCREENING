




# agentic/prompts.py
from typing import Dict, Iterable

def _format_keywords(keywords: Iterable[str], limit: int = 10) -> str:
    kws = [k for k in keywords if isinstance(k, str)]
    return ", ".join(kws[:limit])

def explanation_for_score(jd: Dict, resume: Dict, score: Dict) -> str:
    lines = []
    final = score.get("final_score", 0.0)
    emb = score.get("embedding_score", 0.0)
    key = score.get("keyword_score", 0.0)
    lines.append(f"Final score: {final:.4f}")
    lines.append(f"Embedding similarity: {emb:.4f}")
    lines.append(f"Keyword overlap: {key:.4f}")
    if key > 0:
        jd_kws = jd.get("keywords", []) or []
        res_kws = resume.get("keywords", []) or []
        shared = set(k for k in jd_kws if isinstance(k, str)).intersection(k for k in res_kws if isinstance(k, str))
        if shared:
            lines.append("Shared keywords: " + _format_keywords(list(shared), limit=10))
    return "\n".join(lines)
