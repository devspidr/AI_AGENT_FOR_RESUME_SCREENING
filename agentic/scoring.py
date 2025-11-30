# # scoring.py
# import numpy as np
# from typing import Dict, List, Tuple
# from .embedding_manager import embed_text, embed_texts
# from .utils import extract_keywords
# from .config import WEIGHT_EMBEDDING, WEIGHT_KEYWORD


# def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#     if a is None or b is None:
#         return 0.0
#     numa = np.linalg.norm(a)
#     numb = np.linalg.norm(b)
#     if numa == 0 or numb == 0:
#         return 0.0
#     return float(np.dot(a, b) / (numa * numb))

# def keyword_overlap_score(jd_keywords: List[str], resume_keywords: List[str]) -> float:
#     if not jd_keywords:
#         return 0.0
#     set_j = set(jd_keywords)
#     set_r = set(resume_keywords)
#     inter = set_j.intersection(set_r)
#     return len(inter) / max(1, len(set_j))

# def score_resume_against_jd(jd: Dict, resume: Dict) -> Dict:
#     jd_emb = embed_text(jd["text"])
#     res_emb = embed_text(resume["text"])
#     emb_score = cosine_sim(jd_emb, res_emb)
#     key_score = keyword_overlap_score(jd["keywords"], resume["keywords"])
#     final = WEIGHT_EMBEDDING * emb_score + WEIGHT_KEYWORD * key_score
#     return {
#         "embedding_score": emb_score,
#         "keyword_score": key_score,
#         "final_score": final
#     }

# def bulk_score(jd: Dict, resumes: List[Dict]) -> List[Tuple[int, Dict]]:
#     results = []
#     for i, r in enumerate(resumes):
#         sc = score_resume_against_jd(jd, r)
#         results.append((i, sc))
#     results.sort(key=lambda x: x[1]["final_score"], reverse=True)
#     return results






# agentic/scoring.py
"""
Enhanced scoring combining:
 - embedding similarity (semantic)
 - keyword overlap (including must-have boosts)
 - experience years (extracted heuristically)
 - job-title similarity (title tokens overlap)
 - resume length penalty/bonus
All components are normalized before combining.
"""

import re
import math
import numpy as np
from typing import Dict, List, Tuple

from .embedding_manager import embed_text
from .utils import extract_keywords_simple, clean_text
from .config import WEIGHT_EMBEDDING, WEIGHT_KEYWORD

# Extra configurable weights (tweak in config.py or override at runtime)
WEIGHT_EXPERIENCE = 0.15
WEIGHT_SKILL_BOOST = 0.10
WEIGHT_TITLE = 0.10
LENGTH_PENALTY_WEIGHT = 0.05  # small penalty for extremely short resumes

# Helpers
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def extract_years_experience(text: str) -> float:
    """
    Heuristic to extract years of experience from resume text.
    Looks for patterns like 'X years', 'X yrs', 'X+ years', and date ranges like '2019-2023'.
    Returns a float >= 0.
    """
    if not text:
        return 0.0
    text = text.lower()
    # pattern: '12 years', '3 yrs', '4+ years'
    m = re.findall(r'(\d{1,2})(?:\+)?\s*(?:years|year|yrs|yr)\b', text)
    nums = [int(x) for x in m] if m else []
    years = max(nums) if nums else 0

    # date ranges like 2018-2022 or 2018 to 2022
    ranges = re.findall(r'(\b20\d{2}\b)[\s\-–to]{1,5}(\b20\d{2}\b)', text)
    yrs_from_ranges = []
    for a, b in ranges:
        try:
            diff = abs(int(b) - int(a))
            if diff >= 0 and diff < 80:
                yrs_from_ranges.append(diff)
        except Exception:
            pass
    if yrs_from_ranges:
        years = max(years, max(yrs_from_ranges))

    return float(years)

def title_similarity(jd_text: str, resume_text: str) -> float:
    """
    Simple title similarity based on token overlap of likely title tokens.
    We look at first 40 words from each to guess titles and roles.
    """
    jdt = " ".join(jd_text.split()[:40]).lower()
    rt = " ".join(resume_text.split()[:40]).lower()
    jdt_tokens = set(re.findall(r'[a-zA-Z0-9\+#\.\-]{2,}', jdt))
    rt_tokens = set(re.findall(r'[a-zA-Z0-9\+#\.\-]{2,}', rt))
    if not jdt_tokens or not rt_tokens:
        return 0.0
    inter = jdt_tokens.intersection(rt_tokens)
    return len(inter) / max(1, len(jdt_tokens))

def keyword_overlap_with_boost(jd_keywords: List[str], resume_keywords: List[str], must_have: List[str]=None) -> Tuple[float, float]:
    """
    Compute base overlap score and a 'skill boost' if resume contains must-have keywords.
    Returns (base_overlap, boost_value)
    """
    if not jd_keywords:
        return 0.0, 0.0
    jset = set([k.lower() for k in jd_keywords])
    rset = set([k.lower() for k in resume_keywords])
    inter = jset.intersection(rset)
    base = len(inter) / max(1, len(jset))

    boost = 0.0
    if must_have:
        musts = [m.lower() for m in must_have if m]
        if musts:
            hits = sum(1 for m in musts if m in rset)
            boost = hits / max(1, len(musts))
    return base, boost

def normalize(values: List[float]) -> List[float]:
    """
    Min-max normalize a list of values into [0,1]. If constant, return zeros.
    """
    if not values:
        return values
    arr = np.array(values, dtype=float)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if math.isclose(mx, mn):
        # all equal -> return zeros
        return [0.0 for _ in values]
    norm = ((arr - mn) / (mx - mn)).tolist()
    return norm

# Main scoring functions
def score_resume_against_jd(jd: Dict, resume: Dict) -> Dict:
    """
    Returns detailed scoring components and final composites.
    jd: {"text":..., "keywords":[...], "must_have": [...](optional)}
    resume: {"text":..., "keywords":[...], "meta": {...}}
    """
    jd_text = jd.get("text", "") or ""
    jd_keywords = jd.get("keywords", []) or []
    must_have = jd.get("must_have", []) if isinstance(jd.get("must_have", []), list) else []

    res_text = resume.get("text", "") or ""
    res_keywords = resume.get("keywords", []) or []

    # Embedding similarity (semantic)
    jd_emb = embed_text(jd_text)
    res_emb = embed_text(res_text)
    emb_score = cosine_sim(jd_emb, res_emb)

    # Keyword overlap + boost for must-have hits
    kw_base, kw_boost = keyword_overlap_with_boost(jd_keywords, res_keywords, must_have)

    # Experience heuristic
    years = extract_years_experience(res_text)

    # Title similarity
    title_sim = title_similarity(jd_text, res_text)

    # Length factor: penalize extremely short resumes (under 100 words)
    words = len(res_text.split())
    if words < 100:
        length_factor = max(0.0, words / 100.0)  # e.g. 50 words -> 0.5
    else:
        length_factor = 1.0

    # Put raw component values in a dict
    components = {
        "embedding_raw": emb_score,
        "keyword_base_raw": kw_base,
        "keyword_boost_raw": kw_boost,
        "years_experience_raw": years,
        "title_sim_raw": title_sim,
        "length_raw": length_factor
    }

    # We'll normalize across small groups: embedding & title are semantic; keyword and boost are lexical; years is separate.
    # For a single pair comparison we can't normalize across dataset here, so we'll instead apply sensible scalers:
    # scale years to [0,1] with 10 years map -> 1.0 (cap at 10)
    years_scaled = min(1.0, years / 10.0)

    # Compose intermediate scores
    semantic_score = emb_score * 0.85 + title_sim * 0.15  # embed is main semantic signal
    lexical_score = kw_base
    skill_boost = kw_boost

    # Apply length factor
    semantic_score *= length_factor
    lexical_score *= length_factor
    skill_boost *= length_factor

    # final composite (weights configurable)
    # base weights from config: WEIGHT_EMBEDDING and WEIGHT_KEYWORD
    composite = (
        WEIGHT_EMBEDDING * semantic_score +
        WEIGHT_KEYWORD * lexical_score +
        WEIGHT_EXPERIENCE * years_scaled +
        WEIGHT_SKILL_BOOST * skill_boost +
        WEIGHT_TITLE * title_sim
    )

    # clamp and return
    composite = max(0.0, min(1.0, composite))

    result = {
        "embedding_score": float(emb_score),
        "title_score": float(title_sim),
        "keyword_score": float(kw_base),
        "skill_boost": float(skill_boost),
        "experience_years": float(years),
        "experience_score": float(years_scaled),
        "length_factor": float(length_factor),
        "final_score": float(composite)
    }
    return result

def bulk_score(jd: Dict, resumes: List[Dict]) -> List[Tuple[int, Dict]]:
    """
    Score all resumes and return list of (index, score_dict) sorted by final_score desc.
    """
    results = []
    for i, r in enumerate(resumes):
        sc = score_resume_against_jd(jd, r)
        results.append((i, sc))
    results.sort(key=lambda x: x[1]["final_score"], reverse=True)
    return results
