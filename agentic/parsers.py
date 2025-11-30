# # parsers.py
# from typing import Dict
# from .utils import clean_text, extract_keywords, load_file_text


# def parse_text_source(text: str) -> Dict:
#     text = clean_text(text)
#     keywords = extract_keywords(text)
#     return {"text": text, "keywords": keywords}

# def parse_file(path: str) -> Dict:
#     txt = load_file_text(path)
#     return parse_text_source(txt)

# # convenience wrappers for resume and jd
# def parse_jd(text_or_path: str, from_file: bool = False):
#     if from_file:
#         return parse_file(text_or_path)
#     return parse_text_source(text_or_path)

# def parse_resume(text_or_path: str, from_file: bool = False):
#     if from_file:
#         return parse_file(text_or_path)
#     return parse_text_source(text_or_path)

# agentic/parsers.py
from typing import Dict
from .utils import clean_text
from . import jd_parser as _jd_parser
from . import resume_parser as _resume_parser

def parse_jd(text_or_path: str, from_file: bool = False) -> Dict:
    """
    Return standardized JD dict: {"text": ..., "keywords": [...]}
    If from_file True, text_or_path is a filesystem path handled by your jd_parser.
    """
    if from_file:
        jd = _jd_parser.parse_jd_file(text_or_path)
    else:
        raw = text_or_path if isinstance(text_or_path, str) else ""
        jd = _jd_parser.parse_jd_text(raw)
    # Normalize
    jd_text = clean_text(jd.get("text", "") if isinstance(jd, dict) else jd)
    jd_keywords = jd.get("keywords", []) if isinstance(jd, dict) else []
    return {"text": jd_text, "keywords": jd_keywords}

def parse_resume(text_or_path: str, from_file: bool = False) -> Dict:
    """
    Return standardized resume dict: {"text": ..., "keywords": [...], "meta": {...}}
    """
    if from_file:
        res = _resume_parser.parse_resume_file(text_or_path)
    else:
        raw = text_or_path if isinstance(text_or_path, str) else ""
        res = _resume_parser.parse_resume_text(raw)
    txt = clean_text(res.get("text", "") if isinstance(res, dict) else res)
    kws = res.get("keywords", []) if isinstance(res, dict) else []
    meta = res.get("meta", {}) if isinstance(res, dict) else {}
    return {"text": txt, "keywords": kws, "meta": meta}
