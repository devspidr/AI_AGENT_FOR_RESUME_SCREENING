# agentic/resume_parser.py
from typing import Dict
from .utils import clean_text, pdf_bytes_to_text, extract_keywords_simple

def parse_resume_text(text: str) -> Dict:
    """
    Parse resume from plain text (return dict with text, keywords, meta).
    """
    raw = text or ""
    txt = clean_text(raw)
    keywords = extract_keywords_simple(txt)
    meta = {}
    return {"text": txt, "keywords": keywords, "meta": meta}

def parse_resume_file(path: str) -> Dict:
    """
    Parse resume from a file path. Supports .txt and .pdf (if pdfplumber is installed).
    """
    try:
        if isinstance(path, bytes):
            # If caller accidentally passes bytes, try to extract from bytes
            txt = pdf_bytes_to_text(path)
            if txt:
                return parse_resume_text(txt)
            else:
                # fallback to decoding bytes as text
                try:
                    return parse_resume_text(path.decode("utf-8", errors="ignore"))
                except Exception:
                    return {"text":"", "keywords":[], "meta":{}}
        path_str = str(path or "")
        if path_str.lower().endswith(".pdf"):
            try:
                with open(path_str, "rb") as f:
                    raw = f.read()
                txt = pdf_bytes_to_text(raw)
                return parse_resume_text(txt)
            except Exception:
                return {"text":"", "keywords":[], "meta":{}}
        else:
            try:
                with open(path_str, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                return parse_resume_text(raw)
            except Exception:
                return {"text":"", "keywords":[], "meta":{}}
    except Exception:
        return {"text":"", "keywords":[], "meta":{}}
