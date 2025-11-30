# agentic/jd_parser.py
from .utils import clean_text, extract_keywords_simple

def parse_jd_text(text: str):
    """
    Parse a JD from text.
    Extracts keywords using a simple cleaner + keyword extractor.
    """
    if not text:
        return {"text": "", "keywords": []}

    clean = clean_text(text)
    keywords = extract_keywords_simple(clean)
    return {
        "text": clean,
        "keywords": keywords
    }


def parse_jd_file(path: str):
    """
    Parse a JD stored in a .txt file.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception:
        raw = ""

    return parse_jd_text(raw)
