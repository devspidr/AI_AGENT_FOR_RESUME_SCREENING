# # utils.py
# import re
# from typing import List
# import pdfplumber

# def pdf_to_text(file):
#     with pdfplumber.open(file) as pdf:
#         pages = [page.extract_text() or "" for page in pdf.pages]
#     return "\n".join(pages)

# def clean_text(text: str) -> str:
#     text = text.replace("\r", " ").replace("\n", " ").strip()
#     text = re.sub(r"\s+", " ", text)
#     return text

# def chunk_text(text: str, max_words: int = 250) -> List[str]:
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_words):
#         chunks.append(" ".join(words[i:i+max_words]))
#     return chunks

# def load_file_text(path: str) -> str:
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         return f.read()

# def extract_keywords(text: str, min_len: int = 2) -> List[str]:
#     text = text.lower()
#     tokens = re.findall(r"[a-zA-Z#+\.\-]{2,}", text)
#     # remove very short tokens and common words (simple stoplist)
#     stop = {"and","or","the","to","for","with","in","on","by","of","a","an","is","are","be"}
#     keywords = [t for t in tokens if t not in stop and len(t) >= min_len]
#     # return unique while preserving order
#     seen = set()
#     out = []
#     for k in keywords:
#         if k not in seen:
#             seen.add(k)
#             out.append(k)
#     return out




# agentic/utils.py
import re
from typing import Iterable
import io

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r", " ").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def chunk_text(text: str, max_words: int = 250) -> Iterable[str]:
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# pdf extraction helper (used by resume parser)
def pdf_bytes_to_text(raw: bytes) -> str:
    try:
        import pdfplumber
    except Exception:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception:
        return ""

def extract_keywords_simple(text: str, min_len: int = 2):
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z0-9\+#\.\-]{2,}", text)
    stop = {"and","or","the","to","for","with","in","on","by","of","a","an","is","are","be"}
    out = []
    seen = set()
    for t in tokens:
        if t in stop or len(t) < min_len:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
