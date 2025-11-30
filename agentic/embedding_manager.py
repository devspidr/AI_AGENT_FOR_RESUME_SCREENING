

# agentic/embedding_manager.py
import numpy as np
from typing import List
from .config import EMBEDDING_MODEL_NAME

# Try to use sentence-transformers if available; otherwise fall back to TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
    _ST_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception:
    _ST_AVAILABLE = False
    _ST_MODEL = None
    from sklearn.feature_extraction.text import TfidfVectorizer
    _VECT = None

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts -> numpy array (n_texts x dim)
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    if _ST_AVAILABLE and _ST_MODEL is not None:
        embs = _ST_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.array(embs, dtype=float)
    # fallback tfidf (dense)
    global _VECT
    if _VECT is None:
        _VECT = TfidfVectorizer(max_features=4096)
        _VECT.fit(texts)
    mat = _VECT.transform(texts).toarray()
    return mat.astype(float)

def embed_text(text: str, chunking: bool = True, chunk_size: int = 200) -> np.ndarray:
    """
    Embed a single text. If chunking True, split into chunks and average embeddings.
    """
    if not text:
        # dimension unknown; return zero vector of length 768 if SBERT else 0-length
        if _ST_AVAILABLE and _ST_MODEL is not None:
            return np.zeros(_ST_MODEL.get_sentence_embedding_dimension(), dtype=float)
        else:
            return np.zeros(512, dtype=float)
    if not chunking:
        return embed_texts([text])[0]
    words = text.split()
    if len(words) <= chunk_size:
        return embed_texts([text])[0]
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    embs = embed_texts(chunks)
    return np.mean(embs, axis=0)
