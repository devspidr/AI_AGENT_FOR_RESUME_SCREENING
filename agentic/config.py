# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Embedding model settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Scoring weights: embedding similarity vs keyword overlap
WEIGHT_EMBEDDING = float(os.getenv("WEIGHT_EMBEDDING", 0.7))
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", 0.3))

# Top K results to show
TOP_K = int(os.getenv("TOP_K", 10))

