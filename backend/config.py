import os
from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
DATA_DIR.mkdir(exist_ok=True)
CORPUS_DIR.mkdir(exist_ok=True)

# Deep Learning Model Configurations
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MULTILINGUAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # Dimension for MiniLM-L6-v2 vectors

# Plagiarism Detection Thresholds
SIMILARITY_THRESHOLD_HIGH = 0.85
SIMILARITY_THRESHOLD_MEDIUM = 0.65
SIMILARITY_THRESHOLD_LOW = 0.45

# Sliding Window Settings for Long Text Handling
MAX_SEQ_LENGTH = 256
SLIDING_WINDOW_STRIDE = 128

# Vector DB Settings
INDEX_FILE_PATH = DATA_DIR / "faiss_index.bin"
METADATA_FILE_PATH = DATA_DIR / "corpus_metadata.json"

# Server Host & Port
HOST = "127.0.0.1"
PORT = 8000
