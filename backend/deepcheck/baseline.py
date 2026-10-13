import re
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


class BaselineStringMatcher:
    """
    Traditional Plagiarism Detector Baseline (TF-IDF + Rabin-Karp Exact Matching).
    Serves as comparative baseline against Transformer Semantic Deep Learning models.
    """
    def __init__(self, ngram_range=(1, 3)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")

    def tfidf_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate Cosine Similarity over TF-IDF n-gram vectors."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text_a, text_b])
            sim = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.0

    def rabin_karp_exact_match_ratio(self, text_a: str, text_b: str, k_gram: int = 5) -> float:
        """Rabin-Karp Rolling Hash Exact Substring Match Ratio."""
        tokens_a = re.findall(r'\w+', text_a.lower())
        tokens_b = re.findall(r'\w+', text_b.lower())

        if len(tokens_a) < k_gram or len(tokens_b) < k_gram:
            return 1.0 if text_a.lower() == text_b.lower() else 0.0

        grams_a = set(" ".join(tokens_a[i : i + k_gram]) for i in range(len(tokens_a) - k_gram + 1))
        grams_b = set(" ".join(tokens_b[i : i + k_gram]) for i in range(len(tokens_b) - k_gram + 1))

        if not grams_a or not grams_b:
            return 0.0

        intersection = grams_a.intersection(grams_b)
        jaccard_sim = len(intersection) / float(len(grams_a.union(grams_b)))
        return float(np.clip(jaccard_sim, 0.0, 1.0))

    def evaluate_pair(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """Run full baseline evaluation on a document pair."""
        tfidf_sim = self.tfidf_similarity(text_a, text_b)
        rk_sim = self.rabin_karp_exact_match_ratio(text_a, text_b)
        hybrid_score = round(0.7 * tfidf_sim + 0.3 * rk_sim, 4)

        return {
            "tfidf_cosine_similarity": round(tfidf_sim, 4),
            "rabin_karp_jaccard": round(rk_sim, 4),
            "baseline_score": hybrid_score,
            "method": "TF-IDF + Rabin-Karp N-gram Baseline"
        }
