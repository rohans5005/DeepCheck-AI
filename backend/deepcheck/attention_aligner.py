import re
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.utils.metrics import batch_cosine_similarity, cosine_similarity


class AttentionAligner:
    """
    Attention Mechanism & Sentence Alignment Engine for Visual Heatmap Generation.
    Generates word and sentence level similarity intensity heatmaps with attention score attribution.
    """

    @staticmethod
    def align_sentences(
        input_sentences: List[str],
        source_sentences: List[str],
        input_embeddings: np.ndarray,
        source_embeddings: np.ndarray,
        threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Align input sentences with source document sentences using Cosine Similarity matrix.
        Returns detailed list of aligned sentence mapping objects.
        """
        if len(input_sentences) == 0 or len(source_sentences) == 0:
            return []

        # Pairwise Cosine Similarity Matrix (N_input x M_source)
        sim_matrix = batch_cosine_similarity(input_embeddings, source_embeddings)

        sentence_highlights = []
        for i, in_sent in enumerate(input_sentences):
            row_sims = sim_matrix[i]
            best_match_idx = int(np.argmax(row_sims))
            best_score = float(row_sims[best_match_idx])
            matched_src_sent = source_sentences[best_match_idx]

            # Paraphrase Classification
            if best_score >= 0.88:
                paraphrase_type = "Near-Exact Copy"
                color_code = "high"  # Red
            elif best_score >= 0.72:
                paraphrase_type = "Semantic Paraphrase"
                color_code = "high"  # Red/Orange
            elif best_score >= 0.55:
                paraphrase_type = "Structural / Synonym Alteration"
                color_code = "medium"  # Orange
            elif best_score >= threshold:
                paraphrase_type = "Weak Context Match"
                color_code = "low"  # Yellow
            else:
                paraphrase_type = "Original Content"
                color_code = "none"

            # Compute Word-level Attention Map within sentence
            word_attentions = AttentionAligner._compute_word_attentions(in_sent, matched_src_sent, best_score)

            sentence_highlights.append({
                "sentence_index": i,
                "input_sentence": in_sent,
                "matched_source_sentence": matched_src_sent if best_score >= threshold else None,
                "source_sentence_index": best_match_idx if best_score >= threshold else None,
                "similarity_score": round(best_score, 4),
                "similarity_percentage": round(best_score * 100, 1),
                "is_plagiarized": best_score >= threshold,
                "paraphrase_type": paraphrase_type,
                "severity_level": color_code,
                "word_attentions": word_attentions
            })

        return sentence_highlights

    @staticmethod
    def _compute_word_attentions(input_sent: str, source_sent: str, sentence_score: float) -> List[Dict[str, Any]]:
        """
        Simulate Token Cross-Attention weights between words of input and source sentence.
        Used for word-level glowing hover heatmaps.
        """
        words = input_sent.split()
        source_words_lower = set(w.lower() for w in re.findall(r'\w+', source_sent))
        
        attentions = []
        for w in words:
            clean_w = re.sub(r'[^\w]', '', w).lower()
            if not clean_w:
                attentions.append({"word": w, "attention_weight": 0.0})
                continue
                
            if clean_w in source_words_lower:
                # Exact word overlap receives full weight
                weight = min(1.0, sentence_score * 1.1)
            else:
                # Subword/synonym contextual weight proportional to overall sentence embedding similarity
                weight = max(0.1, sentence_score * 0.75)
                
            attentions.append({
                "word": w,
                "attention_weight": round(float(weight), 3)
            })

        return attentions

    @staticmethod
    def generate_overall_heatmap(
        aligned_sentences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary statistics for document heatmap visualization."""
        total = len(aligned_sentences)
        if total == 0:
            return {"plagiarism_ratio": 0.0, "high_risk_count": 0, "medium_risk_count": 0}

        high_count = sum(1 for s in aligned_sentences if s["severity_level"] == "high")
        med_count = sum(1 for s in aligned_sentences if s["severity_level"] == "medium")
        low_count = sum(1 for s in aligned_sentences if s["severity_level"] == "low")
        plagiarized_count = sum(1 for s in aligned_sentences if s["is_plagiarized"])

        # Weighted Plagiarism Percentage
        scores = [s["similarity_score"] for s in aligned_sentences if s["is_plagiarized"]]
        overall_score = float(np.mean(scores)) if scores else 0.0

        return {
            "overall_plagiarism_score": round(overall_score * 100, 2),
            "plagiarized_sentence_percentage": round((plagiarized_count / total) * 100, 2),
            "total_sentences": total,
            "flagged_sentences": plagiarized_count,
            "high_severity_count": high_count,
            "medium_severity_count": med_count,
            "low_severity_count": low_count
        }
