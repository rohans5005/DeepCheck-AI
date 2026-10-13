import numpy as np
from typing import List, Dict, Any, Tuple


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or 2D numpy vectors."""
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(v1, v2) / (norm1 * norm2))


def batch_cosine_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between matrix1 (N x D) and matrix2 (M x D)."""
    m1_norm = matrix1 / (np.linalg.norm(matrix1, axis=1, keepdims=True) + 1e-9)
    m2_norm = matrix2 / (np.linalg.norm(matrix2, axis=1, keepdims=True) + 1e-9)
    return np.dot(m1_norm, m2_norm.T)


def calculate_evaluation_metrics(
    y_true: List[int],
    y_pred_scores: List[float],
    threshold: float = 0.75
) -> Dict[str, float]:
    """
    Calculate KPIs specified in PRD Section 8:
    - Accuracy, Precision, Recall, F1-Score, Pearson Correlation
    """
    y_true_arr = np.array(y_true, dtype=int)
    scores_arr = np.array(y_pred_scores, dtype=float)
    y_pred_binary = (scores_arr >= threshold).astype(int)

    tp = np.sum((y_true_arr == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_arr == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_arr == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_arr == 0) & (y_pred_binary == 0))

    total = len(y_true_arr)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Pearson Correlation Coefficient
    if len(scores_arr) > 1 and np.std(y_true_arr) > 0 and np.std(scores_arr) > 0:
        pearson_corr = float(np.corrcoef(y_true_arr, scores_arr)[0, 1])
    else:
        pearson_corr = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "pearson_correlation": round(pearson_corr, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_samples": int(total),
        "threshold": threshold
    }
