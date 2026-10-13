import time
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from backend.utils.metrics import calculate_evaluation_metrics, cosine_similarity
from backend.deepcheck.baseline import BaselineStringMatcher

logger = logging.getLogger("DeepCheckTrainer")

# Benchmark Paraphrase Dataset Samples (Simulated MRPC / Quora Question Pairs)
BENCHMARK_DATASET = [
    {
        "text_a": "Artificial intelligence algorithms process vast amounts of unstructured text data to identify patterns.",
        "text_b": "AI models analyze massive volumes of raw textual information to detect complex patterns.",
        "label": 1  # Plagiarized / Paraphrased
    },
    {
        "text_a": "The university library will be closed on Sunday for routine annual electrical maintenance.",
        "text_b": "The academic library remains shut on Sunday owing to scheduled annual power system servicing.",
        "label": 1  # Plagiarized / Paraphrased
    },
    {
        "text_a": "Quantum computing uses qubits which exist in superpositions of zero and one states.",
        "text_b": "Traditional digital computers store data in binary bits that are either exclusively 0 or 1.",
        "label": 0  # Non-plagiarized / Dissimilar
    },
    {
        "text_a": "Climate change is leading to increased frequency and severity of extreme weather events across the globe.",
        "text_b": "Global warming causes more frequent and severe weather disasters around the world.",
        "label": 1  # Plagiarized / Paraphrased
    },
    {
        "text_a": "Photosynthesis turns carbon dioxide and water into glucose and oxygen using light energy.",
        "text_b": "Mitochondria act as the cellular powerhouse by generating ATP through cellular respiration.",
        "label": 0  # Non-plagiarized / Dissimilar
    },
    {
        "text_a": "Deep learning networks rely on multi-layer perceptrons with backpropagation for parameter optimization.",
        "text_b": "Deep neural architectures utilize stacked perceptron layers updated via gradient backpropagation algorithms.",
        "label": 1  # Plagiarized / Paraphrased
    },
    {
        "text_a": "The company reported a 15 percent increase in third-quarter revenue compared to last year.",
        "text_b": "The quarterly earnings report showed a significant rise in annual sales volume.",
        "label": 1  # Paraphrased
    },
    {
        "text_a": "The space shuttle launched successfully into low Earth orbit on Monday morning.",
        "text_b": "Deep sea exploration vehicles require reinforced titanium hulls to withstand extreme abyssal pressures.",
        "label": 0  # Dissimilar
    }
]


class ModelEvaluator:
    """Evaluates Plagiarism Detection models against benchmark paraphrase test dataset."""

    @staticmethod
    def run_evaluation_benchmark(embedder, threshold: float = 0.70) -> Dict[str, Any]:
        """
        Compare Baseline TF-IDF String Matcher vs Deep Learning Sentence Transformer.
        Computes Accuracy, Precision, Recall, F1, Latency, and Pearson correlation.
        """
        baseline_matcher = BaselineStringMatcher()
        
        y_true = [item["label"] for item in BENCHMARK_DATASET]
        
        # 1. Baseline Evaluation
        start_time = time.time()
        baseline_scores = []
        for item in BENCHMARK_DATASET:
            res = baseline_matcher.evaluate_pair(item["text_a"], item["text_b"])
            baseline_scores.append(res["baseline_score"])
        baseline_latency = (time.time() - start_time) / len(BENCHMARK_DATASET)
        
        baseline_metrics = calculate_evaluation_metrics(y_true, baseline_scores, threshold=0.35)
        baseline_metrics["avg_latency_ms"] = round(baseline_latency * 1000, 2)
        baseline_metrics["model_name"] = "Baseline (TF-IDF + Rabin-Karp)"

        # 2. Deep Learning SBERT Embedder Evaluation
        start_time = time.time()
        sbert_scores = []
        for item in BENCHMARK_DATASET:
            vecs = embedder.encode_sentences([item["text_a"], item["text_b"]])
            sim = cosine_similarity(vecs[0], vecs[1])
            sbert_scores.append(sim)
        sbert_latency = (time.time() - start_time) / len(BENCHMARK_DATASET)

        sbert_metrics = calculate_evaluation_metrics(y_true, sbert_scores, threshold=threshold)
        sbert_metrics["avg_latency_ms"] = round(sbert_latency * 1000, 2)
        sbert_metrics["model_name"] = f"DeepCheck SBERT ({embedder.model_name})"

        return {
            "baseline": baseline_metrics,
            "deep_learning_sbert": sbert_metrics,
            "benchmark_dataset_size": len(BENCHMARK_DATASET),
            "similarity_scores_breakdown": [
                {
                    "pair_id": idx + 1,
                    "label": BENCHMARK_DATASET[idx]["label"],
                    "baseline_score": round(baseline_scores[idx], 4),
                    "sbert_score": round(sbert_scores[idx], 4),
                    "text_a_snippet": BENCHMARK_DATASET[idx]["text_a"][:50] + "...",
                    "text_b_snippet": BENCHMARK_DATASET[idx]["text_b"][:50] + "..."
                }
                for idx in range(len(BENCHMARK_DATASET))
            ]
        }


class FineTuningTrainer:
    """Simulates fine-tuning epoch loop over Contrastive / Triplet Loss for Siamese Networks."""

    @staticmethod
    def train_epoch(num_epochs: int = 5, learning_rate: float = 2e-5) -> Dict[str, Any]:
        """
        Simulate training progress over Siamese contrastive loss with learning rate scheduler.
        """
        epoch_history = []
        initial_loss = 0.85
        
        for epoch in range(1, num_epochs + 1):
            # Simulated loss decay curve
            loss = initial_loss * (0.65 ** (epoch - 1)) + np.random.uniform(0.01, 0.03)
            acc = min(0.98, 0.70 + (epoch * 0.05) + np.random.uniform(-0.01, 0.01))
            f1 = min(0.97, 0.68 + (epoch * 0.055))
            
            epoch_history.append({
                "epoch": epoch,
                "loss": round(float(loss), 4),
                "accuracy": round(float(acc), 4),
                "f1_score": round(float(f1), 4),
                "learning_rate": learning_rate * (0.9 ** (epoch - 1))
            })

        return {
            "num_epochs": num_epochs,
            "final_loss": epoch_history[-1]["loss"],
            "final_accuracy": epoch_history[-1]["accuracy"],
            "final_f1_score": epoch_history[-1]["f1_score"],
            "optimizer": "AdamW (weight_decay=0.01)",
            "loss_function": "ContrastiveLoss (margin=1.0)",
            "history": epoch_history
        }
