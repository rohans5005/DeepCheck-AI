import numpy as np
from typing import Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class SiameseDistanceNetwork(nn.Module):
        """
        Siamese Neural Network Architecture for Document & Sentence Paraphrase Similarity.
        Passes dual text embedding representations through twin sub-networks with shared weights,
        calculating Cosine Similarity and non-linear metric projection.
        """
        def __init__(self, input_dim: int = 384, hidden_dim: int = 256, projection_dim: int = 128):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, projection_dim)
            )

        def forward_one(self, x: torch.Tensor) -> torch.Tensor:
            x_proj = self.projection(x)
            return F.normalize(x_proj, p=2, dim=1)

        def forward(self, input_a: torch.Tensor, input_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out_a = self.forward_one(input_a)
            out_b = self.forward_one(input_b)
            cosine_sim = F.cosine_similarity(out_a, out_b, dim=1)
            return cosine_sim, out_a - out_b

    class ContrastiveLoss(nn.Module):
        """
        Contrastive Loss Function for Siamese Network fine-tuning.
        Minimizes distance for positive paraphrased pairs, maximizes for negative pairs up to margin m.
        Formula: L(y, d) = 0.5 * y * d^2 + 0.5 * (1 - y) * max(0, margin - d)^2
        """
        def __init__(self, margin: float = 1.0):
            super().__init__()
            self.margin = margin

        def forward(self, output_a: torch.Tensor, output_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            euclidean_distance = F.pairwise_distance(output_a, output_b)
            loss_contrastive = torch.mean(
                (1 - label) * 0.5 * torch.pow(euclidean_distance, 2) +
                (label) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            )
            return loss_contrastive

    class TripletLoss(nn.Module):
        """
        Triplet Loss Function (Anchor, Positive, Negative).
        Goal: d(Anchor, Positive) + margin < d(Anchor, Negative)
        """
        def __init__(self, margin: float = 0.5):
            super().__init__()
            self.margin = margin

        def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
            distance_positive = F.pairwise_distance(anchor, positive)
            distance_negative = F.pairwise_distance(anchor, negative)
            losses = torch.relu(distance_positive - distance_negative + self.margin)
            return losses.mean()

else:
    # NumPy Fallback implementation of Siamese Contrastive calculation
    class SiameseDistanceNetwork:
        def __init__(self, input_dim: int = 384, hidden_dim: int = 256, projection_dim: int = 128):
            self.input_dim = input_dim

        def predict_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
            norm_a = vec_a / (np.linalg.norm(vec_a) + 1e-9)
            norm_b = vec_b / (np.linalg.norm(vec_b) + 1e-9)
            return float(np.dot(norm_a, norm_b))
