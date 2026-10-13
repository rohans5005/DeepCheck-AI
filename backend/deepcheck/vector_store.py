import os
import json
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepCheckVectorStore")

class VectorStore:
    """
    FAISS-powered Vector Database for efficient semantic document & sentence retrieval.
    Stores dense 384-dim embeddings with metadata mapping.
    """
    def __init__(self, dim: int = 384, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index = None
        self.use_faiss = False
        
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            # Inner Product (IP) index equals Cosine Similarity when vectors are L2 normalized
            self.faiss_index = faiss.IndexFlatIP(self.dim)
            self.use_faiss = True
            logger.info("FAISS IndexFlatIP initialized successfully!")
        except Exception as e:
            logger.warning(f"FAISS not available ({e}). Using Scikit-Learn / NumPy matrix vector store.")
            self.use_faiss = False
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)

    def add_vectors(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add dense embedding vectors and corresponding metadata items to index."""
        if len(embeddings) == 0:
            return
            
        embeddings = np.asarray(embeddings, dtype=np.float32)
        # Normalize vectors for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normalized_embeds = (embeddings / norms).astype(np.float32)

        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.add(normalized_embeds)
        else:
            if self.vectors is None or self.vectors.shape[0] == 0:
                self.vectors = normalized_embeds
            else:
                self.vectors = np.vstack([self.vectors, normalized_embeds])

        self.metadata.extend(metadatas)
        logger.info(f"Added {len(metadatas)} vectors to store. Total vectors: {len(self.metadata)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query vector database for Top-K nearest neighbors by Cosine Similarity.
        Returns list of matched dict items with similarity score and metadata.
        """
        if len(self.metadata) == 0:
            return []

        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_vector) + 1e-9
        normalized_query = (query_vector / norm).astype(np.float32)

        top_k = min(top_k, len(self.metadata))

        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(normalized_query, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            sims = np.dot(self.vectors, normalized_query.T).ravel()
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["similarity_score"] = float(np.clip(score, 0.0, 1.0))
            item["rank"] = rank + 1
            results.append(item)

        return results

    def remove_document(self, doc_id: str) -> int:
        """Remove all vector chunks associated with a specific document ID."""
        keep_indices = [i for i, meta in enumerate(self.metadata) if meta.get("doc_id") != doc_id]
        removed_count = len(self.metadata) - len(keep_indices)

        if removed_count == 0:
            return 0

        new_metadata = [self.metadata[i] for i in keep_indices]
        
        # Re-build vector store
        if self.use_faiss and self.faiss_index is not None:
            # Reconstruct remaining vectors
            if len(keep_indices) > 0:
                all_vecs = np.zeros((len(self.metadata), self.dim), dtype=np.float32)
                for i in range(len(self.metadata)):
                    all_vecs[i] = self.faiss_index.reconstruct(i)
                remaining_vecs = all_vecs[keep_indices]
                import faiss
                self.faiss_index = faiss.IndexFlatIP(self.dim)
                self.faiss_index.add(remaining_vecs)
            else:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(self.dim)
        else:
            if len(keep_indices) > 0 and self.vectors is not None:
                self.vectors = self.vectors[keep_indices]
            else:
                self.vectors = np.zeros((0, self.dim), dtype=np.float32)

        self.metadata = new_metadata
        logger.info(f"Removed document {doc_id}. {removed_count} vectors deleted.")
        return removed_count

    def clear(self):
        """Clear vector database."""
        self.metadata = []
        if self.use_faiss and self.faiss_index is not None:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.dim)
        else:
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Return Vector Database Statistics."""
        unique_docs = len(set(m.get("doc_id") for m in self.metadata if "doc_id" in m))
        return {
            "total_vectors": len(self.metadata),
            "unique_documents": unique_docs,
            "vector_dimension": self.dim,
            "backend": "FAISS IndexFlatIP" if self.use_faiss else "Scikit-Learn Matrix Index",
        }
