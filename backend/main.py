import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from backend.config import SBERT_MODEL_NAME, MULTILINGUAL_MODEL_NAME, SIMILARITY_THRESHOLD_HIGH
from backend.utils.text_processor import TextProcessor
from backend.deepcheck.embedder import DeepCheckEmbedder
from backend.deepcheck.vector_store import VectorStore
from backend.deepcheck.attention_aligner import AttentionAligner
from backend.deepcheck.baseline import BaselineStringMatcher
from backend.deepcheck.cross_lingual import CrossLingualDetector
from backend.deepcheck.trainer import ModelEvaluator, FineTuningTrainer
from backend.deepcheck.corpus_data import DEFAULT_CORPUS_DOCUMENTS
from backend.utils.metrics import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepCheckMain")

app = FastAPI(
    title="DeepCheck: Semantic Plagiarism Detection API",
    description="Deep Learning-based Plagiarism Detection System using Transformer Embeddings, Siamese Networks & FAISS Vector Indexing",
    version="1.0.0"
)

# Enable CORS for local development & frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Deep Learning Components
logger.info("Initializing DeepCheck Deep Learning Models & Vector Store...")
sbert_embedder = DeepCheckEmbedder(model_name=SBERT_MODEL_NAME)
multilingual_detector = CrossLingualDetector(model_name=MULTILINGUAL_MODEL_NAME)
baseline_matcher = BaselineStringMatcher()
vector_store = VectorStore(dim=384)

# Seed Vector Store with Default Academic Corpus
def seed_default_corpus():
    if vector_store.get_stats()["total_vectors"] == 0:
        logger.info("Seeding default academic corpus documents into FAISS Vector Database...")
        for doc in DEFAULT_CORPUS_DOCUMENTS:
            sentences = TextProcessor.segment_into_sentences(doc["text"])
            if not sentences:
                continue
            embeddings = sbert_embedder.encode_sentences(sentences)
            metadatas = [
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "author": doc["author"],
                    "category": doc["category"],
                    "sentence_index": idx,
                    "sentence_text": sentence,
                    "full_document_text": doc["text"]
                }
                for idx, sentence in enumerate(sentences)
            ]
            vector_store.add_vectors(embeddings, metadatas)
        logger.info("Default corpus seeded successfully!")

seed_default_corpus()


# Serve Frontend Static Files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    if (FRONTEND_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    if (FRONTEND_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/")
def read_root():
    """Serve Frontend HTML or API Info."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "system": "DeepCheck Semantic Plagiarism Detection Engine",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/api/health")
def get_health():
    """System Health & Hardware Status."""
    try:
        import torch
        cuda_available = torch.cuda.is_available() if hasattr(torch, "cuda") else False
    except ImportError:
        cuda_available = False

    return {
        "status": "healthy",
        "gpu_available": cuda_available,
        "device": "NVIDIA GPU" if cuda_available else "PyTorch CPU",
        "sbert_model": SBERT_MODEL_NAME,
        "multilingual_model": MULTILINGUAL_MODEL_NAME,
        "vector_store": vector_store.get_stats()
    }


@app.post("/api/check-plagiarism")
async def check_plagiarism(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    model_type: str = Form("sbert"),  # 'sbert', 'multilingual', 'baseline', 'hybrid'
    threshold: float = Form(0.65),
    top_k: int = Form(5)
):
    """
    Main Plagiarism Detection Pipeline (PRD Section 4.2 & 5.2):
    1. Extract & clean text input
    2. Segment into sentences
    3. Generate Transformer Embeddings (v = Model(Sentence))
    4. Query FAISS Vector Database for top K nearest neighbors
    5. Perform Attention Alignment & Heatmap Generation
    """
    start_time = time.time()

    # Step 1: Ingest input text or uploaded document file
    input_text = ""
    filename = "Pasted_Text.txt"
    if file is not None:
        file_bytes = await file.read()
        filename = file.filename or "Uploaded_Document"
        input_text = TextProcessor.extract_text_from_bytes(file_bytes, filename)
    elif text and text.strip():
        input_text = text.strip()

    if not input_text:
        raise HTTPException(status_code=400, detail="No valid text or document file provided.")

    # Step 2: Sentence Segmentation
    sentences = TextProcessor.segment_into_sentences(input_text)
    if not sentences:
        raise HTTPException(status_code=400, detail="Could not extract meaningful sentences from text.")

    # Step 3: Vectorization & FAISS Nearest Neighbor Querying
    if model_type == "multilingual":
        embeddings = multilingual_detector.encode(sentences)
    else:
        embeddings = sbert_embedder.encode_sentences(sentences)

    all_matched_sources = {}
    aligned_results = []

    for i, sent in enumerate(sentences):
        sent_vec = embeddings[i]
        # Query FAISS vector index
        matches = vector_store.search(sent_vec, top_k=top_k)
        
        best_match = None
        best_score = 0.0
        
        if matches:
            top_candidate = matches[0]
            score = top_candidate["similarity_score"]
            best_score = score
            if score >= threshold:
                best_match = top_candidate
                doc_id = top_candidate["doc_id"]
                if doc_id not in all_matched_sources:
                    all_matched_sources[doc_id] = {
                        "doc_id": doc_id,
                        "title": top_candidate.get("title", "Unknown Source"),
                        "author": top_candidate.get("author", "Unknown Author"),
                        "category": top_candidate.get("category", "General"),
                        "match_count": 0,
                        "max_similarity": score,
                        "matched_sentences": []
                    }
                all_matched_sources[doc_id]["match_count"] += 1
                all_matched_sources[doc_id]["max_similarity"] = max(all_matched_sources[doc_id]["max_similarity"], score)
                all_matched_sources[doc_id]["matched_sentences"].append({
                    "input_sentence": sent,
                    "source_sentence": top_candidate.get("sentence_text"),
                    "similarity": score
                })

        # Determine severity
        if best_score >= 0.85:
            severity = "high"
            paraphrase_type = "Near-Exact Copy / Paraphrase"
        elif best_score >= 0.70:
            severity = "high"
            paraphrase_type = "Semantic Paraphrase"
        elif best_score >= threshold:
            severity = "medium"
            paraphrase_type = "Synonym / Structural Swap"
        else:
            severity = "none"
            paraphrase_type = "Original Content"

        word_attentions = AttentionAligner._compute_word_attentions(
            sent,
            best_match.get("sentence_text", "") if best_match else "",
            best_score
        )

        aligned_results.append({
            "sentence_index": i,
            "input_sentence": sent,
            "matched_source": best_match,
            "similarity_score": round(best_score, 4),
            "similarity_percentage": round(best_score * 100, 1),
            "is_plagiarized": best_score >= threshold,
            "severity_level": severity,
            "paraphrase_type": paraphrase_type,
            "word_attentions": word_attentions
        })

    # Step 4: Calculate overall stats & visual heatmap summaries
    summary = AttentionAligner.generate_overall_heatmap(aligned_results)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    matched_sources_list = sorted(
        list(all_matched_sources.values()),
        key=lambda x: x["max_similarity"],
        reverse=True
    )

    return {
        "filename": filename,
        "total_characters": len(input_text),
        "total_sentences": len(sentences),
        "overall_similarity_score": summary["overall_plagiarism_score"],
        "plagiarized_sentence_percentage": summary["plagiarized_sentence_percentage"],
        "severity_breakdown": {
            "high": summary["high_severity_count"],
            "medium": summary["medium_severity_count"],
            "low": summary["low_severity_count"],
            "clean": summary["total_sentences"] - summary["flagged_sentences"]
        },
        "matched_sources": matched_sources_list,
        "sentence_alignments": aligned_results,
        "model_used": model_type,
        "threshold_used": threshold,
        "latency_ms": latency_ms
    }


@app.post("/api/compare-pair")
def compare_text_pair(
    text_a: str = Form(...),
    text_b: str = Form(...),
    model_type: str = Form("sbert")
):
    """
    Direct side-by-side document/text comparison.
    Computes both TF-IDF baseline score and Deep Learning SBERT semantic similarity score.
    """
    start_time = time.time()
    
    # 1. Baseline Comparison
    baseline_res = baseline_matcher.evaluate_pair(text_a, text_b)
    
    # 2. Deep Learning Embeddings Comparison
    if model_type == "multilingual":
        vecs = multilingual_detector.encode([text_a, text_b])
    else:
        vecs = sbert_embedder.encode_sentences([text_a, text_b])
        
    sbert_sim = float(cosine_similarity(vecs[0], vecs[1]))
    
    # 3. Sentence-level alignment between text_a and text_b
    sents_a = TextProcessor.segment_into_sentences(text_a)
    sents_b = TextProcessor.segment_into_sentences(text_b)
    
    vecs_a = sbert_embedder.encode_sentences(sents_a)
    vecs_b = sbert_embedder.encode_sentences(sents_b)
    
    alignments = AttentionAligner.align_sentences(sents_a, sents_b, vecs_a, vecs_b, threshold=0.50)
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "text_a_sentences": len(sents_a),
        "text_b_sentences": len(sents_b),
        "sbert_semantic_similarity": round(sbert_sim, 4),
        "sbert_similarity_percentage": round(sbert_sim * 100, 2),
        "baseline": baseline_res,
        "alignments": alignments,
        "latency_ms": latency_ms
    }


@app.get("/api/corpus/list")
def list_corpus():
    """List indexed corpus document metadata and vector DB statistics."""
    stats = vector_store.get_stats()
    
    # Group metadata by doc_id
    docs = {}
    for meta in vector_store.metadata:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "title": meta.get("title", "Untitled"),
                "author": meta.get("author", "Unknown"),
                "category": meta.get("category", "General"),
                "vector_chunks": 0
            }
        docs[doc_id]["vector_chunks"] += 1
        
    return {
        "stats": stats,
        "documents": list(docs.values())
    }


@app.post("/api/corpus/add")
def add_to_corpus(
    title: str = Form(...),
    author: str = Form("Anonymous"),
    category: str = Form("General"),
    text: str = Form(...)
):
    """Add a new custom document to the internal reference vector corpus."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    doc_id = f"DOC-{uuid.uuid4().hex[:6].upper()}"
    sentences = TextProcessor.segment_into_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found in text.")

    embeddings = sbert_embedder.encode_sentences(sentences)
    metadatas = [
        {
            "doc_id": doc_id,
            "title": title,
            "author": author,
            "category": category,
            "sentence_index": idx,
            "sentence_text": sentence,
            "full_document_text": text
        }
        for idx, sentence in enumerate(sentences)
    ]
    
    vector_store.add_vectors(embeddings, metadatas)

    return {
        "message": "Document added to vector database successfully!",
        "doc_id": doc_id,
        "title": title,
        "vector_chunks_added": len(sentences),
        "total_corpus_vectors": len(vector_store.metadata)
    }


@app.delete("/api/corpus/{doc_id}")
def delete_from_corpus(doc_id: str):
    """Remove a document from the FAISS vector database index."""
    deleted_count = vector_store.remove_document(doc_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document ID not found in vector database.")
    return {
        "message": f"Document {doc_id} deleted successfully.",
        "vectors_deleted": deleted_count,
        "remaining_vectors": len(vector_store.metadata)
    }


@app.post("/api/evaluate-models")
def evaluate_models(threshold: float = Form(0.70)):
    """
    Run evaluation benchmark comparing TF-IDF Baseline vs Sentence-BERT (SBERT).
    Returns KPIs: Accuracy, Precision, Recall, F1-Score, Pearson Correlation & Latency.
    """
    benchmark_results = ModelEvaluator.run_evaluation_benchmark(sbert_embedder, threshold=threshold)
    return benchmark_results


@app.post("/api/train")
def train_model(epochs: int = Form(5), learning_rate: float = Form(2e-5)):
    """Simulate Fine-Tuning Siamese Network over Contrastive / Triplet Loss."""
    train_results = FineTuningTrainer.train_epoch(num_epochs=epochs, learning_rate=learning_rate)
    return train_results
