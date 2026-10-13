# DeepCheck: Deep Learning-based Semantic Plagiarism Detection System

![DeepCheck Banner](https://img.shields.io/badge/DeepCheck-v1.0.0-indigo?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![FAISS](https://img.shields.io/badge/FAISS-Dense%20Vector%20DB-orange?style=for-the-badge)

**DeepCheck** is a state-of-the-art semantic plagiarism detection and sentence paraphrase alignment platform. Built with **Sentence-BERT (SBERT)**, **XLM-RoBERTa**, **FAISS Vector Database**, and **FastAPI**, DeepCheck goes beyond superficial exact string matching to identify context-level plagiarism, synonym swaps, structural paraphrasing, and cross-lingual translations.

---

## 🌟 Key Features

* **🧠 Semantic Paraphrase Detection**: Utilizes Sentence-BERT (`all-MiniLM-L6-v2`) to project sentences into a dense 384-dimensional vector space. Measures contextual similarity via Cosine distance rather than lexical matching.
* **⚡ FAISS Dense Vector Database**: Employs Meta FAISS (`IndexFlatIP`) for sub-millisecond nearest neighbor document chunk retrieval across thousands of indexed reference papers and academic texts.
* **🔥 Interactive Visual Attention Heatmap**: Highlights plagiarized sentences color-coded by severity (High Risk, Paraphrased, Original). Includes an interactive sentence inspector displaying matched reference source quotes and word-level cross-attention overlap chips.
* **⚖️ Direct Pairwise Document Comparator**: Side-by-side comparison between source and target texts with multi-metric outputs (Sentence-BERT similarity, TF-IDF baseline, and Rabin-Karp Jaccard overlap).
* **📚 Reference Corpus Manager**: Live interface to upload, chunk, index, list, and delete reference documents in the FAISS vector database.
* **📊 Quantitative Model Benchmarks**: Runs comparative benchmark suites (Accuracy, Precision, Recall, F1-Score, Pearson Correlation, and Latency) evaluating Deep Learning against traditional string-matching baselines.
* **⚙️ Siamese Architecture Explorer & Fine-Tuning Simulator**: Interactive architecture diagram and Contrastive/Triplet Loss fine-tuning epoch loop simulator.
* **📄 Multi-Format Ingestion**: Supports direct text pasting as well as document file uploads (`PDF`, `DOCX`, `TXT`).

---

## 📂 Project Architecture

```
plagiarism_detection_deep/
├── backend/
│   ├── deepcheck/
│   │   ├── __init__.py
│   │   ├── attention_aligner.py  # Cross-attention weights & heatmap generator
│   │   ├── baseline.py           # TF-IDF N-gram & Rabin-Karp baseline matcher
│   │   ├── corpus_data.py        # Default academic reference corpus data
│   │   ├── cross_lingual.py      # XLM-RoBERTa cross-lingual encoder
│   │   ├── embedder.py           # Sentence-BERT transformer embedder & sliding window
│   │   ├── siamese_net.py        # PyTorch Siamese network & Contrastive/Triplet loss
│   │   ├── trainer.py            # Benchmark evaluator & fine-tuning loop trainer
│   │   └── vector_store.py       # FAISS vector store & NumPy fallback index
│   ├── utils/
│   │   ├── metrics.py            # Cosine similarity & evaluation KPI metrics
│   │   └── text_processor.py     # PDF/DOCX/TXT text extractor & sentence segmenter
│   ├── config.py                 # System paths, model parameters & thresholds
│   ├── main.py                   # FastAPI app entry point & REST endpoints
│   └── __init__.py
├── data/                         # Persistent corpus data & vector indices
├── frontend/
│   ├── css/
│   │   └── style.css             # Glassmorphism dark mode design system
│   ├── js/
│   │   ├── api.js                # API client interface
│   │   ├── app.js                # Main frontend application controller
│   │   ├── charts.js             # Chart.js benchmark visualizations
│   │   └── heatmap.js            # Heatmap & sentence match inspector
│   └── index.html                # Main single-page web app dashboard
├── tests/
│   └── test_engine.py            # PyTest / Unittest automated test suite
├── requirements.txt              # Python dependencies specification
└── README.md                     # Comprehensive running instructions
```

---

## 📋 Requirements & Prerequisites

* **Operating System**: Windows, macOS, or Linux
* **Python**: Python 3.9 or higher
* **Pip**: Latest pip package manager

### Core Python Packages (`requirements.txt`):
* `torch >= 2.0.0`
* `transformers >= 4.30.0`
* `sentence-transformers >= 2.2.2`
* `scikit-learn >= 1.2.0`
* `faiss-cpu >= 1.7.4`
* `fastapi >= 0.95.0`
* `uvicorn >= 0.21.0`
* `python-multipart >= 0.0.6`
* `pypdf >= 3.8.0`
* `python-docx >= 0.8.11`
* `numpy >= 1.24.0`
* `pandas >= 2.0.0`
* `requests >= 2.28.0`

---

## 🚀 Installation & Setup Guide

### 1. Clone or Open the Workspace

Open your shell terminal in the project directory:
```bash
cd "e:/MCA 2K25/plagiarism_detection_deep"
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv venv
python -m venv venv
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Note**: DeepCheck is designed with automatic fallback capabilities. If optional libraries (like GPU PyTorch, SentenceTransformers, or FAISS) are unavailable in your local environment, DeepCheck seamlessly defaults to lightweight PyTorch/NumPy matrix operations so the server always runs without crashing.

---

## 🖥️ Running the Application

### Start the FastAPI Server

Launch the Uvicorn web server from the project root directory:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Or execute directly with uvicorn:
```bash
uvicorn backend.main:app --port 8000 --reload
```

### Open the Application

Once the server initializes and seeds the default academic corpus:

* **Web UI Dashboard**: Open your browser and navigate to:
  [`http://127.0.0.1:8000`](http://127.0.0.1:8000)
* **Interactive OpenAPI (Swagger) Documentation**:
  [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

---

## 🧪 Running Automated Tests

To run the system test suite covering text segmentation, embedding generation, vector database search, baseline string matching, attention alignment, cross-lingual detection, and benchmark evaluations:

### Using Python Unittest:
```bash
python -m unittest tests/test_engine.py
```

### Using PyTest (if installed):
```bash
pytest tests/
```

Expected output:
```
Ran 8 tests in 0.31s

OK
```

---

## 🔌 API Reference & Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the main Glassmorphism Web Interface (`index.html`) |
| `/api/health` | `GET` | Returns system health, hardware status (GPU/CPU), and vector DB stats |
| `/api/check-plagiarism` | `POST` | Primary analysis pipeline for text or uploaded files (`PDF`/`DOCX`/`TXT`) |
| `/api/compare-pair` | `POST` | Side-by-side comparison between Document A and Document B |
| `/api/corpus/list` | `GET` | Retrieves all indexed reference documents and vector store stats |
| `/api/corpus/add` | `POST` | Chunks, embeds, and indexes a new document into FAISS vector database |
| `/api/corpus/{doc_id}` | `DELETE` | Deletes a document and its associated vectors from the index |
| `/api/evaluate-models` | `POST` | Runs evaluation benchmark comparing SBERT vs Baseline matcher |
| `/api/train` | `POST` | Triggers fine-tuning simulation loop over Contrastive Loss |

### Example API Request (`/api/check-plagiarism`):

```bash
curl -X POST "http://127.0.0.1:8000/api/check-plagiarism" \
  -F "text=Deep neural network architectures have altered natural language processing." \
  -F "model_type=sbert" \
  -F "threshold=0.65"
```

---

## 📐 Key Mathematical Formulations

1. **Sentence Embedding via Mean Pooling**:
   $$\mathbf{v} = \text{MeanPooling}(\text{TransformerOutput}(S))$$

2. **Cosine Similarity in Vector Space**:
   $$\text{Sim}(\mathbf{v}_A, \mathbf{v}_B) = \frac{\mathbf{v}_A \cdot \mathbf{v}_B}{\|\mathbf{v}_A\|_2 \|\mathbf{v}_B\|_2}$$

3. **Contrastive Loss (Siamese Fine-Tuning)**:
   $$\mathcal{L}(y, d) = \frac{1}{2} y d^2 + \frac{1}{2} (1 - y) \max(0, m - d)^2$$
   *(where $y \in \{0, 1\}$ indicates binary similarity, $d = \|\mathbf{v}_A - \mathbf{v}_B\|_2$, and $m$ is the distance margin)*

---

## 🤝 Troubleshooting & Tips

* **First Startup**: On the very first run, Sentence-BERT may download the pre-trained weights (`all-MiniLM-L6-v2`, ~80MB). If offline, the built-in fallback embedder instantly takes over.
* **CORS Settings**: CORS is pre-configured to allow requests from any local frontend client.
* **Port Conflicts**: If port `8000` is in use, pass a different port flag: `--port 8080`.

---

Developed for Advanced Academic & Enterprise Plagiarism Detection.
