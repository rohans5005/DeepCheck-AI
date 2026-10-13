/**
 * DeepCheck API Client Engine
 * Handles all backend communication, timeout management, retries, multipart encoding, and normalized error handling.
 * 
 * Assumption on backend response shapes:
 * - /api/health: { status, gpu_available, device, sbert_model, multilingual_model, vector_store: { total_vectors, index_dim } }
 * - /api/check-plagiarism: { filename, total_characters, total_sentences, overall_similarity_score, plagiarized_sentence_percentage, severity_breakdown, matched_sources, sentence_alignments, latency_ms }
 * - /api/compare-pair: { text_a_sentences, text_b_sentences, sbert_semantic_similarity, sbert_similarity_percentage, baseline, alignments, latency_ms }
 * - /api/corpus/list: { stats, documents: [{ doc_id, title, author, category, vector_chunks }] }
 * - /api/corpus/add: { message, doc_id, title, vector_chunks_added, total_corpus_vectors }
 * - /api/corpus/{doc_id}: { message, vectors_deleted, remaining_vectors }
 * - /api/evaluate-models: { baseline, deep_learning_sbert, benchmark_dataset_size, similarity_scores_breakdown }
 * - /api/train: { num_epochs, final_loss, final_accuracy, final_f1_score, history: [{ epoch, loss, accuracy, f1_score, learning_rate }] }
 */

class ApiClient {
  constructor() {
    this.baseUrl = localStorage.getItem('deepcheck:api_url') || 'http://127.0.0.1:8000';
    this.defaultTimeoutMs = 60000;
  }

  setBaseUrl(url) {
    this.baseUrl = url.replace(/\/+$/, '');
    localStorage.setItem('deepcheck:api_url', this.baseUrl);
  }

  getBaseUrl() {
    return this.baseUrl;
  }

  /**
   * Universal Request Wrapper with timeout and retry logic
   */
  async request(path, options = {}) {
    const {
      method = 'GET',
      body = null,
      headers = {},
      signal = null,
      timeout = this.defaultTimeoutMs,
      retries = 2
    } = options;

    const url = `${this.baseUrl}${path.startsWith('/') ? path : '/' + path}`;

    // Combine user signal with timeout signal
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    const onExternalAbort = () => controller.abort();
    if (signal) {
      signal.addEventListener('abort', onExternalAbort);
    }

    let attempt = 0;
    let lastError = null;

    while (attempt <= retries) {
      try {
        const fetchOptions = {
          method,
          headers,
          body,
          signal: controller.signal
        };

        const response = await fetch(url, fetchOptions);
        clearTimeout(timer);
        if (signal) signal.removeEventListener('abort', onExternalAbort);

        if (!response.ok) {
          let errorDetail = 'API request failed';
          try {
            const errJson = await response.json();
            errorDetail = errJson.detail || errJson.message || JSON.stringify(errJson);
          } catch (_) {
            errorDetail = await response.text();
          }

          const normalizedError = {
            status: response.status,
            message: `HTTP ${response.status}: ${response.statusText}`,
            detail: errorDetail || 'Server error occurred'
          };

          // Retry on 5xx server errors if attempts remain
          if (response.status >= 500 && attempt < retries) {
            attempt++;
            await this.sleep(Math.pow(2, attempt) * 500);
            continue;
          }

          throw normalizedError;
        }

        const data = await response.json();
        return data;

      } catch (err) {
        clearTimeout(timer);
        if (signal) signal.removeEventListener('abort', onExternalAbort);

        if (err.name === 'AbortError') {
          throw {
            status: 0,
            message: 'Request Aborted',
            detail: 'The operation was cancelled by the user or timed out after 60 seconds.'
          };
        }

        lastError = err.status !== undefined ? err : {
          status: 0,
          message: 'Network / Connection Error',
          detail: err.message || 'Unable to reach backend server. Check if FastAPI is running.'
        };

        if (attempt < retries && lastError.status === 0) {
          attempt++;
          await this.sleep(Math.pow(2, attempt) * 500);
          continue;
        }

        throw lastError;
      }
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // API Methods matching backend contract

  async getHealth(signal = null) {
    return this.request('/api/health', { signal, retries: 1, timeout: 5000 });
  }

  async checkPlagiarism({ text, file, modelType = 'sbert', threshold = 0.65, topK = 5, signal = null }) {
    const formData = new FormData();
    if (text) formData.append('text', text);
    if (file) formData.append('file', file);
    formData.append('model_type', modelType);
    formData.append('threshold', threshold.toString());
    formData.append('top_k', topK.toString());

    return this.request('/api/check-plagiarism', {
      method: 'POST',
      body: formData,
      signal
    });
  }

  async comparePair({ textA, textB, modelType = 'sbert', signal = null }) {
    const formData = new FormData();
    formData.append('text_a', textA);
    formData.append('text_b', textB);
    formData.append('model_type', modelType);

    return this.request('/api/compare-pair', {
      method: 'POST',
      body: formData,
      signal
    });
  }

  async listCorpus(signal = null) {
    return this.request('/api/corpus/list', { signal });
  }

  async addCorpus({ title, author, category, text }, signal = null) {
    const formData = new FormData();
    formData.append('title', title);
    formData.append('author', author || 'Anonymous');
    formData.append('category', category || 'General');
    formData.append('text', text);

    return this.request('/api/corpus/add', {
      method: 'POST',
      body: formData,
      signal
    });
  }

  async deleteCorpus(docId, signal = null) {
    return this.request(`/api/corpus/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
      signal
    });
  }

  async evaluateModels(threshold = 0.70, signal = null) {
    const formData = new FormData();
    formData.append('threshold', threshold.toString());

    return this.request('/api/evaluate-models', {
      method: 'POST',
      body: formData,
      signal
    });
  }

  async trainModel(epochs = 5, learningRate = 2e-5, signal = null) {
    const formData = new FormData();
    formData.append('epochs', epochs.toString());
    formData.append('learning_rate', learningRate.toString());

    return this.request('/api/train', {
      method: 'POST',
      body: formData,
      signal
    });
  }
}

export const api = new ApiClient();
