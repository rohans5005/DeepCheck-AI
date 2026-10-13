/**
 * DeepCheck State Store
 * Reactive Pub/Sub state management store for DeepCheck SPA.
 */

class Store {
  constructor() {
    this.listeners = new Map();

    // Initial State Definition
    this.state = {
      currentView: 'check',
      theme: localStorage.getItem('deepcheck:theme') || 'dark',
      health: {
        status: 'unknown',
        gpuAvailable: false,
        device: 'Connecting...',
        vectors: 0,
        dim: 384
      },
      currentResult: null,
      history: this.loadHistoryFromStorage(),
      settings: {
        apiUrl: localStorage.getItem('deepcheck:api_url') || 'http://127.0.0.1:8000',
        defaultModel: 'sbert',
        defaultThreshold: 0.65,
        defaultChunkSize: 200
      },
      corpus: {
        stats: null,
        documents: []
      },
      compareResult: null,
      benchmarks: this.loadBenchmarksFromStorage(),
      trainingState: {
        status: 'idle', // 'idle' | 'running' | 'paused'
        currentEpoch: 0,
        totalEpochs: 10,
        logs: [],
        history: []
      }
    };
  }

  getState() {
    return this.state;
  }

  get(key) {
    return this.state[key];
  }

  set(key, value) {
    const oldValue = this.state[key];
    this.state[key] = value;
    this.notify(key, value, oldValue);
  }

  update(key, partialUpdate) {
    const current = this.state[key];
    if (typeof current === 'object' && current !== null) {
      const updated = Array.isArray(current)
        ? [...partialUpdate]
        : { ...current, ...partialUpdate };
      this.set(key, updated);
    } else {
      this.set(key, partialUpdate);
    }
  }

  subscribe(key, listener) {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, new Set());
    }
    this.listeners.get(key).add(listener);

    // Return unsubscribe function
    return () => {
      const set = this.listeners.get(key);
      if (set) set.delete(listener);
    };
  }

  notify(key, newValue, oldValue) {
    const set = this.listeners.get(key);
    if (set) {
      set.forEach(listener => {
        try {
          listener(newValue, oldValue);
        } catch (err) {
          console.error(`Store listener error for key "${key}":`, err);
        }
      });
    }
  }

  // Persistent Storage Utilities
  loadHistoryFromStorage() {
    try {
      const raw = localStorage.getItem('deepcheck:history');
      return raw ? JSON.parse(raw) : [];
    } catch (_) {
      return [];
    }
  }

  saveHistoryToStorage(history) {
    try {
      localStorage.setItem('deepcheck:history', JSON.stringify(history.slice(0, 50)));
    } catch (err) {
      console.warn('Unable to persist history to localStorage:', err);
    }
  }

  loadBenchmarksFromStorage() {
    try {
      const raw = localStorage.getItem('deepcheck:benchmarks');
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  saveBenchmarksToStorage(data) {
    try {
      localStorage.setItem('deepcheck:benchmarks', JSON.stringify(data));
    } catch (_) {}
  }
}

export const store = new Store();
