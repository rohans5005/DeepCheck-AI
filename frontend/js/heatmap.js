/**
 * DeepCheck Heatmap & Sentence Alignment Engine
 * Controls highlighted text view, sentence inspection, word-level token diffs,
 * match density minimap, scroll-locked side-by-side view, and live score recalculation upon citation marking.
 */

import { store } from './store.js';
import { toast } from './toast.js';

export class HeatmapManager {
  constructor() {
    this.currentResult = null;
    this.activeSentenceIndex = null;
    this.currentMatchNavIndex = 0;
    this.matchedSentencesList = [];
    
    this.bindEvents();
  }

  bindEvents() {
    // Drawer Closing Actions
    const closeBtn = document.getElementById('btn-close-inspector');
    const backdrop = document.getElementById('inspector-backdrop');
    if (closeBtn) closeBtn.addEventListener('click', () => this.closeInspector());
    if (backdrop) backdrop.addEventListener('click', () => this.closeInspector());

    // Drawer Buttons
    const btnMarkCitation = document.getElementById('btn-mark-citation');
    const btnExclude = document.getElementById('btn-exclude-sentence');
    const btnCopyInput = document.getElementById('btn-copy-input-sentence');
    const btnCopyCitation = document.getElementById('btn-copy-citation');

    if (btnMarkCitation) btnMarkCitation.addEventListener('click', () => this.toggleMarkCitation());
    if (btnExclude) btnExclude.addEventListener('click', () => this.toggleExcludeSentence());
    if (btnCopyInput) btnCopyInput.addEventListener('click', () => this.copyInputSentence());
    if (btnCopyCitation) btnCopyCitation.addEventListener('click', () => this.copyCitationText());

    // Highlight Filters
    const fltHigh = document.getElementById('flt-high');
    const fltMedium = document.getElementById('flt-medium');
    const fltClean = document.getElementById('flt-clean');
    const docSearch = document.getElementById('doc-search-input');

    [fltHigh, fltMedium, fltClean].forEach(cb => {
      if (cb) cb.addEventListener('change', () => this.applyHighlightedFilters());
    });

    if (docSearch) {
      docSearch.addEventListener('input', () => this.applyHighlightedFilters());
    }

    // Side-by-Side Match Navigation
    const btnPrev = document.getElementById('btn-prev-match');
    const btnNext = document.getElementById('btn-next-match');

    if (btnPrev) btnPrev.addEventListener('click', () => this.navigateMatch(-1));
    if (btnNext) btnNext.addEventListener('click', () => this.navigateMatch(1));

    // Keyboard Shortcuts for Side-by-Side Navigation
    window.addEventListener('keydown', (e) => {
      const currentView = store.get('currentView');
      const dashboardVisible = !document.getElementById('results-dashboard')?.classList.contains('hidden');
      const isSbsActive = document.querySelector('.view-switch-btn[data-results-view="sidebyside"]')?.classList.contains('active');

      if (currentView === 'check' && dashboardVisible && isSbsActive) {
        if (e.key === 'ArrowLeft') {
          this.navigateMatch(-1);
        } else if (e.key === 'ArrowRight') {
          this.navigateMatch(1);
        }
      }
    });
  }

  /**
   * Render complete result dataset into all views
   */
  renderResult(data) {
    this.currentResult = data;
    store.set('currentResult', data);

    // 1. Render Hero Score Donut Gauge & Verdict
    this.renderHeroScoreCard(data);

    // 2. Render Highlighted Text View & Minimap
    this.renderHighlightedTextView(data);
    this.renderMinimap(data);

    // 3. Render Side-by-Side View
    this.renderSideBySideView(data);

    // 4. Render Sources Table
    this.renderSourcesTable(data);
  }

  renderHeroScoreCard(data) {
    const scoreVal = data.overall_similarity_score || 0;
    const scoreTextEl = document.getElementById('gauge-score-text');
    const fillCircle = document.getElementById('gauge-fill-circle');
    const riskPill = document.getElementById('risk-pill-badge');
    const verdictTitle = document.getElementById('hero-verdict-title');
    const verdictDesc = document.getElementById('hero-verdict-desc');
    const filenameTag = document.getElementById('results-filename-tag');

    if (filenameTag) filenameTag.textContent = data.filename || 'Document.txt';

    // Animate Number from 0 to scoreVal over 900ms
    this.animateNumber(scoreTextEl, scoreVal, '%');

    // Circle circumference for r=68 is 2 * PI * 68 = ~427.26
    const circumference = 427.26;
    const offset = circumference - (scoreVal / 100) * circumference;

    if (fillCircle) {
      fillCircle.style.strokeDasharray = `${circumference}`;
      fillCircle.style.strokeDashoffset = `${offset}`;

      // Color banding based on risk score
      if (scoreVal >= 40) {
        fillCircle.style.stroke = 'var(--risk-high)';
      } else if (scoreVal >= 15) {
        fillCircle.style.stroke = 'var(--risk-medium)';
      } else {
        fillCircle.style.stroke = 'var(--risk-clean)';
      }
    }

    // Risk Pill & Verdict text
    if (riskPill && verdictTitle && verdictDesc) {
      if (scoreVal >= 40) {
        riskPill.className = 'risk-pill high';
        riskPill.textContent = 'High Plagiarism Risk';
        verdictTitle.textContent = 'Significant Overlap Detected';
        verdictDesc.textContent = 'High level of matching text found against indexed reference sources. Review flagged passages.';
      } else if (scoreVal >= 15) {
        riskPill.className = 'risk-pill medium';
        riskPill.textContent = 'Moderate Overlap';
        verdictTitle.textContent = 'Moderate Similarity Detected';
        verdictDesc.textContent = 'Several paraphrased or structural similarities detected. Ensure proper citation.';
      } else {
        riskPill.className = 'risk-pill clean';
        riskPill.textContent = 'Original Content';
        verdictTitle.textContent = 'Mostly Original Document';
        verdictDesc.textContent = 'Low similarity score. Content appears highly original with minimal overlap.';
      }
    }

    // Stat tiles
    document.getElementById('stat-total-sentences').textContent = data.total_sentences || 0;
    document.getElementById('stat-flagged-sentences').textContent =
      (data.severity_breakdown?.high || 0) + (data.severity_breakdown?.medium || 0);
    document.getElementById('stat-sources-count').textContent = data.matched_sources?.length || 0;

    const plagiarizedSents = (data.sentence_alignments || []).filter(s => s.is_plagiarized && !s.is_excluded && !s.is_citation);
    const avgSim = plagiarizedSents.length
      ? Math.round(plagiarizedSents.reduce((acc, curr) => acc + curr.similarity_score, 0) / plagiarizedSents.length * 100)
      : 0;
    document.getElementById('stat-avg-similarity').textContent = `${avgSim}%`;
    document.getElementById('stat-latency').textContent = `${data.latency_ms || 0}ms`;
    document.getElementById('stat-model-name').textContent = (data.model_used || 'SBERT').toUpperCase();
    document.getElementById('source-list-badge-cnt').textContent = data.matched_sources?.length || 0;
  }

  animateNumber(element, target, suffix = '') {
    if (!element) return;
    const start = 0;
    const duration = 900;
    const startTime = performance.now();

    const update = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const current = Math.round(start + (target - start) * (1 - Math.pow(1 - progress, 3)));
      element.textContent = `${current}${suffix}`;
      if (progress < 1) requestAnimationFrame(update);
    };

    requestAnimationFrame(update);
  }

  /**
   * Render inline sentence spans in Highlighted Text View
   */
  renderHighlightedTextView(data) {
    const container = document.getElementById('document-text-container');
    if (!container) return;
    container.innerHTML = '';

    const alignments = data.sentence_alignments || [];
    alignments.forEach((align, idx) => {
      const span = document.createElement('span');
      span.className = `sent-span ${align.severity_level || 'clean'}`;
      if (align.is_citation) span.classList.add('citation');
      span.dataset.index = idx;
      span.textContent = align.input_sentence + ' ';

      // Tooltip preview
      if (align.is_plagiarized && align.matched_source) {
        const srcTitle = align.matched_source.title || 'Source';
        span.title = `[${align.similarity_percentage}% Match] ${srcTitle} — Click to inspect`;
      }

      span.addEventListener('click', () => this.openInspector(idx));
      container.appendChild(span);
    });
  }

  applyHighlightedFilters() {
    const showHigh = document.getElementById('flt-high')?.checked;
    const showMedium = document.getElementById('flt-medium')?.checked;
    const showClean = document.getElementById('flt-clean')?.checked;
    const query = document.getElementById('doc-search-input')?.value.toLowerCase().trim() || '';

    const spans = document.querySelectorAll('#document-text-container .sent-span');
    spans.forEach(span => {
      const isHigh = span.classList.contains('high');
      const isMedium = span.classList.contains('medium');
      const isClean = span.classList.contains('clean');
      const text = span.textContent.toLowerCase();

      let visible = false;
      if (isHigh && showHigh) visible = true;
      if (isMedium && showMedium) visible = true;
      if (isClean && showClean) visible = true;

      if (query && !text.includes(query)) {
        visible = false;
      }

      span.style.display = visible ? 'inline' : 'none';
    });
  }

  /**
   * Render Sticky Minimap Density Bar
   */
  renderMinimap(data) {
    const track = document.getElementById('minimap-track');
    if (!track) return;
    track.innerHTML = '';

    const alignments = data.sentence_alignments || [];
    alignments.forEach((align, idx) => {
      const block = document.createElement('div');
      block.className = `minimap-block ${align.severity_level || 'clean'}`;
      block.title = `Sentence #${idx + 1}: ${align.similarity_percentage || 0}%`;
      block.addEventListener('click', () => {
        const targetSpan = document.querySelector(`.sent-span[data-index="${idx}"]`);
        if (targetSpan) {
          targetSpan.scrollIntoView({ behavior: 'smooth', block: 'center' });
          targetSpan.classList.add('active-inspect');
          setTimeout(() => targetSpan.classList.remove('active-inspect'), 1500);
        }
      });
      track.appendChild(block);
    });
  }

  /**
   * Render Side-by-Side View
   */
  renderSideBySideView(data) {
    const leftCol = document.getElementById('sbs-left-content');
    const rightCol = document.getElementById('sbs-right-content');
    if (!leftCol || !rightCol) return;

    leftCol.innerHTML = '';
    rightCol.innerHTML = '';

    const alignments = data.sentence_alignments || [];
    this.matchedSentencesList = alignments.filter(s => s.is_plagiarized);
    this.currentMatchNavIndex = 0;

    alignments.forEach((align, idx) => {
      const pLeft = document.createElement('div');
      pLeft.className = `sbs-sent-item ${align.severity_level || 'clean'}`;
      pLeft.id = `sbs-left-${idx}`;
      pLeft.innerHTML = `<strong>#${idx + 1}:</strong> ${align.input_sentence}`;
      leftCol.appendChild(pLeft);

      const pRight = document.createElement('div');
      pRight.className = `sbs-sent-item ${align.severity_level || 'clean'}`;
      pRight.id = `sbs-right-${idx}`;
      if (align.matched_source) {
        pRight.innerHTML = `<strong>Source:</strong> ${align.matched_source.sentence_text || 'Matched passage'}<br><small class="text-muted">${align.matched_source.title || ''}</small>`;
      } else {
        pRight.innerHTML = `<span class="text-muted">Original sentence — no direct source match above threshold.</span>`;
      }
      rightCol.appendChild(pRight);
    });

    this.updateMatchStepIndicator();
  }

  navigateMatch(direction) {
    if (!this.matchedSentencesList.length) return;

    this.currentMatchNavIndex += direction;
    if (this.currentMatchNavIndex < 0) this.currentMatchNavIndex = 0;
    if (this.currentMatchNavIndex >= this.matchedSentencesList.length) {
      this.currentMatchNavIndex = this.matchedSentencesList.length - 1;
    }

    const currentMatch = this.matchedSentencesList[this.currentMatchNavIndex];
    if (!currentMatch) return;

    const idx = currentMatch.sentence_index;
    const leftEl = document.getElementById(`sbs-left-${idx}`);
    const rightEl = document.getElementById(`sbs-right-${idx}`);

    if (leftEl) leftEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    if (rightEl) rightEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

    this.updateMatchStepIndicator();
  }

  updateMatchStepIndicator() {
    const indicator = document.getElementById('match-step-indicator');
    if (!indicator) return;
    const total = this.matchedSentencesList.length;
    const current = total > 0 ? this.currentMatchNavIndex + 1 : 0;
    indicator.textContent = `Match ${current} of ${total}`;
  }

  /**
   * Render Sources Table View
   */
  renderSourcesTable(data) {
    const tbody = document.getElementById('sources-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    const sources = data.matched_sources || [];
    if (!sources.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted" style="padding:2rem;">No matching corpus sources detected.</td></tr>`;
      return;
    }

    sources.forEach(src => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${this.escapeHtml(src.title || 'Untitled Source')}</strong></td>
        <td>${this.escapeHtml(src.author || 'Unknown Author')}</td>
        <td><span class="badge badge-secondary">${this.escapeHtml(src.category || 'General')}</span></td>
        <td><span class="badge badge-accent">${src.match_count || 1} passages</span></td>
        <td><strong class="text-high">${Math.round((src.max_similarity || 0) * 100)}%</strong></td>
        <td>
          <button class="btn btn-secondary btn-sm view-src-passages-btn" data-docid="${src.doc_id}">
            <i class="fa-solid fa-eye"></i> Excerpts
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  /**
   * Open Sentence Inspector Drawer
   */
  openInspector(sentenceIndex) {
    if (!this.currentResult) return;
    const alignments = this.currentResult.sentence_alignments || [];
    const item = alignments[sentenceIndex];
    if (!item) return;

    this.activeSentenceIndex = sentenceIndex;

    const backdrop = document.getElementById('inspector-backdrop');
    const drawer = document.getElementById('inspector-drawer');

    document.getElementById('insp-sentence-idx').textContent = sentenceIndex + 1;
    document.getElementById('insp-input-sentence').textContent = item.input_sentence;

    const sourceSentEl = document.getElementById('insp-source-sentence');
    const sourceTitleEl = document.getElementById('insp-source-title');

    if (item.matched_source) {
      sourceSentEl.textContent = item.matched_source.sentence_text || 'Matched sentence text unavailable.';
      sourceTitleEl.textContent = `Source: ${item.matched_source.title || 'Academic Corpus'}`;
    } else {
      sourceSentEl.textContent = 'No matching corpus sentence found above similarity threshold.';
      sourceTitleEl.textContent = 'Source: N/A';
    }

    const simScore = item.similarity_percentage || 0;
    document.getElementById('insp-sim-val').textContent = `${simScore}%`;
    const simFill = document.getElementById('insp-sim-fill');
    if (simFill) {
      simFill.style.width = `${simScore}%`;
      simFill.style.background = simScore >= 70 ? 'var(--risk-high)' : 'var(--risk-medium)';
    }

    // Word-Level Token Overlap Chips
    this.renderTokenChips(item);

    // Metadata
    document.getElementById('insp-para-type').textContent = item.paraphrase_type || 'Original Content';
    document.getElementById('insp-cosine-score').textContent = (item.similarity_score || 0).toFixed(4);
    document.getElementById('insp-severity').textContent = (item.severity_level || 'none').toUpperCase();

    // Citation button state
    const btnMark = document.getElementById('btn-mark-citation');
    const btnExclude = document.getElementById('btn-exclude-sentence');
    if (btnMark) btnMark.innerHTML = item.is_citation ? `<i class="fa-solid fa-rotate-left"></i> Unmark Citation` : `<i class="fa-solid fa-quote-right"></i> Mark as Citation`;
    if (btnExclude) btnExclude.innerHTML = item.is_excluded ? `<i class="fa-solid fa-eye"></i> Include in Score` : `<i class="fa-solid fa-eye-slash"></i> Exclude from Score`;

    if (backdrop) backdrop.classList.remove('hidden');
    if (drawer) drawer.classList.remove('hidden');
  }

  closeInspector() {
    const backdrop = document.getElementById('inspector-backdrop');
    const drawer = document.getElementById('inspector-drawer');
    if (backdrop) backdrop.classList.add('hidden');
    if (drawer) drawer.classList.add('hidden');
  }

  /**
   * Calculate client-side token diff and render word chips
   */
  renderTokenChips(item) {
    const container = document.getElementById('insp-token-chips');
    if (!container) return;
    container.innerHTML = '';

    const words = item.word_attentions || [];
    if (!words.length) {
      // Fallback tokenize sentence
      const rawWords = item.input_sentence.split(/\s+/);
      rawWords.forEach(w => {
        const chip = document.createElement('span');
        chip.className = 'token-chip normal';
        chip.textContent = w;
        container.appendChild(chip);
      });
      return;
    }

    words.forEach(wObj => {
      const chip = document.createElement('span');
      const weight = wObj.attention_weight || 0;
      if (weight >= 0.7) {
        chip.className = 'token-chip matched';
      } else {
        chip.className = 'token-chip normal';
      }
      chip.textContent = wObj.word;
      container.appendChild(chip);
    });
  }

  toggleMarkCitation() {
    if (this.activeSentenceIndex === null || !this.currentResult) return;
    const align = this.currentResult.sentence_alignments[this.activeSentenceIndex];
    if (!align) return;

    align.is_citation = !align.is_citation;
    if (align.is_citation) align.is_plagiarized = false;

    toast.info(align.is_citation ? 'Sentence marked as valid citation.' : 'Sentence unmarked.');
    this.recomputeOverallScore();
    this.openInspector(this.activeSentenceIndex);
  }

  toggleExcludeSentence() {
    if (this.activeSentenceIndex === null || !this.currentResult) return;
    const align = this.currentResult.sentence_alignments[this.activeSentenceIndex];
    if (!align) return;

    align.is_excluded = !align.is_excluded;
    toast.info(align.is_excluded ? 'Sentence excluded from overall plagiarism score.' : 'Sentence included.');
    this.recomputeOverallScore();
    this.openInspector(this.activeSentenceIndex);
  }

  /**
   * Recalculate displayed overall plagiarism score live
   */
  recomputeOverallScore() {
    if (!this.currentResult) return;
    const alignments = this.currentResult.sentence_alignments || [];
    const validAlignments = alignments.filter(a => !a.is_excluded && !a.is_citation);

    if (!validAlignments.length) {
      this.currentResult.overall_similarity_score = 0;
    } else {
      const plagiarized = validAlignments.filter(a => a.similarity_score >= (this.currentResult.threshold_used || 0.65));
      const scores = plagiarized.map(a => a.similarity_score);
      const meanScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / validAlignments.length) : 0;
      this.currentResult.overall_similarity_score = Math.round(meanScore * 100);
    }

    this.renderHeroScoreCard(this.currentResult);
    this.renderHighlightedTextView(this.currentResult);
    this.renderMinimap(this.currentResult);
  }

  copyInputSentence() {
    if (this.activeSentenceIndex === null || !this.currentResult) return;
    const sentence = this.currentResult.sentence_alignments[this.activeSentenceIndex]?.input_sentence;
    if (sentence) {
      navigator.clipboard.writeText(sentence);
      toast.success('Sentence copied to clipboard!');
    }
  }

  copyCitationText() {
    if (this.activeSentenceIndex === null || !this.currentResult) return;
    const item = this.currentResult.sentence_alignments[this.activeSentenceIndex];
    if (!item) return;

    const citationText = `"${item.input_sentence}" — Adapted from ${item.matched_source?.title || 'Corpus Source'} (${item.matched_source?.author || 'Unknown'}).`;
    navigator.clipboard.writeText(citationText);
    toast.success('Formatted citation copied to clipboard!');
  }

  escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
}

export const heatmap = new HeatmapManager();
