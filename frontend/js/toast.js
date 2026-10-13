/**
 * DeepCheck Stacked Toast Notification System
 * Renders stacked notifications with auto-dismiss, progress indicator, and manual close.
 */

class ToastManager {
  constructor() {
    this.container = document.getElementById('toast-container');
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = 'toast-container';
      this.container.className = 'toast-container';
      document.body.appendChild(this.container);
    }
  }

  show(message, type = 'info', durationMs = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let iconClass = 'fa-solid fa-circle-info text-cyan';
    if (type === 'success') iconClass = 'fa-solid fa-circle-check text-success';
    if (type === 'error') iconClass = 'fa-solid fa-circle-exclamation text-danger';

    toast.innerHTML = `
      <i class="${iconClass}"></i>
      <span class="toast-message">${this.escapeHtml(message)}</span>
      <button class="close-btn" aria-label="Dismiss Toast">&times;</button>
    `;

    const closeBtn = toast.querySelector('.close-btn');
    closeBtn.addEventListener('click', () => this.dismiss(toast));

    this.container.appendChild(toast);

    if (durationMs > 0) {
      setTimeout(() => {
        this.dismiss(toast);
      }, durationMs);
    }

    return toast;
  }

  success(message, duration = 4000) {
    return this.show(message, 'success', duration);
  }

  error(message, duration = 6000) {
    return this.show(message, 'error', duration);
  }

  info(message, duration = 4000) {
    return this.show(message, 'info', duration);
  }

  dismiss(toast) {
    if (!toast || !toast.parentNode) return;
    toast.style.animation = 'slideInToast 200ms reverse forwards';
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 200);
  }

  escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
}

export const toast = new ToastManager();
