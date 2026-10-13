/**
 * DeepCheck Hash Router
 * Single Page Application Hash Router supporting view switching and tab activation.
 */

import { store } from './store.js';

class Router {
  constructor() {
    this.routes = ['check', 'compare', 'corpus', 'benchmarks', 'training', 'history', 'settings'];
    this.defaultRoute = 'check';
  }

  init() {
    window.addEventListener('hashchange', () => this.handleHashChange());
    // Initial Route Resolution
    this.handleHashChange();
  }

  handleHashChange() {
    const rawHash = window.location.hash.replace('#', '').trim();
    const targetRoute = this.routes.includes(rawHash) ? rawHash : this.defaultRoute;
    this.navigateTo(targetRoute, false);
  }

  navigateTo(viewName, updateHash = true) {
    if (!this.routes.includes(viewName)) {
      viewName = this.defaultRoute;
    }

    if (updateHash && window.location.hash !== `#${viewName}`) {
      window.location.hash = `#${viewName}`;
      return;
    }

    // Update Store
    store.set('currentView', viewName);

    // Update View DOM Visibility
    this.routes.forEach(route => {
      const el = document.getElementById(`view-${route}`);
      const tab = document.getElementById(`tab-${route}`);

      if (route === viewName) {
        if (el) {
          el.classList.remove('hidden');
          el.classList.add('active');
        }
        if (tab) {
          tab.classList.add('active');
          tab.setAttribute('aria-selected', 'true');
        }
      } else {
        if (el) {
          el.classList.add('hidden');
          el.classList.remove('active');
        }
        if (tab) {
          tab.classList.remove('active');
          tab.setAttribute('aria-selected', 'false');
        }
      }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

export const router = new Router();
