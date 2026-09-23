/**
 * frontend/js/state.js
 * ---------------------
 * Central reactive application state container.
 */

const AppState = {
  // System Health
  health: {
    status: 'checking',
    service: '',
    environment: '',
    phase: '',
    agents_active: false,
    llm_configured: false,
  },

  // Current Investigation Configuration
  config: {
    dataset_path: 'data/raw/sales_problematic.csv',
    approval_status: 'PENDING',
    corrected_dataset_path: '',
    schema_name: 'sales',
    use_llm: false,
  },

  // Active Execution State
  isInvestigating: false,
  investigationResult: null,
  error: null,

  // Listeners for state change notifications
  listeners: [],

  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  },

  notify() {
    this.listeners.forEach(listener => listener(this));
  },

  setHealth(healthData) {
    this.health = { ...this.health, ...healthData };
    this.notify();
  },

  setConfig(partialConfig) {
    this.config = { ...this.config, ...partialConfig };
    this.notify();
  },

  setInvestigating(loading, error = null) {
    this.isInvestigating = loading;
    this.error = error;
    this.notify();
  },

  setResult(result) {
    this.investigationResult = result;
    this.isInvestigating = false;
    this.error = null;
    this.notify();
  },

  clearResult() {
    this.investigationResult = null;
    this.error = null;
    this.notify();
  },
};

window.AppState = AppState;
