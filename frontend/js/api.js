/**
 * frontend/js/api.js
 * -------------------
 * API client module for communication with FastAPI backend.
 */

const API = {
  baseUrl: '',

  /**
   * Fetch backend system health status.
   * @returns {Promise<Object>}
   */
  async getHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed with HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API Error (getHealth):', error);
      throw error;
    }
  },

  /**
   * Execute autonomous data quality investigation workflow.
   * @param {Object} payload
   * @param {string} payload.dataset_path
   * @param {string} payload.approval_status
   * @param {string|null} payload.corrected_dataset_path
   * @param {string} [payload.schema_name]
   * @param {boolean} [payload.use_llm]
   * @returns {Promise<Object>}
   */
  async runInvestigation(payload) {
    try {
      const response = await fetch(`${this.baseUrl}/investigate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        const errorDetail = data?.detail || `Investigation failed with HTTP ${response.status}`;
        const errorMsg = Array.isArray(errorDetail)
          ? errorDetail.map(e => e.msg || JSON.stringify(e)).join(', ')
          : String(errorDetail);
        throw new Error(errorMsg);
      }

      return data;
    } catch (error) {
      console.error('API Error (runInvestigation):', error);
      throw error;
    }
  },

  /**
   * Upload local CSV dataset file to FastAPI backend.
   * @param {File} file
   * @returns {Promise<Object>}
   */
  async uploadDataset(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${this.baseUrl}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorDetail = data?.detail || `Upload failed with HTTP ${response.status}`;
        const errorMsg = Array.isArray(errorDetail)
          ? errorDetail.map(e => e.msg || JSON.stringify(e)).join(', ')
          : String(errorDetail);
        throw new Error(errorMsg);
      }

      return data;
    } catch (error) {
      console.error('API Error (uploadDataset):', error);
      throw error;
    }
  },

  /**
   * Get secure download URL for corrected dataset copy.
   * @param {string} [filename]
   * @param {string} [path]
   * @returns {string}
   */
  getCorrectedDownloadUrl(filename, path) {
    if (filename) return `${this.baseUrl}/download/corrected?filename=${encodeURIComponent(filename)}`;
    if (path) return `${this.baseUrl}/download/corrected?path=${encodeURIComponent(path)}`;
    return `${this.baseUrl}/download/corrected`;
  },

  /**
   * Trigger browser download of corrected dataset.
   * @param {string} [filename]
   * @param {string} [path]
   */
  downloadCorrectedDataset(filename, path) {
    const url = this.getCorrectedDownloadUrl(filename, path);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'corrected_dataset.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  },
};

window.API = API;
