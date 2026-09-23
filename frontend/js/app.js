/**
 * frontend/js/app.js
 * -------------------
 * Main Application Orchestrator & Event Listener Coordinator.
 */

const App = {
  async init() {
    console.log('Initializing DQ INSIGHT Frontend Application...');

    // 1. Subscribe UI to State Changes
    window.AppState.subscribe((state) => {
      window.UI.renderHealth(state.health);
      if (!state.investigationResult && !state.isInvestigating) {
        window.UI.clearResults();
      }
    });

    // 2. Fetch System Health
    await this.checkHealth();

    // 3. Register Event Listeners
    this.registerEventListeners();

    console.log('DQ INSIGHT Frontend Ready.');
  },

  async checkHealth() {
    try {
      const health = await window.API.getHealth();
      window.AppState.setHealth(health);
    } catch (error) {
      window.AppState.setHealth({ status: 'offline' });
    }
  },

  registerEventListeners() {
    // A. Quick-Select Dataset Pills
    const pills = document.querySelectorAll('.pill-btn');
    const datasetInput = document.getElementById('dataset-path-input');

    pills.forEach(pill => {
      pill.addEventListener('click', () => {
        pills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const path = pill.getAttribute('data-path');
        if (datasetInput) {
          datasetInput.value = path;
          window.AppState.setConfig({ dataset_path: path });
        }
        window.AppState.clearResult();
        this.resetDropZone();
      });
    });

    // B. Analyze Dataset Button
    const btnAnalyze = document.getElementById('btn-analyze-dataset');
    if (btnAnalyze) {
      btnAnalyze.addEventListener('click', () => this.runInvestigation());
    }

    // C. Approval Status Dropdown change (Toggle Corrected Dataset Input)
    const approvalSelect = document.getElementById('approval-status-select');
    const correctedGroup = document.getElementById('corrected-dataset-group');

    if (approvalSelect && correctedGroup) {
      approvalSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        window.AppState.setConfig({ approval_status: val });
        correctedGroup.style.display = val === 'APPROVED' ? 'flex' : 'none';
      });
    }

    // D. Drag and Drop Zone & File Selection
    const dropZone = document.getElementById('dataset-drop-zone');
    const fileInput = document.getElementById('file-input');

    if (dropZone && fileInput) {
      dropZone.addEventListener('click', () => {
        const fi = document.getElementById('file-input');
        if (fi) fi.click();
      });

      ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
          e.preventDefault();
          dropZone.classList.add('dragover');
        });
      });

      ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
          e.preventDefault();
          dropZone.classList.remove('dragover');
        });
      });

      dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
          this.handleFileUpload(files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // E. Accordions for Evidence Section
    const accordions = document.querySelectorAll('.accordion-header');
    accordions.forEach(header => {
      header.addEventListener('click', () => {
        const item = header.parentElement;
        item.classList.toggle('open');
      });
    });

    // F. Copy & Download Report Buttons
    const btnCopy = document.getElementById('btn-copy-report');
    const btnDownload = document.getElementById('btn-download-report');

    if (btnCopy) {
      btnCopy.addEventListener('click', () => {
        const report = window.AppState.investigationResult;
        if (report) {
          const textReport = window.UI.generateHumanReadableTextReport(report);
          navigator.clipboard.writeText(textReport)
            .then(() => alert('Executive Investigation Report copied to clipboard!'))
            .catch(() => alert('Failed to copy report.'));
        }
      });
    }

    if (btnDownload) {
      btnDownload.addEventListener('click', () => {
        const report = window.AppState.investigationResult;
        if (report) {
          const textReport = window.UI.generateHumanReadableTextReport(report);
          const blob = new Blob([textReport], { type: 'text/plain;charset=utf-8' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          const rawName = (report.dataset?.name || report.dataset?.path?.split('/').pop() || 'dataset').replace(/\.[^/.]+$/, '');
          a.href = url;
          a.download = `investigation_report_${rawName}_${Date.now()}.txt`;
          a.click();
          URL.revokeObjectURL(url);
        }
      });
    }

    // G. Hero Section Actions
    const btnNewInv = document.getElementById('btn-hero-new');
    if (btnNewInv) {
      btnNewInv.addEventListener('click', () => {
        const inputCard = document.getElementById('investigation-input-card');
        if (inputCard) inputCard.scrollIntoView({ behavior: 'smooth' });
      });
    }
  },

  async runInvestigation(customApproval = null) {
    const datasetInput = document.getElementById('dataset-path-input');
    const approvalSelect = document.getElementById('approval-status-select');
    const correctedInput = document.getElementById('corrected-path-input');
    const llmToggle = document.getElementById('llm-toggle');

    const datasetPath = datasetInput ? datasetInput.value.trim() : 'data/raw/sales_problematic.csv';
    const approvalStatus = customApproval || (approvalSelect ? approvalSelect.value : 'PENDING');
    
    let correctedPath = null;
    if (approvalStatus === 'APPROVED' && correctedInput && correctedInput.value.trim()) {
      correctedPath = correctedInput.value.trim();
    }

    const useLlm = llmToggle ? llmToggle.checked : false;

    const payload = {
      dataset_path: datasetPath,
      approval_status: approvalStatus,
      corrected_dataset_path: correctedPath,
      schema_name: 'sales',
      use_llm: useLlm,
    };

    window.AppState.clearResult();
    window.AppState.setInvestigating(true);
    this.showLoadingStepper(true);

    try {
      const result = await window.API.runInvestigation(payload);
      window.AppState.setResult(result);
      window.UI.renderInvestigationResults(result);
    } catch (error) {
      alert(`Investigation Error: ${error.message}`);
      window.AppState.setInvestigating(false, error.message);
    } finally {
      this.showLoadingStepper(false);
    }
  },

  executeApprovedWorkflow() {
    const approvalSelect = document.getElementById('approval-status-select');
    if (approvalSelect) approvalSelect.value = 'APPROVED';
    this.runInvestigation('APPROVED');
  },

  executeRejectedWorkflow() {
    const approvalSelect = document.getElementById('approval-status-select');
    if (approvalSelect) approvalSelect.value = 'REJECTED';
    this.runInvestigation('REJECTED');
  },

  async handleFileUpload(file) {
    if (!file) return;

    // Validate client-side extension
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Only .csv files are supported. Please select a valid CSV dataset.');
      return;
    }

    // Validate client-side size (not 0 bytes)
    if (file.size === 0) {
      alert('The selected file is empty. Please select a CSV dataset containing data.');
      return;
    }

    const dropZone = document.getElementById('dataset-drop-zone');
    const datasetInput = document.getElementById('dataset-path-input');
    const pills = document.querySelectorAll('.pill-btn');

    // Visual uploading feedback in drop zone
    if (dropZone) {
      dropZone.innerHTML = `
        <div class="upload-icon" style="animation: spin 1s linear infinite;">⏳</div>
        <div class="upload-text">Uploading ${file.name}...</div>
        <div class="upload-subtext">Uploading to secure project workspace</div>
      `;
    }

    try {
      const response = await window.API.uploadDataset(file);
      const serverPath = response.dataset_path;

      // Populate dataset path input
      if (datasetInput) {
        datasetInput.value = serverPath;
      }
      window.AppState.setConfig({ dataset_path: serverPath });

      // Deselect preset pills
      pills.forEach(p => p.classList.remove('active'));

      // Clear previous investigation state
      window.AppState.clearResult();

      // Update drop zone with success state
      if (dropZone) {
        dropZone.innerHTML = `
          <div class="upload-icon" style="color: var(--success);">✓</div>
          <div class="upload-text" style="color: var(--success); font-weight: 700;">Uploaded: ${file.name}</div>
          <div class="upload-subtext">Saved as <code>${serverPath}</code> (${(response.size_bytes / 1024).toFixed(1)} KB) — Click to choose another file</div>
          <input type="file" id="file-input" accept=".csv" style="display: none;" />
        `;
        const newFileInput = document.getElementById('file-input');
        if (newFileInput) {
          newFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) this.handleFileUpload(e.target.files[0]);
          });
        }
      }
    } catch (error) {
      alert(`Upload Failed: ${error.message}`);
      this.resetDropZone();
    }
  },

  resetDropZone() {
    const dropZone = document.getElementById('dataset-drop-zone');
    if (dropZone) {
      dropZone.innerHTML = `
        <div class="upload-icon">📁</div>
        <div class="upload-text">Drop your CSV dataset here</div>
        <div class="upload-subtext">or click to browse local files</div>
        <input type="file" id="file-input" accept=".csv" style="display: none;" />
      `;
      const newFileInput = document.getElementById('file-input');
      if (newFileInput) {
        newFileInput.addEventListener('change', (e) => {
          if (e.target.files && e.target.files.length > 0) this.handleFileUpload(e.target.files[0]);
        });
      }
    }
  },

  showLoadingStepper(show) {
    const loader = document.getElementById('loading-overlay');
    if (!loader) return;

    if (show) {
      loader.style.display = 'flex';
      const steps = [
        'Profiling dataset...',
        'Detecting statistical anomalies...',
        'Analyzing schema constraints...',
        'Investigating root cause...',
        'Formulating remediation plan...',
        'Evaluating human approval state...',
        'Validating corrections...',
        'Compiling final investigation report...',
      ];

      let currentStep = 0;
      const subtext = document.getElementById('loading-step-text');
      if (subtext) subtext.textContent = steps[0];

      this.stepInterval = setInterval(() => {
        currentStep = (currentStep + 1) % steps.length;
        if (subtext) subtext.textContent = steps[currentStep];
      }, 700);
    } else {
      loader.style.display = 'none';
      if (this.stepInterval) clearInterval(this.stepInterval);
    }
  },
};

window.App = App;

// Bootstrap application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.App.init();
});
