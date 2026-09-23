/**
 * frontend/js/ui.js
 * ------------------
 * DOM Rendering Engine for Data Quality Dashboard.
 */

const UI = {
  /**
   * Render System Health in Header and Sidebar.
   * @param {Object} health
   */
  renderHealth(health) {
    const apiBadge = document.getElementById('api-health-badge');
    const llmBadge = document.getElementById('llm-status-badge');
    const systemIndicator = document.getElementById('system-status-indicator');

    if (apiBadge) {
      if (health.status === 'healthy') {
        apiBadge.className = 'badge badge-success';
        apiBadge.innerHTML = '● API Online';
      } else {
        apiBadge.className = 'badge badge-critical';
        apiBadge.innerHTML = '✕ API Offline';
      }
    }

    if (llmBadge) {
      if (health.llm_configured) {
        llmBadge.className = 'badge badge-info';
        llmBadge.innerHTML = '✦ Gemini AI Active';
      } else {
        llmBadge.className = 'badge badge-neutral';
        llmBadge.innerHTML = '⚙ Deterministic Mode';
      }
    }

    if (systemIndicator) {
      systemIndicator.innerHTML = health.status === 'healthy'
        ? '<span class="status-dot"></span> System Ready'
        : '<span class="status-dot" style="background-color: var(--critical); box-shadow: 0 0 8px var(--critical);"></span> System Offline';
    }
  },

  /**
   * Reset and clear previous investigation UI.
   */
  clearResults() {
    const emptyState = document.getElementById('empty-state');
    const resultsContainer = document.getElementById('results-container');
    if (emptyState) emptyState.style.display = 'block';
    if (resultsContainer) resultsContainer.style.display = 'none';

    ['kpi-missing', 'kpi-duplicates', 'kpi-anomalies', 'kpi-schema'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '—';
    });

    const fields = [
      'root-cause-primary', 'root-cause-confidence', 'root-cause-score', 'root-cause-reasoning',
      'evidence-profiling', 'evidence-anomaly', 'evidence-schema', 'evidence-root-cause',
      'final-report-text', 'execution-trace-json', 'final-recommendation-text',
      'validation-checks-list', 'before-after-container', 'validation-verdict-banner',
      'human-report-container'
    ];
    fields.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '';
    });
  },

  /**
   * Render Main Dashboard after Workflow Execution.
   * @param {Object} report Final Workflow Report JSON
   */
  renderInvestigationResults(report) {
    if (!report) return;

    // Show Results Container & Hide Empty State
    const emptyState = document.getElementById('empty-state');
    const resultsContainer = document.getElementById('results-container');
    if (emptyState) emptyState.style.display = 'none';
    if (resultsContainer) resultsContainer.style.display = 'block';

    // 1. KPI Cards
    this.renderKPIs(report.detected_issues || {});

    // 2. Dataset Overview
    this.renderDatasetProfile(report);

    // 3. Workflow Timeline
    this.renderWorkflowTimeline(report.execution_trace || [], report.workflow_status);

    // 4. Root Cause Investigation
    this.renderRootCause(report.root_cause || {});

    // 5. Evidence Accordions
    this.renderEvidence(report);

    // 6. Correction Recommendations
    this.renderRecommendations(report.correction_recommendation || {});

    // 7. Human Approval Gate
    this.renderApprovalGate(report);

    // 7b. Remediation Audit & Data Changes
    this.renderRemediationAudit(report);

    // 7c. Correction Result Card (Download & Summary)
    this.renderCorrectionResult(report);

    // 8. Validation Dashboard
    this.renderValidation(report.validation_status || {}, report.unresolved_issues || []);

    // 9. Final Human-Readable Report & Technical Trace
    this.renderFinalReport(report);

    // Smooth scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  /**
   * Render Top KPI Metric Cards.
   * @param {Object} issues
   */
  renderKPIs(issues) {
    const missing = issues.missing_values_count || 0;
    const dups = issues.duplicate_rows_count || 0;
    const anoms = issues.anomalous_dates_count || 0;
    const schemas = issues.schema_violations_count || 0;

    const kpiMissing = document.getElementById('kpi-missing');
    const kpiDups = document.getElementById('kpi-duplicates');
    const kpiAnoms = document.getElementById('kpi-anomalies');
    const kpiSchema = document.getElementById('kpi-schema');

    if (kpiMissing) kpiMissing.textContent = missing.toLocaleString();
    if (kpiDups) kpiDups.textContent = dups.toLocaleString();
    if (kpiAnoms) kpiAnoms.textContent = anoms.toLocaleString();
    if (kpiSchema) kpiSchema.textContent = schemas.toLocaleString();

    // Visual Charts
    const chartContainer = document.getElementById('issues-chart-container');
    if (chartContainer && window.Charts) {
      window.Charts.renderIssueDistribution(chartContainer, issues);
    }
  },

  /**
   * Render Dataset Profile summary.
   * @param {Object} report
   */
  renderDatasetProfile(report) {
    const dataset = report.dataset || {};
    const dsName = document.getElementById('profile-dataset-name');
    const dsStatus = document.getElementById('profile-workflow-status');
    const dsSummary = document.getElementById('profile-summary-text');

    if (dsName) dsName.textContent = dataset.name || dataset.path || 'Sales Dataset';
    if (dsStatus) {
      dsStatus.textContent = report.workflow_status;
      dsStatus.className = `badge ${this.getWorkflowStatusBadge(report.workflow_status)}`;
    }
    if (dsSummary) {
      dsSummary.textContent = report.detected_issues?.profiling_summary || 'Profiling analysis completed successfully.';
    }
  },

  /**
   * Render Agent Workflow visual timeline.
   * @param {Array} trace
   * @param {string} workflowStatus
   */
  renderWorkflowTimeline(trace, workflowStatus) {
    const timelineContainer = document.getElementById('workflow-timeline-container');
    if (!timelineContainer) return;

    const traceMap = {};
    (trace || []).forEach(step => {
      const nodeKey = step.node.replace('_node', '');
      traceMap[nodeKey] = step.status;
    });

    const nodes = [
      { id: 'profiling', name: 'Profiling', icon: '📊' },
      { id: 'anomaly', name: 'Anomaly', icon: '📈' },
      { id: 'schema', name: 'Schema', icon: '🔍' },
      { id: 'root_cause', name: 'Root Cause', icon: '🧠' },
      { id: 'correction', name: 'Correction', icon: '🛠' },
      { id: 'approval', name: 'Approval', icon: '👤' },
      { id: 'validation', name: 'Validation', icon: '✓' },
      { id: 'final_report', name: 'Final Report', icon: '📄' },
    ];

    let html = '';
    nodes.forEach((node, idx) => {
      const status = traceMap[node.id] || (workflowStatus === 'NO_CORRECTION_REQUIRED' && ['correction', 'approval', 'validation'].includes(node.id) ? 'skipped' : 'skipped');
      const isCompleted = ['completed', 'approved'].includes(status);
      const isPending = status === 'pending';
      const isSkipped = status === 'skipped';
      const isFailed = status === 'failed';

      let statusClass = 'status-waiting';
      let badgeLabel = 'WAITING';
      let badgeClass = 'badge-neutral';

      if (isCompleted) {
        statusClass = 'status-completed';
        badgeLabel = 'COMPLETED';
        badgeClass = 'badge-success';
      } else if (isPending) {
        statusClass = 'status-pending';
        badgeLabel = 'PENDING';
        badgeClass = 'badge-warning';
      } else if (isSkipped) {
        statusClass = 'status-skipped';
        badgeLabel = 'SKIPPED';
        badgeClass = 'badge-neutral';
      } else if (isFailed) {
        statusClass = 'status-failed';
        badgeLabel = 'FAILED';
        badgeClass = 'badge-critical';
      }

      html += `
        <div class="workflow-node ${statusClass}">
          <div class="node-icon-circle">${node.icon}</div>
          <div class="node-name">${node.name}</div>
          <span class="badge ${badgeClass} node-status-badge">${badgeLabel}</span>
        </div>
      `;

      if (idx < nodes.length - 1) {
        const nextNode = nodes[idx + 1];
        const nextStatus = traceMap[nextNode.id];
        const connectorActive = isCompleted && nextStatus && nextStatus !== 'skipped';
        html += `<div class="workflow-connector ${connectorActive ? 'active' : ''}"></div>`;
      }
    });

    timelineContainer.innerHTML = html;
  },

  /**
   * Render Remediation Audit and Data Change Summary
   * @param {Object} report
   */
  renderRemediationAudit(report) {
    const auditSection = document.getElementById('audit-section');
    if (!auditSection) return;

    if (!report.approval_record && !report.data_change_summary) {
      auditSection.style.display = 'none';
      return;
    }

    auditSection.style.display = 'block';

    // 1. Human Approval Record
    const approvalGrid = document.getElementById('approval-record-grid');
    if (approvalGrid && report.approval_record) {
      const rec = report.approval_record;
      approvalGrid.innerHTML = `
        <div class="metric-card">
          <div class="metric-label">Approver</div>
          <div class="metric-value" style="font-size: 1.25rem;">${rec.approver || 'Human Operator'}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Status</div>
          <div class="metric-value" style="font-size: 1.25rem; color: var(--success);">${rec.status || 'APPROVED'}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Timestamp (UTC)</div>
          <div class="metric-value" style="font-size: 1.1rem;">${new Date(rec.timestamp).toLocaleString()}</div>
        </div>
      `;
    }

    // 2. Approved Remediation Actions
    const actionsList = document.getElementById('approved-actions-list');
    if (actionsList && report.approval_record && report.approval_record.approved_actions) {
      let html = '';
      report.approval_record.approved_actions.forEach(act => {
        html += `
          <div style="background-color: var(--bg-primary); padding: 1rem; border: 1px solid var(--border-color); border-radius: 4px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
              <strong style="color: var(--text-primary);">${act.description || act.action_type}</strong>
              <span class="badge badge-success">${act.status}</span>
            </div>
            <div style="font-size: 0.85rem; color: var(--text-secondary);">
              Priority: <strong style="color: var(--text-primary);">${act.priority}</strong> | Risk: <strong>${act.risk}</strong>
            </div>
          </div>
        `;
      });
      actionsList.innerHTML = html || '<div style="color: var(--text-secondary);">No specific actions approved.</div>';
    }

    // 3. Data Change Summary
    const changeGrid = document.getElementById('data-change-metrics');
    if (changeGrid && report.data_change_summary) {
      const changes = report.data_change_summary;
      let catColor = 'var(--text-primary)';
      if (changes.change_category === 'CORRECTED') catColor = 'var(--success)';
      else if (changes.change_category === 'PARTIALLY_CORRECTED') catColor = 'var(--warning)';
      else if (changes.change_category === 'RECORD_REMOVED') catColor = 'var(--info)';
      else if (changes.change_category === 'UNRESOLVED') catColor = 'var(--critical)';

      changeGrid.innerHTML = `
        <div class="metric-card">
          <div class="metric-label">Category</div>
          <div class="metric-value" style="font-size: 1.2rem; color: ${catColor};">${changes.change_category}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Changed Cells</div>
          <div class="metric-value">${changes.changed_cell_values}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Removed Duplicates</div>
          <div class="metric-value">${changes.removed_duplicate_rows}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">New Missing Values</div>
          <div class="metric-value">${changes.newly_introduced_missing_values}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Unresolved Values</div>
          <div class="metric-value">${changes.unresolved_values}</div>
        </div>
      `;
    }
  },

  /**
   * Render Root Cause Investigation Section.
   * @param {Object} rc
   */
  renderRootCause(rc) {
    const rcTitle = document.getElementById('root-cause-primary');
    const rcConfidence = document.getElementById('root-cause-confidence');
    const rcScore = document.getElementById('root-cause-score');
    const rcReasoning = document.getElementById('root-cause-reasoning');
    const rcBar = document.getElementById('confidence-bar-fill');

    const primary = rc.primary_cause || 'No dominant root cause identified';
    const conf = (rc.confidence || 'NONE').toUpperCase();
    const score = rc.score !== undefined ? rc.score : 0;
    const reasoning = rc.narrative || rc.reasoning || 'Diagnostic agents found no critical upstream pipeline failure.';

    if (rcTitle) rcTitle.textContent = primary;
    if (rcConfidence) {
      rcConfidence.textContent = conf;
      if (primary.toLowerCase().includes('no data quality issues') || conf === 'NONE') {
        rcConfidence.className = 'badge badge-success';
      } else {
        rcConfidence.className = `badge ${conf === 'HIGH' ? 'badge-critical' : conf === 'MEDIUM' ? 'badge-warning' : 'badge-neutral'}`;
      }
    }
    if (rcScore) rcScore.textContent = `Score: ${score}`;
    if (rcReasoning) rcReasoning.textContent = reasoning;

    if (rcBar) {
      let pct = '0%';
      if (primary.toLowerCase().includes('no data quality issues')) pct = '100%';
      else if (conf === 'HIGH') pct = '95%';
      else if (conf === 'MEDIUM') pct = '60%';
      else if (conf === 'LOW') pct = '30%';
      rcBar.style.width = pct;
    }
  },

  /**
   * Render Multi-Agent Evidence Section.
   * @param {Object} report
   */
  renderEvidence(report) {
    const profEvidence = document.getElementById('evidence-profiling');
    const anomEvidence = document.getElementById('evidence-anomaly');
    const schEvidence = document.getElementById('evidence-schema');
    const rcEvidence = document.getElementById('evidence-root-cause');

    if (profEvidence) {
      profEvidence.textContent = report.detected_issues?.profiling_summary || 'No profiling anomalies.';
    }
    if (anomEvidence) {
      anomEvidence.textContent = report.detected_issues?.anomaly_summary || 'No statistical anomalies detected.';
    }
    if (schEvidence) {
      schEvidence.textContent = report.detected_issues?.schema_summary || 'Schema definition valid.';
    }
    if (rcEvidence) {
      rcEvidence.textContent = report.root_cause?.reasoning || report.root_cause?.narrative || 'No root cause evidence.';
    }
  },

  /**
   * Render Correction Recommendations.
   * @param {Object} corr
   */
  renderRecommendations(corr) {
    const container = document.getElementById('recommendations-container');
    if (!container) return;

    const actions = corr.actions || [];
    if (actions.length === 0) {
      container.innerHTML = `
        <div style="background: var(--surface); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border); color: var(--text-secondary); text-align: center;">
          ✓ No corrective remediation actions required. Dataset meets quality standards.
        </div>
      `;
      return;
    }

    const html = actions.map((act, idx) => {
      const priority = act.priority || 'HIGH';
      const badgeClass = priority === 'CRITICAL' ? 'badge-critical' : priority === 'HIGH' ? 'badge-warning' : 'badge-info';
      const recId = act.recommendation_id || `REC-00${idx + 1}`;
      const desc = act.description || act.action_type || 'Remediation Step';
      const targetCol = act.target_column ? `Column: ${act.target_column}` : 'Multi-column';
      const risk = act.risk_level || 'LOW';

      return `
        <div class="recommendation-card">
          <div>
            <div class="rec-header">
              <span class="rec-id">${recId}</span>
              <span class="badge ${badgeClass}">${priority}</span>
            </div>
            <div class="rec-desc" style="margin: 0.5rem 0;">${desc}</div>
            <div class="rec-meta">
              <span class="badge badge-neutral">${targetCol}</span>
              <span class="badge badge-neutral">Risk: ${risk}</span>
              <span class="badge badge-warning">Approval Required</span>
            </div>
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted); border-top: 1px solid var(--border-light); padding-top: 0.5rem;">
            ${act.expected_outcome || 'Restores dataset integrity adhering to strict validation criteria.'}
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = html;
  },

  /**
   * Render Interactive Human Approval Gate Banner.
   * @param {Object} report
   */
  renderApprovalGate(report) {
    const banner = document.getElementById('approval-gate-banner');
    if (!banner) return;

    const status = report.approval_status || 'PENDING';
    const wfStatus = report.workflow_status;

    if (wfStatus === 'NO_CORRECTION_REQUIRED') {
      banner.style.display = 'none';
      return;
    }

    banner.style.display = 'block';

    if (status === 'PENDING') {
      banner.innerHTML = `
        <div class="approval-banner">
          <div class="approval-title">⚠️ HUMAN APPROVAL REQUIRED</div>
          <div class="approval-subtitle">
            Correction recommendations have been generated. Human approval is required before creating the corrected dataset.
          </div>
          <div class="approval-actions">
            <button class="btn btn-success" id="btn-approve-workflow">
              ✓ Approve & Generate Corrected CSV
            </button>
            <button class="btn btn-danger" id="btn-reject-workflow">
              ✕ Reject Remediation Plan
            </button>
          </div>
        </div>
      `;

      // Wire interactive approval buttons
      const btnApprove = document.getElementById('btn-approve-workflow');
      const btnReject = document.getElementById('btn-reject-workflow');

      if (btnApprove) {
        btnApprove.addEventListener('click', () => {
          if (window.App) window.App.executeApprovedWorkflow();
        });
      }
      if (btnReject) {
        btnReject.addEventListener('click', () => {
          if (window.App) window.App.executeRejectedWorkflow();
        });
      }
    } else if (status === 'APPROVED') {
      banner.innerHTML = `
        <div style="background: var(--success-bg); border: 1px solid var(--success-border); border-radius: 12px; padding: 1.25rem; text-align: center; color: var(--success); font-weight: 600;">
          ✓ Human Operator APPROVED remediation plan. Corrected dataset copy generated and post-correction validation executed.
        </div>
      `;
    } else if (status === 'REJECTED') {
      banner.innerHTML = `
        <div style="background: var(--critical-bg); border: 1px solid var(--critical-border); border-radius: 12px; padding: 1.25rem; text-align: center; color: var(--critical); font-weight: 600;">
          ✕ Human Operator REJECTED remediation plan. Workflow halted safely without creating a corrected dataset.
        </div>
      `;
    }
  },

  /**
   * Render Generic Corrected Dataset Result Card.
   * @param {Object} report
   */
  renderCorrectionResult(report) {
    const section = document.getElementById('correction-result-section');
    if (!section) return;

    const approval = (report.approval_status || 'PENDING').toUpperCase();
    const wfStatus = report.workflow_status;
    const corrDs = report.corrected_dataset || {};
    const val = report.validation_status || {};
    const valVerdict = (val.verdict || val.status || 'NOT_VALIDATED').toUpperCase();

    if (wfStatus === 'NO_CORRECTION_REQUIRED' || approval === 'REJECTED' || approval === 'PENDING') {
      section.style.display = 'none';
      return;
    }

    if (approval === 'APPROVED' && (corrDs.corrected_filename || report.corrected_dataset_path || val.corrected_dataset_path)) {
      section.style.display = 'block';

      const origName = corrDs.original_filename || report.dataset?.name || report.dataset?.path?.split('/').pop() || 'dataset.csv';
      const origPath = corrDs.original_dataset_path || report.dataset?.path || 'N/A';
      const corrName = corrDs.corrected_filename || (report.corrected_dataset_path || val.corrected_dataset_path || '').split('/').pop() || `${origName.replace('.csv', '')}_corrected.csv`;
      const corrPath = corrDs.corrected_dataset_path || report.corrected_dataset_path || val.corrected_dataset_path || 'N/A';

      const appliedCount = corrDs.corrections_applied !== undefined ? corrDs.corrections_applied : (val.passed_checks || 0);
      const remainingCount = corrDs.issues_remaining !== undefined ? corrDs.issues_remaining : (report.unresolved_issues?.length || 0);

      const elOrigName = document.getElementById('corr-orig-filename');
      const elOrigPath = document.getElementById('corr-orig-path');
      const elNewName = document.getElementById('corr-new-filename');
      const elNewPath = document.getElementById('corr-new-path');
      const elApplied = document.getElementById('corr-applied-count');
      const elRemaining = document.getElementById('corr-remaining-count');
      const elValBadge = document.getElementById('corr-val-badge');
      const btnDownload = document.getElementById('btn-download-corrected-csv');

      if (elOrigName) elOrigName.textContent = origName;
      if (elOrigPath) elOrigPath.textContent = origPath;
      if (elNewName) elNewName.textContent = corrName;
      if (elNewPath) elNewPath.textContent = corrPath;
      if (elApplied) elApplied.textContent = appliedCount.toLocaleString();
      if (elRemaining) elRemaining.textContent = remainingCount.toLocaleString();

      if (elValBadge) {
        if (valVerdict === 'PASSED') {
          elValBadge.innerHTML = '<span class="badge badge-success" style="font-size: 0.85rem;">✓ POST-CORRECTION VALIDATION PASSED</span>';
        } else if (valVerdict === 'PARTIAL') {
          elValBadge.innerHTML = '<span class="badge badge-warning" style="font-size: 0.85rem;">⚠️ VALIDATION PARTIAL (Unresolved Issues)</span>';
        } else if (valVerdict === 'FAILED') {
          elValBadge.innerHTML = '<span class="badge badge-critical" style="font-size: 0.85rem;">✕ VALIDATION FAILED</span>';
        } else {
          elValBadge.innerHTML = `<span class="badge badge-neutral" style="font-size: 0.85rem;">${valVerdict}</span>`;
        }
      }

      if (btnDownload) {
        btnDownload.onclick = () => {
          if (window.API) {
            window.API.downloadCorrectedDataset(corrName, corrPath);
          }
        };
      }
    } else {
      section.style.display = 'none';
    }
  },

  /**
   * Render Validation Section & Before/After Comparison.
   * @param {Object} val
   * @param {Array} unresolved
   */
  renderValidation(val, unresolved) {
    const valSection = document.getElementById('validation');
    const verdictBanner = document.getElementById('validation-verdict-banner');
    const checksList = document.getElementById('validation-checks-list');
    const comparisonContainer = document.getElementById('before-after-container');

    if (!valSection) return;

    const verdict = (val?.verdict || val?.status || 'NOT_VALIDATED').toUpperCase();

    if (verdict === 'NOT_APPLICABLE') {
      if (verdictBanner) {
        verdictBanner.className = 'verdict-banner';
        verdictBanner.style.background = 'var(--surface)';
        verdictBanner.style.border = '1px solid var(--border)';
        verdictBanner.innerHTML = `
          <div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--success);">✓ VALIDATION NOT REQUIRED</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
              Dataset is verified clean with zero defects; no corrective action or post-correction validation needed.
            </div>
          </div>
          <span class="badge badge-success">CLEAN DATASET</span>
        `;
      }
      if (checksList) checksList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">All dataset quality checks satisfied on baseline profiling.</p>';
      if (comparisonContainer) comparisonContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">No remediation was required.</p>';
      return;
    }

    if (verdict === 'NOT_VALIDATED' || val?.status === 'not_executed') {
      if (verdictBanner) {
        verdictBanner.className = 'verdict-banner';
        verdictBanner.style.background = 'var(--surface)';
        verdictBanner.style.border = '1px solid var(--border)';
        verdictBanner.innerHTML = `
          <div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-muted);">— VALIDATION NOT EXECUTED</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">
              Post-correction validation is pending human approval or post-correction fixture.
            </div>
          </div>
          <span class="badge badge-neutral">PENDING APPROVAL</span>
        `;
      }
      if (checksList) checksList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">Validation checks will execute once the correction plan is approved.</p>';
      if (comparisonContainer) comparisonContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">Validation metrics pending execution.</p>';
      return;
    }

    // Verdict Banner (PASSED / PARTIAL / FAILED)
    if (verdictBanner) {
      let vClass = 'verdict-failed';
      let vIcon = '✕';
      let badgeClass = 'badge-critical';
      if (verdict === 'PASSED') {
        vClass = 'verdict-passed';
        vIcon = '✓';
        badgeClass = 'badge-success';
      } else if (verdict === 'PARTIAL') {
        vClass = 'verdict-partial';
        vIcon = '⚠️';
        badgeClass = 'badge-warning';
      }

      const passed = val.passed_checks || 0;
      const failed = val.failed_checks || 0;
      const partial = val.partial_checks || 0;

      verdictBanner.className = `verdict-banner ${vClass}`;
      verdictBanner.style.background = '';
      verdictBanner.style.border = '';
      verdictBanner.innerHTML = `
        <div>
          <div style="font-size: 1.25rem; font-weight: 800;">${vIcon} VALIDATION ${verdict}</div>
          <div style="font-size: 0.85rem; margin-top: 0.25rem; opacity: 0.9;">
            ${passed} Passed &nbsp;•&nbsp; ${failed} Failed &nbsp;•&nbsp; ${partial} Partial
          </div>
        </div>
        <span class="badge ${badgeClass}">${verdict}</span>
      `;
    }

    // Checks List (Dynamic from API)
    if (checksList) {
      const checks = val.checks || [];
      if (checks.length === 0) {
        checksList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">No individual check records returned.</p>';
      } else {
        checksList.innerHTML = checks.map(chk => {
          const status = (chk.status || 'PASS').toUpperCase();
          let bClass = 'badge-success';
          let icon = '✓';
          let iconColor = 'var(--success)';

          if (status === 'FAIL') {
            bClass = 'badge-critical';
            icon = '✕';
            iconColor = 'var(--critical)';
          } else if (status === 'PARTIAL') {
            bClass = 'badge-warning';
            icon = '⚠️';
            iconColor = 'var(--warning)';
          }

          const name = chk.check_name || chk.check_id || 'Validation Check';
          const cat = chk.category ? `(${chk.category})` : '';
          const evidence = chk.evidence || chk.actual || 'Check executed.';

          return `
            <div class="check-item">
              <div class="check-left">
                <span style="color: ${iconColor}; font-size: 1.1rem; font-weight: 800;">${icon}</span>
                <div>
                  <div class="check-name">${name} <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: normal;">${cat}</span></div>
                  <div class="check-evidence">${evidence}</div>
                </div>
              </div>
              <span class="badge ${bClass}">${status}</span>
            </div>
          `;
        }).join('');
      }
    }

    // Before / After Visualizer (Dynamic from API)
    if (comparisonContainer && window.Charts) {
      window.Charts.renderBeforeAfter(comparisonContainer, val.before_after_comparison || {});
    }
  },

  /**
   * Render Professional Human-Readable Final Investigation Report.
   * @param {Object} report
   */
  renderFinalReport(report) {
    const humanContainer = document.getElementById('human-report-container');
    const reportText = document.getElementById('final-report-text');
    const traceJson = document.getElementById('execution-trace-json');

    // Update Technical Details Raw Views
    if (reportText) {
      reportText.textContent = JSON.stringify(report, null, 2);
    }
    if (traceJson) {
      traceJson.textContent = JSON.stringify(report.execution_trace || [], null, 2);
    }

    if (!humanContainer || !report) return;

    const dataset = report.dataset || {};
    const dsName = dataset.name || dataset.path?.split('/').pop() || 'Sales Dataset';
    const dsPath = dataset.path || 'N/A';
    const wfStatus = report.workflow_status || 'UNKNOWN';
    const approval = (report.approval_status || 'PENDING').toUpperCase();
    const val = report.validation_status || {};
    const valVerdict = (val.verdict || val.status || 'NOT_VALIDATED').toUpperCase();
    const issues = report.detected_issues || {};
    const missing = issues.missing_values_count || 0;
    const dups = issues.duplicate_rows_count || 0;
    const anoms = issues.anomalous_dates_count || 0;
    const schemas = issues.schema_violations_count || 0;
    const totalDefects = missing + dups + schemas;

    // Determine Overall Risk & Badges
    let overallStatusLabel = 'ISSUES DETECTED';
    let overallStatusClass = 'badge-critical';
    let riskLevel = 'HIGH';
    let riskClass = 'badge-critical';

    if (wfStatus === 'NO_CORRECTION_REQUIRED' || totalDefects === 0) {
      overallStatusLabel = 'CLEAN';
      overallStatusClass = 'badge-success';
      riskLevel = 'NONE';
      riskClass = 'badge-success';
    } else if (wfStatus === 'COMPLETED' && valVerdict === 'PASSED') {
      overallStatusLabel = 'COMPLETED';
      overallStatusClass = 'badge-success';
      riskLevel = 'RESOLVED';
      riskClass = 'badge-success';
    } else if (wfStatus === 'PENDING_APPROVAL') {
      overallStatusLabel = 'PENDING APPROVAL';
      overallStatusClass = 'badge-warning';
      riskLevel = 'HIGH (Action Required)';
      riskClass = 'badge-critical';
    } else if (wfStatus === 'REJECTED') {
      overallStatusLabel = 'REJECTED';
      overallStatusClass = 'badge-critical';
      riskLevel = 'UNRESOLVED';
      riskClass = 'badge-warning';
    } else if (wfStatus === 'FAILED') {
      overallStatusLabel = 'FAILED';
      overallStatusClass = 'badge-critical';
      riskLevel = 'CRITICAL';
      riskClass = 'badge-critical';
    }

    // Approval Badge
    let approvalBadgeClass = 'badge-warning';
    if (approval === 'APPROVED') approvalBadgeClass = 'badge-success';
    else if (approval === 'REJECTED') approvalBadgeClass = 'badge-critical';
    else if (wfStatus === 'NO_CORRECTION_REQUIRED') approvalBadgeClass = 'badge-neutral';

    // Validation Badge
    let valBadgeClass = 'badge-neutral';
    let valLabel = valVerdict;
    if (valVerdict === 'PASSED') { valBadgeClass = 'badge-success'; valLabel = 'VALIDATION PASSED'; }
    else if (valVerdict === 'PARTIAL') { valBadgeClass = 'badge-warning'; valLabel = 'VALIDATION PARTIAL'; }
    else if (valVerdict === 'FAILED') { valBadgeClass = 'badge-critical'; valLabel = 'VALIDATION FAILED'; }
    else if (valVerdict === 'NOT_APPLICABLE') { valBadgeClass = 'badge-success'; valLabel = 'NOT REQUIRED (CLEAN)'; }
    else if (valVerdict === 'NOT_VALIDATED') { valBadgeClass = 'badge-neutral'; valLabel = 'PENDING APPROVAL'; }

    // Root cause details
    const rc = report.root_cause || {};
    const rcPrimary = rc.primary_cause || 'No dominant root cause identified';
    const rcConfidence = (rc.confidence || 'NONE').toUpperCase();
    const rcScore = rc.score !== undefined ? rc.score : 0;
    const rcReasoning = rc.narrative || rc.reasoning || 'Diagnostic agents found no critical upstream pipeline failure.';

    // Remediation actions
    const actions = report.correction_recommendation?.actions || [];

    // Final Outcome description
    let outcomeText = '';
    let outcomeIcon = '✓';
    let outcomeClass = 'outcome-success';

    if (wfStatus === 'NO_CORRECTION_REQUIRED' || totalDefects === 0) {
      outcomeText = 'The dataset was analyzed successfully and no corrective action is required.';
      outcomeIcon = '✓';
      outcomeClass = 'outcome-success';
    } else if (wfStatus === 'PENDING_APPROVAL') {
      outcomeText = 'Data-quality issues were detected and a remediation plan has been generated. Human approval is required before validation.';
      outcomeIcon = '⚠️';
      outcomeClass = 'outcome-warning';
    } else if (wfStatus === 'COMPLETED' && valVerdict === 'PASSED') {
      outcomeText = 'The investigation and remediation workflow completed successfully. Post-correction validation passed all required checks.';
      outcomeIcon = '✓';
      outcomeClass = 'outcome-success';
    } else if (wfStatus === 'REJECTED') {
      outcomeText = 'The remediation plan was rejected by the operator. Investigation halted safely without data modification.';
      outcomeIcon = '✕';
      outcomeClass = 'outcome-critical';
    } else if (wfStatus === 'FAILED' || valVerdict === 'FAILED') {
      outcomeText = 'The investigation or validation identified unresolved data-quality issues.';
      outcomeIcon = '✕';
      outcomeClass = 'outcome-critical';
    } else {
      outcomeText = report.final_recommendation || 'Investigation workflow completed.';
      outcomeIcon = 'ℹ️';
      outcomeClass = 'outcome-neutral';
    }

    // Build the complete human-readable HTML
    let html = `
      <!-- 1. INVESTIGATION SUMMARY -->
      <div class="report-section-title" style="margin-top: 0.5rem;">
        <span>📋</span> 1. Investigation Summary
      </div>
      <div class="report-summary-grid">
        <div class="report-summary-card">
          <div class="report-summary-label">Target Dataset</div>
          <div class="report-summary-value" style="font-size: 0.85rem; font-family: var(--font-mono);">${dsName}</div>
          <div class="report-summary-path" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;"><code>${dsPath}</code></div>
        </div>

        <div class="report-summary-card">
          <div class="report-summary-label">Investigation Status</div>
          <div class="report-summary-value">
            <span class="badge ${overallStatusClass}">${overallStatusLabel}</span>
          </div>
        </div>

        <div class="report-summary-card">
          <div class="report-summary-label">Overall Severity / Risk</div>
          <div class="report-summary-value">
            <span class="badge ${riskClass}">${riskLevel}</span>
          </div>
        </div>

        <div class="report-summary-card">
          <div class="report-summary-label">Approval Status</div>
          <div class="report-summary-value">
            <span class="badge ${approvalBadgeClass}">${wfStatus === 'NO_CORRECTION_REQUIRED' ? 'NOT REQUIRED' : approval}</span>
          </div>
        </div>

        <div class="report-summary-card">
          <div class="report-summary-label">Validation Status</div>
          <div class="report-summary-value">
            <span class="badge ${valBadgeClass}">${valLabel}</span>
          </div>
        </div>
      </div>

      <!-- 2. DATA QUALITY FINDINGS -->
      <div class="report-section-title">
        <span>🔍</span> 2. Data Quality Findings
      </div>
      <div class="report-findings-grid">
        <div class="report-finding-card">
          <div class="report-finding-header">
            <span style="font-weight: 600; font-size: 0.85rem; color: var(--text-primary);">Missing Values</span>
            <span class="badge ${missing > 0 ? 'badge-warning' : 'badge-success'}">${missing > 0 ? 'DEFECTS FOUND' : 'CLEAN'}</span>
          </div>
          <div class="report-finding-count" style="color: ${missing > 0 ? 'var(--warning)' : 'var(--success)'};">${missing.toLocaleString()}</div>
          <div class="report-finding-desc">${missing > 0 ? `${missing} null or empty cells detected across non-nullable fields.` : '100% completeness across all records; no nulls.'}</div>
        </div>

        <div class="report-finding-card">
          <div class="report-finding-header">
            <span style="font-weight: 600; font-size: 0.85rem; color: var(--text-primary);">Duplicate Records</span>
            <span class="badge ${dups > 0 ? 'badge-critical' : 'badge-success'}">${dups > 0 ? 'DUPLICATES FOUND' : 'CLEAN'}</span>
          </div>
          <div class="report-finding-count" style="color: ${dups > 0 ? 'var(--critical)' : 'var(--success)'};">${dups.toLocaleString()}</div>
          <div class="report-finding-desc">${dups > 0 ? `${dups} duplicate rows detected violating primary key uniqueness.` : 'All transaction IDs and record rows are distinct.'}</div>
        </div>

        <div class="report-finding-card">
          <div class="report-finding-header">
            <span style="font-weight: 600; font-size: 0.85rem; color: var(--text-primary);">Statistical Anomalies</span>
            <span class="badge ${anoms > 0 ? 'badge-info' : 'badge-success'}">${anoms > 0 ? 'ANOMALIES' : 'NORMAL'}</span>
          </div>
          <div class="report-finding-count" style="color: ${anoms > 0 ? 'var(--info)' : 'var(--success)'};">${anoms.toLocaleString()}</div>
          <div class="report-finding-desc">${anoms > 0 ? `${anoms} anomalous date/volume spikes identified by statistical diagnostics.` : 'No temporal or volume spikes detected.'}</div>
        </div>

        <div class="report-finding-card">
          <div class="report-finding-header">
            <span style="font-weight: 600; font-size: 0.85rem; color: var(--text-primary);">Schema Violations</span>
            <span class="badge ${schemas > 0 ? 'badge-critical' : 'badge-success'}">${schemas > 0 ? 'VIOLATIONS' : 'CONFORMS'}</span>
          </div>
          <div class="report-finding-count" style="color: ${schemas > 0 ? 'var(--critical)' : 'var(--success)'};">${schemas.toLocaleString()}</div>
          <div class="report-finding-desc">${schemas > 0 ? `${schemas} schema, data type, or categorical domain constraint failures.` : 'Strict schema conformance across all columns.'}</div>
        </div>
      </div>

      <!-- 3. ROOT CAUSE ANALYSIS -->
      <div class="report-section-title">
        <span>🧠</span> 3. Root Cause Analysis
      </div>
      <div class="report-rc-card">
        <div style="margin-bottom: 0.6rem;">
          <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 700; margin-bottom: 0.2rem;">Primary Root Cause:</div>
          <div style="font-weight: 700; font-size: 1.05rem; color: var(--text-primary);">${rcPrimary}</div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap;">
          <div style="font-size: 0.85rem; color: var(--text-secondary);">
            <strong>Confidence:</strong> <span class="badge ${rcPrimary.toLowerCase().includes('no data quality issues') ? 'badge-success' : rcConfidence === 'HIGH' ? 'badge-critical' : 'badge-warning'}">${rcConfidence}</span>
          </div>
          <div style="font-size: 0.85rem; color: var(--text-secondary);">
            <strong>Score:</strong> <span style="font-family: var(--font-mono); color: var(--accent-light); font-weight: 600;">${rcScore}</span>
          </div>
        </div>
        <div>
          <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 700; margin-bottom: 0.2rem;">Explanation:</div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">${rcReasoning}</div>
        </div>
      </div>

      <!-- 4. RECOMMENDED REMEDIATION -->
      <div class="report-section-title">
        <span>🛠</span> 4. Recommended Remediation
      </div>
    `;

    if (actions.length === 0) {
      html += `
        <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1rem; color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1.25rem;">
          ✓ No corrective remediation required. The dataset adheres to all quality standards.
        </div>
      `;
    } else {
      html += `<div class="report-recs-list">`;
      actions.forEach((act, idx) => {
        const priority = act.priority || 'HIGH';
        const pBadge = priority === 'CRITICAL' ? 'badge-critical' : priority === 'HIGH' ? 'badge-warning' : 'badge-info';
        const title = act.description || act.action_type || `Remediation Action`;
        const recId = act.recommendation_id || `REC-00${idx + 1}`;
        const target = act.target_column ? act.target_column : 'Multi-column / Dataset';
        const risk = act.risk_level || 'LOW';
        const riskBadge = risk === 'HIGH' ? 'badge-critical' : risk === 'MEDIUM' ? 'badge-warning' : 'badge-success';

        html += `
          <div class="report-rec-item">
            <div class="report-rec-header">
              <div class="report-rec-title-row">
                <span class="report-rec-number">${idx + 1}.</span>
                <span class="report-rec-title">${title}</span>
              </div>
              <span class="badge ${pBadge}">${priority}</span>
            </div>
            <div class="report-rec-meta">
              <div class="report-rec-field"><strong>Priority:</strong> ${priority}</div>
              <div class="report-rec-field"><strong>Risk:</strong> ${risk}</div>
              <div class="report-rec-field"><strong>Target:</strong> <code>${target}</code></div>
              <div class="report-rec-field"><strong>Action ID:</strong> <code>${recId}</code></div>
              <div class="report-rec-field"><strong>Approval Required:</strong> Yes</div>
            </div>
            <div class="report-rec-desc">
              <strong>Description:</strong> ${act.action_type || 'Remediation'} targeting ${target} (${priority} priority, ${risk} risk). Automated fix generated for human review.
            </div>
          </div>
        `;
      });
      html += `</div>`;
    }

    // 5. HUMAN APPROVAL STATUS
    let approvalDesc = '';
    if (wfStatus === 'NO_CORRECTION_REQUIRED') {
      approvalDesc = 'Dataset is clean; automated human approval gate was bypassed.';
    } else if (approval === 'PENDING') {
      approvalDesc = 'Data-quality issues were detected and a remediation plan has been generated. Human approval is required before validation.';
    } else if (approval === 'APPROVED') {
      approvalDesc = 'Remediation plan authorized by human operator. Validation executed against the post-correction test fixture.';
    } else if (approval === 'REJECTED') {
      approvalDesc = 'Remediation plan rejected by operator. Workflow halted safely without modifying data.';
    }

    html += `
      <!-- 5. HUMAN APPROVAL STATUS -->
      <div class="report-section-title">
        <span>👤</span> 5. Human Approval Status
      </div>
      <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1.15rem; margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
          <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary);">
            Approval Status: <span class="badge ${approvalBadgeClass}">${wfStatus === 'NO_CORRECTION_REQUIRED' ? 'NOT REQUIRED' : approval}</span>
          </div>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.4; margin-bottom: ${approval === 'PENDING' && wfStatus === 'PENDING_APPROVAL' ? '1rem' : '0'};">
          ${approvalDesc}
        </div>
    `;

    // If PENDING, embed approval action buttons right inside the report
    if (approval === 'PENDING' && wfStatus === 'PENDING_APPROVAL') {
      html += `
        <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border-light);">
          <button class="btn btn-success" id="btn-report-approve" style="padding: 0.45rem 1rem; font-size: 0.85rem;">
            <span>✓</span> Approve & Run Validation
          </button>
          <button class="btn btn-critical" id="btn-report-reject" style="padding: 0.45rem 1rem; font-size: 0.85rem;">
            <span>✕</span> Reject & Halt Workflow
          </button>
        </div>
      `;
    }

    html += `
      </div>

      <!-- 6. POST-CORRECTION VALIDATION -->
      <div class="report-section-title">
        <span>✓</span> 6. Post-Correction Validation
      </div>
    `;

    if (valVerdict === 'NOT_APPLICABLE') {
      html += `
        <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1.15rem; margin-bottom: 1.25rem; color: var(--text-secondary); font-size: 0.85rem;">
          ✓ Validation not required. Dataset is verified clean with zero defects.
        </div>
      `;
    } else if (valVerdict === 'NOT_VALIDATED' || val.status === 'not_executed') {
      html += `
        <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1.15rem; margin-bottom: 1.25rem; color: var(--text-secondary); font-size: 0.85rem; line-height: 1.4;">
          ⚠️ <strong>Validation has not been executed yet. Human approval is required before post-correction validation.</strong>
        </div>
      `;
    } else {
      const passed = val.passed_checks || 0;
      const failed = val.failed_checks || 0;
      const partial = val.partial_checks || 0;
      const checks = val.checks || [];

      html += `
        <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1.25rem; margin-bottom: 1.25rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
            <div>
              <div style="font-weight: 700; font-size: 1rem; color: ${valVerdict === 'PASSED' ? 'var(--success)' : 'var(--critical)'};">
                ${valVerdict === 'PASSED' ? '✓' : '✕'} Validation Verdict: ${valVerdict}
              </div>
              <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.2rem;">
                Passed: <strong>${passed}</strong> &nbsp;•&nbsp; Failed: <strong>${failed}</strong> &nbsp;•&nbsp; Partial: <strong>${partial}</strong>
              </div>
            </div>
            <span class="badge ${valVerdict === 'PASSED' ? 'badge-success' : 'badge-critical'}">${valVerdict}</span>
          </div>

          <div style="margin-top: 0.75rem;">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;">Individual Validation Checks:</div>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
              ${checks.map(c => `
                <div style="font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: var(--surface); border: 1px solid var(--border-light); border-radius: 6px;">
                  <div>
                    <span style="color: var(--text-primary); font-weight: 600;">${c.check_name || c.check_id}</span>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem;">${c.evidence || c.actual || 'Check executed.'}</div>
                  </div>
                  <span class="badge ${c.status === 'PASS' ? 'badge-success' : 'badge-critical'}">${c.status}</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      `;
    }

    // 7. FINAL OUTCOME / CONCLUSION
    html += `
      <!-- 7. FINAL CONCLUSION -->
      <div class="report-section-title">
        <span>🎯</span> 7. Final Conclusion
      </div>
      <div class="report-outcome-banner ${outcomeClass}">
        <div style="font-size: 1.5rem; font-weight: 800;">${outcomeIcon}</div>
        <div>
          <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.25rem;">Conclusion</div>
          <div style="font-size: 0.85rem; line-height: 1.4;">${outcomeText}</div>
        </div>
      </div>
    `;

    humanContainer.innerHTML = html;

    // Wire up any interactive report approval buttons
    const btnReportApprove = document.getElementById('btn-report-approve');
    const btnReportReject = document.getElementById('btn-report-reject');
    if (btnReportApprove) {
      btnReportApprove.addEventListener('click', () => {
        if (window.App) window.App.executeApprovedWorkflow();
      });
    }
    if (btnReportReject) {
      btnReportReject.addEventListener('click', () => {
        if (window.App) window.App.executeRejectedWorkflow();
      });
    }
  },

  /**
   * Generate Clean Human-Readable Text Version of Investigation Report for Clipboard & File Export.
   * @param {Object} report
   * @returns {string}
   */
  generateHumanReadableTextReport(report) {
    if (!report) return '';

    const dataset = report.dataset || {};
    const dsName = dataset.name || dataset.path?.split('/').pop() || 'Sales Dataset';
    const dsPath = dataset.path || 'N/A';
    const wfStatus = report.workflow_status || 'UNKNOWN';
    const approval = (report.approval_status || 'PENDING').toUpperCase();
    const val = report.validation_status || {};
    const valVerdict = (val.verdict || val.status || 'NOT_VALIDATED').toUpperCase();
    const issues = report.detected_issues || {};
    const missing = issues.missing_values_count || 0;
    const dups = issues.duplicate_rows_count || 0;
    const anoms = issues.anomalous_dates_count || 0;
    const schemas = issues.schema_violations_count || 0;

    const rc = report.root_cause || {};
    const rcPrimary = rc.primary_cause || 'No dominant root cause identified';
    const rcConfidence = (rc.confidence || 'NONE').toUpperCase();
    const rcScore = rc.score !== undefined ? rc.score : 0;
    const rcReasoning = rc.narrative || rc.reasoning || 'Diagnostic agents found no critical upstream pipeline failure.';

    const actions = report.correction_recommendation?.actions || [];
    const checks = val.checks || [];

    const lines = [];
    lines.push('================================================================================');
    lines.push('AUTONOMOUS DATA QUALITY INVESTIGATION REPORT');
    lines.push('================================================================================');
    lines.push('');
    lines.push('1. INVESTIGATION SUMMARY');
    lines.push('--------------------------------------------------------------------------------');
    lines.push(`Dataset Name:        ${dsName}`);
    lines.push(`Dataset Path:        ${dsPath}`);
    lines.push(`Investigation Status: ${wfStatus}`);
    lines.push(`Approval Status:     ${wfStatus === 'NO_CORRECTION_REQUIRED' ? 'NOT REQUIRED' : approval}`);
    lines.push(`Validation Status:   ${valVerdict}`);
    lines.push(`Overall Severity:    ${wfStatus === 'NO_CORRECTION_REQUIRED' ? 'NONE' : wfStatus === 'COMPLETED' ? 'RESOLVED' : 'HIGH'}`);
    lines.push('');
    lines.push('2. DATA QUALITY FINDINGS');
    lines.push('--------------------------------------------------------------------------------');
    lines.push(`• Missing Values:        ${missing.toLocaleString()} null/empty cells (${missing > 0 ? 'DEFECTS DETECTED' : 'CLEAN'})`);
    lines.push(`• Duplicate Records:     ${dups.toLocaleString()} duplicate rows/IDs (${dups > 0 ? 'DUPLICATES DETECTED' : 'CLEAN'})`);
    lines.push(`• Statistical Anomalies: ${anoms.toLocaleString()} temporal outliers (${anoms > 0 ? 'ANOMALIES DETECTED' : 'NORMAL'})`);
    lines.push(`• Schema Violations:     ${schemas.toLocaleString()} type/constraint errors (${schemas > 0 ? 'VIOLATIONS DETECTED' : 'CONFORMS'})`);
    lines.push('');
    lines.push('3. ROOT CAUSE ANALYSIS');
    lines.push('--------------------------------------------------------------------------------');
    lines.push(`Primary Root Cause: ${rcPrimary}`);
    lines.push(`Confidence:         ${rcConfidence} (Score: ${rcScore})`);
    lines.push(`Explanation:        ${rcReasoning}`);
    lines.push('');
    lines.push('4. RECOMMENDED REMEDIATION');
    lines.push('--------------------------------------------------------------------------------');
    if (actions.length === 0) {
      lines.push('No corrective remediation required. The dataset adheres to all quality standards.');
    } else {
      actions.forEach((act, idx) => {
        const priority = act.priority || 'HIGH';
        const risk = act.risk_level || 'LOW';
        const title = act.description || act.action_type || 'Remediation Step';
        const target = act.target_column ? ` (Target: ${act.target_column})` : '';
        lines.push(`${idx + 1}. ${title}${target}`);
        lines.push(`   Priority:    ${priority}`);
        lines.push(`   Risk:        ${risk}`);
        lines.push(`   Description: Automated fix generated for human review.`);
        lines.push('');
      });
    }
    lines.push('5. HUMAN APPROVAL');
    lines.push('--------------------------------------------------------------------------------');
    lines.push(`Approval Status: ${wfStatus === 'NO_CORRECTION_REQUIRED' ? 'NOT REQUIRED' : approval}`);
    if (wfStatus === 'NO_CORRECTION_REQUIRED') {
      lines.push('Dataset is clean; automated human approval gate was bypassed.');
    } else if (approval === 'PENDING') {
      lines.push('Data-quality issues were detected and a remediation plan has been generated. Human approval is required before validation.');
    } else if (approval === 'APPROVED') {
      lines.push('Remediation plan authorized by human operator. Validation executed against the post-correction test fixture.');
    } else if (approval === 'REJECTED') {
      lines.push('Remediation plan rejected by operator. Workflow halted safely without data modification.');
    }
    lines.push('');
    lines.push('6. POST-CORRECTION VALIDATION');
    lines.push('--------------------------------------------------------------------------------');
    if (valVerdict === 'NOT_APPLICABLE') {
      lines.push('Validation not required. Dataset is verified clean with zero defects.');
    } else if (valVerdict === 'NOT_VALIDATED' || val.status === 'not_executed') {
      lines.push('Validation has not been executed yet. Human approval is required before post-correction validation.');
    } else {
      lines.push(`Validation Status: ${valVerdict}`);
      lines.push(`Passed Checks:     ${val.passed_checks || 0}`);
      lines.push(`Failed Checks:     ${val.failed_checks || 0}`);
      lines.push(`Partial Checks:    ${val.partial_checks || 0}`);
      lines.push('Individual Checks:');
      checks.forEach(c => {
        lines.push(`  • [${c.status}] ${c.check_name || c.check_id}: ${c.evidence || c.actual || 'Executed'}`);
      });
    }
    lines.push('');
    lines.push('7. FINAL CONCLUSION');
    lines.push('--------------------------------------------------------------------------------');
    if (wfStatus === 'NO_CORRECTION_REQUIRED') {
      lines.push('The dataset was analyzed successfully and no corrective action is required.');
    } else if (wfStatus === 'PENDING_APPROVAL') {
      lines.push('Data-quality issues were detected and a remediation plan has been generated. Human approval is required before validation.');
    } else if (wfStatus === 'COMPLETED' && valVerdict === 'PASSED') {
      lines.push('The investigation and remediation workflow completed successfully. Post-correction validation passed all required checks.');
    } else if (wfStatus === 'REJECTED') {
      lines.push('The remediation plan was rejected by the operator. Investigation halted safely without data modification.');
    } else {
      lines.push(report.final_recommendation || 'The investigation or validation identified unresolved data-quality issues.');
    }
    lines.push('================================================================================');

    return lines.join('\n');
  },

  getWorkflowStatusBadge(status) {
    switch (status) {
      case 'COMPLETED':
      case 'NO_CORRECTION_REQUIRED':
        return 'badge-success';
      case 'PENDING_APPROVAL':
        return 'badge-warning';
      case 'REJECTED':
      case 'FAILED':
        return 'badge-critical';
      default:
        return 'badge-neutral';
    }
  },
};

window.UI = UI;
