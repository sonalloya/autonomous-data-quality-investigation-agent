/**
 * frontend/js/charts.js
 * ----------------------
 * Pure SVG & Canvas Charting Engine for Data Quality Visualizations.
 */

const Charts = {
  /**
   * Render SVG Issues Distribution Donut/Bar Chart.
   * @param {HTMLElement} container
   * @param {Object} detectedIssues
   */
  renderIssueDistribution(container, detectedIssues) {
    if (!container) return;

    const data = [
      { label: 'Missing Values', count: detectedIssues.missing_values_count || 0, color: '#f59e0b' },
      { label: 'Duplicates', count: detectedIssues.duplicate_rows_count || 0, color: '#ef4444' },
      { label: 'Anomalies', count: detectedIssues.anomalous_dates_count || 0, color: '#3b82f6' },
      { label: 'Schema Issues', count: detectedIssues.schema_violations_count || 0, color: '#8b5cf6' },
    ];

    const total = data.reduce((sum, d) => sum + d.count, 0);

    if (total === 0) {
      container.innerHTML = `
        <div style="text-align: center; padding: 2rem; color: var(--success);">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">✓</div>
          <p style="font-weight: 600;">Zero Data Quality Issues Detected</p>
          <p style="font-size: 0.8rem; color: var(--text-muted);">Dataset passed all quality thresholds cleanly.</p>
        </div>
      `;
      return;
    }

    let barsHtml = data.map(item => {
      const pct = total > 0 ? ((item.count / total) * 100).toFixed(1) : 0;
      return `
        <div style="margin-bottom: 0.85rem;">
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.3rem;">
            <span style="color: var(--text-primary); font-weight: 500;">${item.label}</span>
            <span style="color: var(--text-secondary); font-family: var(--font-mono);">${item.count.toLocaleString()} (${pct}%)</span>
          </div>
          <div style="height: 10px; background: var(--bg-secondary); border-radius: 9999px; overflow: hidden;">
            <div style="width: ${pct}%; height: 100%; background: ${item.color}; border-radius: 9999px; transition: width 0.6s ease;"></div>
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = `
      <div style="padding: 0.5rem 0;">
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 1rem;">
          <span style="font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Total Detected Defects</span>
          <span style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); font-family: var(--font-mono);">${total.toLocaleString()}</span>
        </div>
        ${barsHtml}
      </div>
    `;
  },

  /**
   * Render Before / After Comparison Visuals.
   * @param {HTMLElement} container
   * @param {Object} comparisonData
   */
  renderBeforeAfter(container, comparisonData) {
    if (!container || !comparisonData || Object.keys(comparisonData).length === 0) {
      container.innerHTML = `<p style="color: var(--text-muted); font-size: 0.85rem;">No before/after validation comparison available.</p>`;
      return;
    }

    const items = Object.entries(comparisonData).map(([key, val]) => {
      const before = val.before !== undefined ? val.before : 'N/A';
      const after = val.after !== undefined ? val.after : 'N/A';
      const improvement = val.improvement_pct !== undefined ? `${val.improvement_pct}% improvement` : null;

      let label = key.replace(/_/g, ' ').toUpperCase();
      let badgeClass = 'badge-success';
      if (after > 0 && typeof after === 'number') {
        badgeClass = 'badge-warning';
      }

      return `
        <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; font-size: 0.85rem; color: var(--text-primary);">${label}</span>
            ${improvement ? `<span class="badge ${badgeClass}">${improvement}</span>` : ''}
          </div>
          <div style="display: flex; align-items: center; gap: 1rem; font-family: var(--font-mono); font-size: 0.95rem;">
            <span style="color: var(--critical);">${typeof before === 'number' ? before.toLocaleString() : before}</span>
            <span style="color: var(--text-muted);">➔</span>
            <span style="color: var(--success); font-weight: 700;">${typeof after === 'number' ? after.toLocaleString() : after}</span>
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = items;
  },
};

window.Charts = Charts;
