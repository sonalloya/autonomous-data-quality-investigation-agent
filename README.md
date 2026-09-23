# Autonomous Data Quality Investigation Agent

> **Phase 11 — Professional Web Frontend (HTML5 + CSS3 + Vanilla JavaScript)**
>
> **286 Automated Tests Passing (100% Deterministic, Multi-Agent & Safe)**

---

## 1. Project Overview

The **Autonomous Data Quality Investigation Agent** is an enterprise-grade agentic AI system designed to automatically diagnose, investigate, recommend fixes for, and validate data quality defects in tabular datasets.

Unlike traditional passive data quality tools that only trigger alerts, this system:
1. **Profiles** datasets to discover nulls, cardinality, and duplicate transaction IDs.
2. **Detects** statistical outliers and anomalies using Z-scores and Interquartile Ranges (IQR).
3. **Analyzes** schemas for type mismatches, non-nullable field breaches, and date format errors.
4. **Investigates Root Causes** by correlating multi-agent empirical evidence into probabilistic hypotheses.
5. **Recommends Remediations** with prioritized action plans and risk ratings.
6. **Enforces Human-in-the-Loop Approval** (`APPROVED`, `PENDING`, `REJECTED`) before any downstream steps.
7. **Independently Validates** post-correction datasets against baseline metrics with deterministic checks.
8. **Compiles Structured Reports** and provides a validated FastAPI interface + Web UI Dashboard.

---

## 2. Architecture & Multi-Agent Flow

```text
Dataset (CSV - Read-Only)
  ↓
Profiling Agent (app/agents/profiling_agent.py)
  ↓
Anomaly Detection Agent (app/agents/anomaly_agent.py)
  ↓
Schema Analysis Agent (app/agents/schema_agent.py)
  ↓
Root Cause Agent (app/agents/root_cause_agent.py)
  ↓
Correction Recommendation Agent (app/agents/correction_agent.py)
  ↓
Human Approval Gate (APPROVED / PENDING / REJECTED)
  ↓
Validation Agent (app/agents/validation_agent.py) [Only if APPROVED + corrected fixture]
  ↓
Final Workflow Report (app/graph/report.py)
```

---

## 3. Web Frontend Dashboard (Phase 11)

The web dashboard is served directly from the FastAPI backend at `http://127.0.0.1:8000/`:

- **Enterprise Dark-Themed UI**: Modern, high-contrast dashboard with glassmorphic cards and intuitive visual hierarchy.
- **Interactive Workflow Timeline**: Dynamic stepper visualizing agent execution states (`COMPLETED`, `PENDING`, `RUNNING`, `SKIPPED`, `FAILED`).
- **Interactive Human-in-the-Loop Approval**: Review recommendations with one-click **"Approve & Validate"** or **"Reject"** action triggers.
- **Defect Distribution & Validation Charts**: Pure SVG/Canvas data visualizations with zero external framework dependencies.
- **Before/After Empirical Comparison**: Side-by-side metric tables tracking duplicates, nullability, type mismatches, and revenue recalculations.
- **Report Export**: Copy report JSON to clipboard or download formatted JSON reports with a single click.

---

## 4. Quick Start & Execution

### A. Run the Web Application & API Server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
# or
python main.py
```

- **Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger Interactive API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### B. Run Interactive End-to-End Demo (CLI)

Demonstrates Scenario A (Approved + Validated) and Scenario B (Pending Approval):

```bash
python examples/end_to_end_demo.py
```

To run a specific scenario:
```bash
python examples/end_to_end_demo.py --scenario a   # Approved + Validation PASSED
python examples/end_to_end_demo.py --scenario b   # Pending approval
```

---

### C. Command-Line Interface (CLI)

```bash
# Problematic dataset with default PENDING approval
python -m app.graph.workflow data/raw/sales_problematic.csv

# Problematic dataset with APPROVED status and corrected fixture for validation
python -m app.graph.workflow data/raw/sales_problematic.csv \
  --approval-status APPROVED \
  --corrected-dataset-path data/test/fully_corrected.csv \
  --verbose

# Clean dataset (skips correction & validation)
python -m app.graph.workflow data/raw/sales_clean.csv
```

---

## 5. REST API Endpoints

### 1. Health Check
`GET /health`

```bash
curl -X GET http://127.0.0.1:8000/health
```

### 2. Investigate Dataset
`POST /investigate`

```bash
curl -X POST http://127.0.0.1:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/raw/sales_problematic.csv",
    "approval_status": "APPROVED",
    "corrected_dataset_path": "data/test/fully_corrected.csv"
  }'
```

---

## 6. Running Tests

Run the complete test suite:

```bash
pytest -v
```

*Expected: **286 passed (0 failures)***.
