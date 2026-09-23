# Autonomous Data Quality Investigation Agent — Technical Project Overview

## 1. Executive Summary & Problem Statement

Modern enterprise data architectures suffer from severe data quality issues: silent duplicate ingestion, missing foreign keys, column data type corruptions, and anomalous time-series spikes. Traditional rule-based monitors flag alerts but fail to:
1. Explain the underlying **root cause** of why data went wrong across upstream systems.
2. Formulate concrete **remedial actions** with quantified priority.
3. Enforce strict **human-in-the-loop governance** to prevent destructive data loss.
4. Independently **validate** that post-correction datasets actually resolve identified anomalies without introducing regressions.

The **Autonomous Data Quality Investigation Agent** solves this through a deterministic, agentic multi-stage architecture orchestrated via **LangGraph**.

---

## 2. System Architecture

```
                    ┌───────────────────────────┐
                    │      Input Dataset        │
                    │   (CSV - Read-Only)       │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Data Profiling Agent    │
                    │  (Metrics, Missing, Dups) │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Anomaly Detection Agent  │
                    │  (Z-Score, IQR, Isolation)│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Schema Analysis Agent   │
                    │ (Types, Constraints, Nulls│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Root Cause Investigation │
                    │ (Multi-Factor Bayes/Rules)│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Correction Recommendation │
                    │ (Prioritized Remediation) │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Human-in-the-Loop Gate   │
                    │ (APPROVED/PENDING/REJECT) │
                    └──────┬─────────────┬──────┘
                           │             │
              [APPROVED with             │ [PENDING /
             Corrected Dataset]          │  REJECTED]
                           │             │
                           ▼             │
                    ┌─────────────┐      │
                    │ Validation  │      │
                    │    Agent    │      │
                    └──────┬──────┘      │
                           │             │
                           ▼             ▼
                    ┌───────────────────────────┐
                    │    Final Report Engine    │
                    │ (JSON Contract & Summary) │
                    └───────────────────────────┘
```

---

## 3. Specialized Diagnostic & Decision Agents

| Agent / Stage | Underlying Tool | Primary Responsibilities |
|---|---|---|
| **Data Profiling** | `profiling_tool.py` | Calculates descriptive statistics, missingness matrices, duplicate row & transaction counts, and cardinality. |
| **Anomaly Detection** | `anomaly_tool.py` | Employs Z-score, IQR, and statistical bounds across temporal transaction dates and numeric volumes. |
| **Schema Analysis** | `schema_tool.py` | Validates column presence, strict types, nullability constraints, date formats, and foreign key integrity. |
| **Root Cause Agent** | `root_cause_tool.py` | Synthesizes multi-agent diagnostic findings into evidence-backed root cause hypotheses with confidence scores. |
| **Correction Agent** | `correction_tool.py` | Formulates structured, prioritized remediation plans (deduplication, imputation, casting, date standardizing). |
| **Approval Gate** | `approval_node` | Enforces explicit human operator authorization (`APPROVED`, `PENDING`, `REJECTED`) before any validation runs. |
| **Validation Agent** | `validation_tool.py` | Independently re-executes diagnostics on post-correction datasets; compares before/after metrics to issue authoritative verdicts (`PASSED`, `PARTIAL`, `FAILED`). |
| **Final Report** | `report.py` | Assembles deterministic findings and synthesis narratives into strict structured JSON and readable text representations. |

---

## 4. LangGraph Multi-Agent Orchestration

The system uses a state machine defined in `app/graph/workflow.py` over an immutable typed dictionary (`InvestigationState`):

```python
builder = StateGraph(InvestigationState)
# Sequential diagnostic pipeline with conditional branching
builder.add_node("profiling_node", profiling_node)
builder.add_node("anomaly_node", anomaly_node)
builder.add_node("schema_node", schema_node)
builder.add_node("root_cause_node", root_cause_node)
builder.add_node("correction_node", correction_node)
builder.add_node("approval_node", approval_node)
builder.add_node("validation_node", validation_node)
builder.add_node("final_report_node", final_report_node)
```

### Key Conditional Routes:
1. **Clean Datasets (`route_after_root_cause`)**: When zero defects exist, the workflow bypasses Correction and Validation, immediately completing with `NO_CORRECTION_REQUIRED`.
2. **Human Approval Gate (`route_after_approval`)**:
   - `APPROVED` + `corrected_dataset_path` $\rightarrow$ routes to `validation_node`.
   - `APPROVED` without corrected dataset $\rightarrow$ routes directly to `final_report_node`.
   - `PENDING` or `REJECTED` $\rightarrow$ routes directly to `final_report_node`, safely preventing unapproved validation runs.
3. **Error Handling**: Any node exception captures structured errors into `state["errors"]` and routes cleanly to `final_report_node` with status `FAILED`.

---

## 5. Human-in-the-Loop (HITL) & Safety Guarantees

1. **Read-Only Ingestion**: Raw datasets (`data/raw/sales_problematic.csv`) are opened with read-only semantics. SHA-256 hashes are verified before and after execution to guarantee zero mutation.
2. **No Autonomous Production Mutation**: The agent generates recommendations but never overwrites production tables.
3. **Deterministic Verdicts**: Validation statuses (`PASSED`, `PARTIAL`, `FAILED`) are computed purely through empirical checks, never hallucinated by LLMs.
4. **Isolated Test Fixtures**: Corrected datasets are maintained in `data/test/fully_corrected.csv` as isolated demonstration fixtures.

---

## 6. End-to-End Workflow Examples

### Scenario A: Problematic Data with Approval & Validation
- **Input**: `data/raw/sales_problematic.csv`
- **Approval**: `APPROVED`
- **Corrected Fixture**: `data/test/fully_corrected.csv`
- **Result**:
  - Detected: 1,063 missing values, 165 duplicates, 26 schema violations, 35 temporal anomalies.
  - Root Cause: `Duplicate transaction ingestion in data pipeline` (Confidence: `HIGH`).
  - Correction: Action plan generated.
  - Validation: `PASSED` (5/5 checks passed).
  - Workflow Status: `COMPLETED`.

### Scenario B: Problematic Data with Pending Approval
- **Input**: `data/raw/sales_problematic.csv`
- **Approval**: `PENDING`
- **Result**:
  - Diagnostic and correction plans generated.
  - Validation status: `NOT_VALIDATED` (skipped).
  - Workflow Status: `PENDING_APPROVAL`.

---

## 7. Interfaces

### FastAPI REST API
- `GET /health`: Service health and environment status.
- `POST /investigate`: Run investigation workflow with JSON payload and Pydantic validation.

### Python CLI
```bash
python app/graph/workflow.py data/raw/sales_problematic.csv --approval-status APPROVED --corrected-dataset-path data/test/fully_corrected.csv
```

### Standalone Interactive Demo
```bash
python examples/end_to_end_demo.py
```

---

## 8. Limitations & Future Enhancements

### Current Limitations:
- Supports single-table tabular CSV inputs.
- Schema presets configured for retail/sales domains.

### Future Roadmap:
- Multi-table relational join graph validation.
- Enterprise data warehouse connectors (Snowflake, BigQuery, PostgreSQL).
- Asynchronous Celery / Redis queue workers for multi-gigabyte files.
- Automated PR generation for ETL pipeline fixes.
