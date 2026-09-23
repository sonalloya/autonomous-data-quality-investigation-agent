# Root Cause Investigation Documentation — Phase 6

## Overview
The **Root Cause Investigation Agent** and **Root Cause Tool** correlate evidence findings across Data Profiling, Anomaly Detection, and Schema Analysis tools to determine the most likely underlying causes for detected data-quality defects.

---

## Evidence Model & Normalization
Findings from diagnostic tools are converted into a standardized `EvidenceItem` model:

- `source`: `"profiling"`, `"anomaly"`, or `"schema"`
- `finding_type`: `duplicate_records`, `missing_values`, `metric_anomaly`, `distribution_anomaly`, `numerical_outlier`, `missing_column`, `unexpected_column`, `type_mismatch`, `format_issue`, `nullability_violation`, `constraint_violation`
- `severity`: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`
- `column`: Relevant column name
- `observed_value` & `expected_value`: Empirical evidence metrics

---

## Correlation Rules & Candidate Generation

1. **Duplicate Transaction Ingestion**:
   - Evidence: Duplicate records in profiling/anomaly tools + revenue metric anomalies.
2. **Upstream Data Loss or Missing Fields**:
   - Evidence: Nullability violations or missing values in required fields (`customer_id`, `region`, `quantity`, `payment_method`) + missing columns or row count loss.
3. **Data Transformation and Type-Conversion Failure**:
   - Evidence: Schema type mismatches (e.g., `quantity` as string with `"units"` suffix) + numeric boundary constraint violations (`quantity > 0`, `unit_price > 0`).
4. **Date Parsing and Timestamp Inconsistency**:
   - Evidence: Format issues (e.g., `MM/DD/YYYY` vs ISO `YYYY-MM-DD`) + temporal date anomalies.
5. **Invalid Categorical Domain Values**:
   - Evidence: Categorical constraint violations in `region`, `payment_method`, or `sales_channel`.
6. **Natural Business Price Range Variation (Clean Data)**:
   - Evidence: Statistical numerical outliers occurring in a structurally valid schema with zero duplicate, missing, or format errors.

---

## Explainable Ranking & Confidence Scoring
Candidates are scored based on:
- Sum of severity weights of supporting findings (`CRITICAL`: 3.0, `HIGH`: 2.0, `MEDIUM`: 1.0, `LOW`: 0.5)
- Multi-tool agreement bonus (+1.5 per additional agreeing tool)
- Subtraction penalty for contradictory evidence

### Confidence Levels
- **HIGH**: Score ≥ 4.0 with evidence from ≥ 2 independent tools.
- **MEDIUM**: Score ≥ 2.0.
- **LOW**: Score < 2.0.

---

## Handling Inconclusive / Clean Scenarios
- If no data quality issues exist (`sales_clean.csv`), `investigation_status = "no_issues_detected"` and primary cause is `"No Data Quality Issues Detected"`.
- If evidence is weak or contradictory, primary cause is `"Undetermined"` with documented unresolved questions.

---

## CLI & Programmatic Usage

### CLI Execution
```bash
python -m app.agents.root_cause_agent data/raw/sales_problematic.csv
python -m app.agents.root_cause_agent data/test/combined_issues.csv
```

### Python API
```python
from app.agents.root_cause_agent import run_root_cause_agent

response = run_root_cause_agent("data/raw/sales_problematic.csv", use_llm=True)
print("Primary Root Cause:", response.primary_root_cause)
print("Contributing Causes:", response.contributing_causes)
print("Narrative:\n", response.llm_interpretation)
```

---

## Verification & Test Results
- **232 Total Automated Tests Passing** (`pytest tests/test_phase6_root_cause.py`)
- `sales_clean.csv`: `no_issues_detected`
- `sales_problematic.csv`: Identified `Duplicate transaction ingestion in data pipeline` as primary root cause (`HIGH` confidence) and `Upstream data loss or missing required fields` as contributing cause (`HIGH` confidence).

---

## Limitations
- Root cause determination relies strictly on empirical evidence provided by upstream tools.
- Correlation does not guarantee physical causation without external pipeline log verification (addressed in Phase 7/8).
