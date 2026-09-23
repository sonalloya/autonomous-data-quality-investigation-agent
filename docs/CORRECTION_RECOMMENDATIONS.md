# Correction Recommendation Documentation — Phase 7

## Overview
The **Correction Recommendation Agent** and **Correction Strategy Tool** convert diagnostic root-cause evidence into a prioritized, explainable remediation plan without modifying production data or raw source datasets.

---

## Safety Guarantee & Principles
> [!IMPORTANT]
> The Correction Recommendation Agent produces **recommendations only**. It does **NOT** execute database modifications, row deletions, or dataset updates. All data-altering or quarantining steps explicitly enforce `human_approval_required: true`.

---

## Recommendation Model Structure
Each recommendation is encapsulated in a `CorrectionRecommendation` structure:

- `recommendation_id`: Unique identifier (e.g., `REC-001`, `REC-002`).
- `priority`: Priority level (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
- `issue`: Human-readable summary of the data quality problem.
- `root_cause`: Identified root cause from Phase 6.
- `affected_columns`: Target dataset columns.
- `evidence`: Supporting empirical findings from upstream diagnostic tools.
- `action_steps`: Sequential, step-by-step remediation procedure.
- `expected_outcome`: Measurable target outcome upon remediation.
- `validation_criteria`: Verifiable criteria to be evaluated by Phase 8 Validation Agent.
- `human_approval_required`: Boolean flag (`true` for data-altering steps).
- `risk_level`: Assessment of operational risk (`HIGH`, `MEDIUM`, `LOW`).

---

## Correction Strategies Supported

1. **Duplicate Transaction Ingestion**:
   - Actions: Identify duplicate IDs → Determine canonical records → Quarantine duplicates → Recalculate revenue → Re-run validation.
   - Approval: `human_approval_required = true` (`HIGH` risk).

2. **Upstream Data Loss / Missing Fields**:
   - Actions: Identify affected records → Inspect source extraction logs → Recover authoritative values → Quarantine unrecoverable records → Reprocess batch.
   - Approval: `human_approval_required = true` (`HIGH` risk).

3. **Data Transformation & Type-Conversion Failure**:
   - Actions: Identify malformed strings (e.g. `"3 units"`) → Parse convertible values → Quarantine unparseable text → Cast logical dtypes → Re-run schema checks.
   - Approval: `human_approval_required = true` (`MEDIUM` risk).

4. **Date Parsing & Timestamp Format Inconsistency**:
   - Actions: Identify non-ISO dates → Normalize valid strings to ISO `%Y-%m-%d` → Quarantine invalid dates → Re-run temporal checks.
   - Approval: `human_approval_required = true` (`MEDIUM` risk).

5. **Invalid Categorical Domain Values**:
   - Actions: Identify out-of-domain values → Compare against reference domain → Map typos/aliases → Quarantine unknown values → Re-run domain checks.
   - Approval: `human_approval_required = true` (`MEDIUM` risk).

6. **Clean Dataset (`sales_clean.csv`)**:
   - Returns: `status = "no_action_required"`, summary: `"No corrective action required."`

7. **Undetermined Root Cause**:
   - Actions: Recommend log collection and human data governance review without inventing arbitrary data fixes (`human_approval_required = false`, safe analysis).

---

## CLI & Programmatic Usage

### CLI Execution
```bash
python -m app.agents.correction_agent data/raw/sales_problematic.csv
python -m app.agents.correction_agent data/raw/sales_clean.csv
```

### Python API
```python
from app.agents.correction_agent import run_correction_agent

response = run_correction_agent("data/raw/sales_problematic.csv", use_llm=True)
print("Correction Status:", response.status)
print("Human Approval Summary:", response.human_approval_summary)
print("Action Plan:\n", response.llm_interpretation)
```

---

## Verification & Test Results
- **242 Total Automated Tests Passing** (`pytest tests/test_phase7_correction.py`)
- Clean dataset (`sales_clean.csv`): `no_action_required`, 0 recommendations.
- Problematic dataset (`sales_problematic.csv`): `recommendations_generated`, 2 prioritized recommendations with `human_approval_required: true`.
- No data modification verified: Source dataset files remain completely untouched.
