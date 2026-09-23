# Validation Agent Documentation — Phase 8

## Overview
The **Validation Agent** and **Validation Tool** independently verify whether data-quality corrections applied to a dataset actually resolved the issues identified by previous phases.

---

## Core Principles & Guarantees
> [!IMPORTANT]
> - **INDEPENDENT VERIFICATION**: The Validation Agent independently re-runs `profiling_tool`, `anomaly_tool`, and `schema_tool` on the post-correction dataset. It does **NOT** trust previous assertions without empirical evidence.
> - **READ-ONLY GUARANTEE**: Validation is strictly read-only. It does **NOT** modify original datasets, post-correction CSV files, or DataFrames.
> - **DETERMINISTIC VERDICT**: Authoritative verdict states (`PASSED`, `PARTIAL`, `FAILED`, `NOT_APPLICABLE`) are determined by Python rules. Gemini LLM synthesizes narratives but cannot alter verdicts.

---

## Overall Validation Status Verdicts
- `PASSED`: All validation checks pass cleanly with 100% resolution (`failed_checks == 0` and `passed_checks > 0`).
- `PARTIAL`: Some validation checks pass while others linger (e.g., duplicates resolved, but missing values remain).
- `FAILED`: Primary correction objectives failed (`failed_checks > 0`).
- `NOT_APPLICABLE`: Original dataset was clean (`sales_clean.csv`); zero validation checks were required.

---

## Validation Check Categories

1. **Duplicate Validation**: Re-evaluates exact duplicate rows and duplicate transaction IDs on post-correction data. Checks `duplicate_transaction_ids == 0`.
2. **Completeness & Nullability Validation**: Re-evaluates missing values in required schema columns. Checks `nullability_violations == 0`.
3. **Schema & Type Validation**: Re-evaluates logical data types and string text suffixes (e.g., `"units"`). Checks `schema_valid == true` and `type_mismatches == 0`.
4. **Date Format Validation**: Re-evaluates date parseability against standard ISO 8601 `%Y-%m-%d`. Checks `format_issues == 0`.
5. **Business Metric Validation**: Calculates post-correction total monetary revenue and compares against baseline expected revenue.

---

## CLI & Programmatic Usage

### CLI Execution
```bash
# Validate uncorrected dataset (returns FAILED)
python -m app.agents.validation_agent data/raw/sales_problematic.csv

# Validate clean dataset (returns NOT_APPLICABLE)
python -m app.agents.validation_agent data/raw/sales_clean.csv
```

### Python API
```python
from app.agents.validation_agent import run_validation_agent

response = run_validation_agent(post_correction_df, original_df, use_llm=True)
print("Validation Status:", response.validation_status)
print("Passed Checks:", response.passed_checks, "/", response.total_checks)
print("Report Narrative:\n", response.llm_interpretation)
```

---

## Verification & Test Results
- **252 Total Automated Tests Passing** (`pytest tests/test_phase8_validation.py`)
- Clean dataset (`sales_clean.csv`): `NOT_APPLICABLE`
- Fully corrected dataset: `PASSED` (100% resolved)
- Partially corrected dataset: `PARTIAL`
- Uncorrected problematic dataset (`sales_problematic.csv`): `FAILED` (4 failed checks)
- Read-only guarantee: Source CSV files and DataFrames verified completely unchanged.
