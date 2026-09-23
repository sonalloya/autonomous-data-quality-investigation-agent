# Schema Analysis Documentation — Phase 5

## Overview
The **Schema Analysis Agent** and **Schema Tool** validate dataset structure, data types, format consistency, nullability rules, and domain constraints against defined schemas.

---

## Architecture & Configuration

### Schema Definition Format
Schemas are stored as JSON files in `app/config/schemas/` (or provided dynamically as dictionaries).
The primary schema definition for sales transactions is [`app/config/schemas/sales_schema.json`](file:///c:/projects/SB%20projects/Autonomous%20Data%20Quality%20Investigation%20Agent/autonomous-data-quality-agent/app/config/schemas/sales_schema.json).

```json
{
  "dataset_name": "sales",
  "version": "1.0",
  "columns": [
    {
      "name": "transaction_id",
      "logical_type": "string",
      "required": true,
      "nullable": false
    },
    {
      "name": "transaction_date",
      "logical_type": "datetime",
      "required": true,
      "nullable": false,
      "expected_format": "%Y-%m-%d"
    },
    {
      "name": "quantity",
      "logical_type": "integer",
      "required": true,
      "nullable": false,
      "min_value_exclusive": 0
    },
    {
      "name": "unit_price",
      "logical_type": "float",
      "required": true,
      "nullable": false,
      "min_value_exclusive": 0.0
    },
    {
      "name": "total_amount",
      "logical_type": "float",
      "required": true,
      "nullable": false,
      "min_value": 0.0
    },
    {
      "name": "region",
      "logical_type": "string",
      "required": true,
      "nullable": false,
      "allowed_values": ["North", "South", "East", "West", "Central"]
    }
  ]
}
```

---

## Core Detection Capabilities

1. **Missing Columns**: Flags expected schema columns absent in the input dataset (`CRITICAL` for `transaction_id`, `HIGH` for other required fields).
2. **Unexpected Columns**: Identifies extra columns not defined in the schema (`LOW` severity).
3. **Type Mismatches**: Maps Pandas dtypes to logical types (`string`, `integer`, `float`, `datetime`, `boolean`). Ignores false positive differences between `int64` and `float64` when numbers are whole integers. Flags non-numeric strings in numeric columns (e.g. `"3 units"`).
4. **Date/Time Format Analysis**: Evaluates date parseability against `%Y-%m-%d` and counts format inconsistencies (e.g. `MM/DD/YYYY`).
5. **Nullability Checks**: Flags null/missing values in `nullable: false` columns.
6. **Constraint Checks**: Verifies numeric boundary constraints (`> 0`, `>= 0`) and categorical allowed domain lists.

---

## CLI & Programmatic Usage

### CLI Execution
```bash
python -m app.agents.schema_agent data/raw/sales_problematic.csv
python -m app.agents.schema_agent data/test/schema_issue.csv
```

### Python API
```python
from app.agents.schema_agent import run_schema_agent

response = run_schema_agent("data/raw/sales_clean.csv", schema="sales", use_llm=True)
print("Schema Valid:", response.schema_valid)
print("Issues Summary:", response.summary)
print("Interpretation:\n", response.llm_interpretation)
```

---

## Verification & Test Results
- **221 Total Automated Tests Passing** (`pytest tests/test_phase5_schema.py`)
- Clean dataset (`sales_clean.csv`): `schema_valid = True`, 0 issues.
- Problematic dataset (`sales_problematic.csv`): `schema_valid = False`, 13 issues flagged cleanly.
- Test dataset (`schema_issue.csv`): `schema_valid = False`, 2 issues flagged cleanly.

---

## Known Limitations
- The Schema Analysis Agent identifies structural non-conformance but does not determine the multi-factor root cause (Phase 6).
- Final data cleaning / correction recommendations belong to Phase 7.
