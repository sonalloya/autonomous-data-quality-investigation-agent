# Data Directory — Phase 2 Documentation

## Overview

This directory contains all datasets used by the **Autonomous Data Quality
Investigation Agent**. The datasets simulate a realistic business scenario
where a company's sales dashboard suddenly reports an approximate 20% revenue
drop, and an agentic AI system must investigate the root cause.

---

## Directory Structure

```
data/
├── raw/
│   ├── customers.csv          # Reference table: customer master data
│   ├── products.csv           # Reference table: product catalog
│   ├── sales_clean.csv        # Baseline clean sales transactions
│   └── sales_problematic.csv  # Dirty copy with injected data-quality issues
│
├── test/
│   ├── missing_values.csv     # Focused: missing-value scenario
│   ├── duplicates.csv         # Focused: duplicate-transaction scenario
│   ├── anomaly.csv            # Focused: extreme-value / anomaly scenario
│   ├── schema_issue.csv       # Focused: type / format inconsistency scenario
│   ├── combined_issues.csv    # Combined multi-issue scenario
│   └── ground_truth.json      # Machine-readable description of all injections
│
└── processed/                 # Reserved for Phase 3 agent outputs
```

---

## Dataset Details

### `raw/customers.csv`

| Column              | Type   | Description                    |
|---------------------|--------|--------------------------------|
| `customer_id`       | string | Unique customer identifier (CUST000001–CUST000750) |
| `customer_name`     | string | Full name                      |
| `customer_segment`  | string | Consumer / Corporate / Small Business |
| `city`              | string | City name                      |
| `region`            | string | North / South / East / West / Central |

- **Approximate rows:** 750  
- **Clean state:** No missing values; all IDs unique; segments and regions from fixed valid sets.

---

### `raw/products.csv`

| Column         | Type    | Description                        |
|----------------|---------|------------------------------------|
| `product_id`   | string  | Unique product identifier (PROD000001–PROD000075) |
| `product_name` | string  | Descriptive product name           |
| `category`     | string  | Electronics / Office Supplies / Furniture / Software / Peripherals / Networking / Accessories |
| `unit_price`   | float   | Standard list price (USD)          |

- **Approximate rows:** 75  
- **Clean state:** No missing values; all IDs unique; prices are positive.

---

### `raw/sales_clean.csv`

The **baseline ground-truth** sales dataset used as the reference for revenue calculations.

| Column             | Type    | Description                               |
|--------------------|---------|-------------------------------------------|
| `transaction_id`   | string  | Unique transaction ID (TXN0000001–TXN0008000) |
| `customer_id`      | string  | FK → customers.csv                        |
| `product_id`       | string  | FK → products.csv                         |
| `transaction_date` | string  | Date in `YYYY-MM-DD` format               |
| `quantity`         | integer | Number of units sold (1–19)               |
| `unit_price`       | float   | Price per unit (USD)                      |
| `total_amount`     | float   | Exactly `quantity × unit_price` (USD)     |
| `region`           | string  | North / South / East / West / Central     |
| `payment_method`   | string  | Credit Card / Debit Card / Bank Transfer / PayPal / Cash |
| `sales_channel`    | string  | Online / In-Store / Phone / Partner / Direct Sales |

- **Approximate rows:** 8,000  
- **Clean state characteristics:**
  - `transaction_id` is unique in every row
  - `total_amount = quantity × unit_price` (to 2 decimal places)
  - All `quantity` values are positive integers (1–19)
  - All `unit_price` and `total_amount` values are positive
  - `transaction_date` is uniformly formatted as `YYYY-MM-DD`
  - All foreign keys resolve in `customers.csv` and `products.csv`
  - No missing values in any column

---

### `raw/sales_problematic.csv`

A copy of `sales_clean.csv` with **controlled data-quality problems injected**.
This is the dataset that the investigation agents will analyse.

**Approximate rows:** 8,180 (base 8,000 + 180 duplicate rows)

#### Injected Issues

| Category | Description | Approximate Scope |
|----------|-------------|-------------------|
| **A. Missing Values** | `customer_id`, `region`, `quantity`, `payment_method` set to `NaN` | ~4% per targeted column |
| **B. Duplicate Transactions** | 180 rows duplicated (same `transaction_id`) | 180 extra rows |
| **C. Invalid Values** | Negative quantity, zero quantity, negative `unit_price`, negative `total_amount` | ~2% of rows |
| **D. Anomalous Values** | Extreme `quantity` (500–2,000) or inflated `total_amount` ($50k–$250k) | 35 records |
| **E. Categorical Inconsistency** | `region` = `UNKNOWN_REGION`, invalid `payment_method`, invalid `sales_channel` | 50 records |
| **F. Type/Format Inconsistency** | `quantity` stored as `"5.0 units"`, dates as `MM/DD/YYYY`, `unit_price` as `"$12.99"` | 40 records |

#### Revenue Anomaly

The problematic dataset is designed so that the **sum of parseable `total_amount`
values differs noticeably from the clean dataset's total revenue**, driven by:

1. Duplicate transactions inflate the sum.
2. Extreme anomalous `total_amount` values add outlier revenue.
3. Negative `total_amount` records reduce the sum.
4. Records with string/non-numeric `total_amount` are excluded from numeric sums.

> The exact percentage difference is recorded in `data/test/ground_truth.json`
> under `revenue_anomaly.expected_difference_percent`.

---

## Test Datasets (`data/test/`)

Each test dataset is a **small, focused sample** intended for unit/integration
testing of individual data-quality detection components.

| File                  | Rows (approx.) | Primary Issue                         |
|-----------------------|----------------|---------------------------------------|
| `missing_values.csv`  | 50             | Controlled NaN values in 4 columns    |
| `duplicates.csv`      | 60             | 20 rows duplicated from 40 base rows  |
| `anomaly.csv`         | 30             | 5 extreme `total_amount` outliers      |
| `schema_issue.csv`    | 25             | 5 quantity-as-string, 4 date format   |
| `combined_issues.csv` | 70             | Missing + duplicates + invalid + anomaly + bad category |

---

## Ground Truth (`data/test/ground_truth.json`)

Machine-readable metadata documenting **every intentional injection** made to
`sales_problematic.csv`. Key sections:

```json
{
  "missing_values":   { "expected": true, "columns": [...], "counts_per_column": {...} },
  "duplicates":       { "expected": true, "count": 180, "transaction_ids": [...] },
  "invalid_values":   { "expected": true, "count": ..., "records": [...] },
  "anomalies":        { "expected": true, "count": 35, "records": [...] },
  "categorical_issues": { ... },
  "schema_issues":    { "expected": true, "count": 40, "issues": [...] },
  "revenue_anomaly":  { "expected": true, "expected_difference_percent": ..., "direction": "..." }
}
```

This file is used by:
- `tests/test_phase2_datasets.py` — automated validation
- Future agent evaluation scripts — to score detection accuracy

---

## Dataset Relationships

```
customers.csv ──────────┐
                         ├──> sales_clean.csv
products.csv  ──────────┘         │
                                   ├──> sales_problematic.csv  (injected issues)
                                   └──> data/test/*.csv        (focused subsets)
```

---

## Regenerating the Datasets

All datasets are **fully reproducible** using a fixed random seed (`42`).

```bash
# From the project root:
python app/services/generate_dataset.py
```

Running this command twice will produce byte-for-byte identical CSVs.

### Validating the Datasets

```bash
python app/services/validate_datasets.py
```

### Running Phase 2 Tests

```bash
pytest tests/test_phase2_datasets.py -v
```

---

## Notes for Developers

- **Do not commit** the generated CSV files if they are listed in `.gitignore`.
  Always regenerate from the script for a fresh environment.
- **Ground truth is for development only.** Do not expose `ground_truth.json`
  through any user-facing API or dashboard.
- The `data/processed/` directory is reserved for agent-generated outputs in
  Phase 3 and beyond; it should remain empty at the end of Phase 2.
