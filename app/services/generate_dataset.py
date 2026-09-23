"""
generate_dataset.py
-------------------
Phase 2 — Dataset Creation and Data Quality Scenario Generation.

Generates all datasets needed for the Autonomous Data Quality Investigation
Agent demonstration scenario:

    1. Clean customer data        → data/raw/customers.csv
    2. Clean product data         → data/raw/products.csv
    3. Clean sales data           → data/raw/sales_clean.csv
    4. Problematic sales data     → data/raw/sales_problematic.csv
    5. Focused test datasets      → data/test/*.csv
    6. Ground-truth metadata      → data/test/ground_truth.json

All generation is deterministic; use RANDOM_SEED to reproduce the same
datasets on every run.

Run:
    python app/services/generate_dataset.py
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]   # project root
RAW_DIR  = BASE_DIR / "data" / "raw"
TEST_DIR = BASE_DIR / "data" / "test"

RAW_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED        = 42
N_CUSTOMERS        = 750
N_PRODUCTS         = 75
N_TRANSACTIONS     = 8_000   # target row count for clean dataset

# Problematic injection parameters
MISSING_RATE       = 0.04    # 4 % of records per targeted column
DUPLICATE_COUNT    = 180     # number of duplicate rows added
INVALID_RATE       = 0.02    # 2 % invalid-value records
ANOMALY_COUNT      = 35      # extreme-value records
CATEGORICAL_BAD    = 50      # bad-category records
FORMAT_BAD         = 40      # format-inconsistency records

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

REGIONS = ["North", "South", "East", "West", "Central"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Bank Transfer", "PayPal", "Cash"]
SALES_CHANNELS  = ["Online", "In-Store", "Phone", "Partner", "Direct Sales"]

CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Small Business"]

CITIES_BY_REGION: dict[str, list[str]] = {
    "North":   ["Chicago", "Minneapolis", "Detroit", "Milwaukee", "Cleveland"],
    "South":   ["Atlanta", "Houston", "Miami", "Dallas", "New Orleans"],
    "East":    ["New York", "Boston", "Philadelphia", "Washington DC", "Baltimore"],
    "West":    ["Los Angeles", "San Francisco", "Seattle", "Denver", "Phoenix"],
    "Central": ["Kansas City", "Indianapolis", "Columbus", "Memphis", "Louisville"],
}

PRODUCT_CATEGORIES = {
    "Electronics":   ["Laptop Pro", "Wireless Headphones", "Smart Tablet", "USB-C Hub",
                      "Bluetooth Speaker", "Webcam HD", "Mechanical Keyboard",
                      "Curved Monitor", "Gaming Mouse", "External SSD"],
    "Office Supplies": ["Premium Notebook", "Ergonomic Pen Set", "Desk Organizer",
                        "Sticky Notes Pack", "Whiteboard Markers", "Stapler Set",
                        "File Folders (50pk)", "Label Maker", "Paper Shredder",
                        "Printer Ink Bundle"],
    "Furniture":     ["Standing Desk", "Ergonomic Chair", "Monitor Stand", "Bookshelf",
                      "Filing Cabinet", "Meeting Table", "Reception Desk",
                      "Storage Ottoman", "Coat Rack", "Wall Shelving Unit"],
    "Software":      ["Project Mgmt License", "Antivirus Suite", "Design Suite",
                      "Accounting Software", "CRM Platform", "Cloud Backup (1yr)",
                      "Video Conf License", "Email Marketing Tool",
                      "HR Platform License", "Data Analytics Suite"],
    "Peripherals":   ["Ergonomic Mouse", "Numeric Keypad", "Drawing Tablet",
                      "VR Headset", "Smart Card Reader", "Docking Station",
                      "KVM Switch", "Barcode Scanner", "Fingerprint Reader",
                      "HDMI Matrix Switch"],
    "Networking":    ["WiFi Router Pro", "Network Switch 24pt", "Firewall Appliance",
                      "Network Cable (100m)", "PoE Injector", "Patch Panel",
                      "Server Rack 12U", "UPS 1500VA", "NAS Enclosure",
                      "SFP Transceiver"],
    "Accessories":   ["Laptop Bag", "Screen Cleaner Kit", "Cable Management Box",
                      "Monitor Privacy Screen", "Wrist Rest Pad", "Anti-Glare Filter",
                      "Surge Protector", "Extension Cord Pro", "RFID Wallet",
                      "Webcam Cover (3pk)"],
}

# Price ranges per category (min, max)
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "Electronics":     (49.99,  1_499.99),
    "Office Supplies": (4.99,   89.99),
    "Furniture":       (99.99,  2_999.99),
    "Software":        (29.99,  899.99),
    "Peripherals":     (19.99,  599.99),
    "Networking":      (24.99,  1_199.99),
    "Accessories":     (4.99,   79.99),
}

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Barbara", "David", "Susan", "Richard", "Jessica",
    "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Lisa", "Christopher",
    "Nancy", "Daniel", "Betty", "Matthew", "Margaret", "Anthony", "Sandra",
    "Mark", "Ashley", "Donald", "Dorothy", "Steven", "Kimberly", "Paul",
    "Emily", "Andrew", "Donna", "Joshua", "Michelle",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rng() -> np.random.Generator:
    """Return the global NumPy RNG (seeded once at module level)."""
    return _GLOBAL_RNG


_GLOBAL_RNG = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


def _round2(value: float) -> float:
    return round(float(value), 2)


def _random_id(prefix: str, num: int, width: int = 6) -> str:
    return f"{prefix}{str(num).zfill(width)}"


# ---------------------------------------------------------------------------
# 1. Generate customers
# ---------------------------------------------------------------------------

def generate_customers() -> pd.DataFrame:
    """Generate N_CUSTOMERS realistic customer records."""
    rng = _rng()
    customer_ids = [_random_id("CUST", i + 1) for i in range(N_CUSTOMERS)]

    regions      = rng.choice(REGIONS, size=N_CUSTOMERS).tolist()
    segments     = rng.choice(CUSTOMER_SEGMENTS, size=N_CUSTOMERS).tolist()

    names, cities = [], []
    for region in regions:
        first  = random.choice(FIRST_NAMES)
        last   = random.choice(LAST_NAMES)
        names.append(f"{first} {last}")
        cities.append(random.choice(CITIES_BY_REGION[region]))

    df = pd.DataFrame({
        "customer_id":       customer_ids,
        "customer_name":     names,
        "customer_segment":  segments,
        "city":              cities,
        "region":            regions,
    })
    return df


# ---------------------------------------------------------------------------
# 2. Generate products
# ---------------------------------------------------------------------------

def generate_products() -> pd.DataFrame:
    """Generate N_PRODUCTS realistic product records."""
    rng = _rng()
    rows = []
    product_num = 1

    for category, names_list in PRODUCT_CATEGORIES.items():
        per_cat = N_PRODUCTS // len(PRODUCT_CATEGORIES)
        # Sample products (with replacement if needed to hit quota)
        chosen = random.choices(names_list, k=per_cat)
        price_min, price_max = PRICE_RANGES[category]

        for name in chosen:
            price = _round2(rng.uniform(price_min, price_max))
            rows.append({
                "product_id":   _random_id("PROD", product_num),
                "product_name": name,
                "category":     category,
                "unit_price":   price,
            })
            product_num += 1

    # Pad to N_PRODUCTS if rounding left us short
    while len(rows) < N_PRODUCTS:
        cat  = random.choice(list(PRODUCT_CATEGORIES.keys()))
        name = random.choice(PRODUCT_CATEGORIES[cat])
        pmin, pmax = PRICE_RANGES[cat]
        rows.append({
            "product_id":   _random_id("PROD", product_num),
            "product_name": name,
            "category":     cat,
            "unit_price":   _round2(rng.uniform(pmin, pmax)),
        })
        product_num += 1

    df = pd.DataFrame(rows[:N_PRODUCTS])
    return df


# ---------------------------------------------------------------------------
# 3. Generate clean sales transactions
# ---------------------------------------------------------------------------

def generate_clean_sales(
    customers: pd.DataFrame,
    products:  pd.DataFrame,
) -> pd.DataFrame:
    """Generate N_TRANSACTIONS clean, internally consistent sales records."""
    rng = _rng()

    customer_ids = customers["customer_id"].tolist()
    product_ids  = products["product_id"].tolist()

    # Build product lookup for price
    product_price = dict(zip(products["product_id"], products["unit_price"]))

    # Date range: 2024-01-01 → 2024-12-31 (full year)
    start_date = pd.Timestamp("2024-01-01")
    end_date   = pd.Timestamp("2024-12-31")
    date_range_days = (end_date - start_date).days

    rows = []
    for i in range(N_TRANSACTIONS):
        txn_id     = _random_id("TXN", i + 1, width=7)
        cust_id    = random.choice(customer_ids)
        prod_id    = random.choice(product_ids)
        quantity   = int(rng.integers(1, 20))          # 1–19
        unit_price = product_price[prod_id]
        total      = _round2(quantity * unit_price)
        days_offset = int(rng.integers(0, date_range_days + 1))
        txn_date   = (start_date + pd.Timedelta(days=int(days_offset), unit="D")).strftime("%Y-%m-%d")
        region     = customers.loc[customers["customer_id"] == cust_id, "region"].values[0]
        payment    = random.choice(PAYMENT_METHODS)
        channel    = random.choice(SALES_CHANNELS)

        rows.append({
            "transaction_id":  txn_id,
            "customer_id":     cust_id,
            "product_id":      prod_id,
            "transaction_date": txn_date,
            "quantity":        quantity,
            "unit_price":      unit_price,
            "total_amount":    total,
            "region":          region,
            "payment_method":  payment,
            "sales_channel":   channel,
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 4. Generate problematic sales dataset
# ---------------------------------------------------------------------------

def generate_problematic_sales(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Introduce controlled data-quality problems into the clean sales dataset.

    Returns the problematic DataFrame and a change-log dict describing
    every injection (used to build ground_truth.json).
    """
    rng   = _rng()
    df    = clean_df.copy()
    log: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # A. MISSING VALUES
    # -----------------------------------------------------------------------
    missing_columns = {
        "customer_id":    MISSING_RATE,
        "region":         MISSING_RATE,
        "quantity":       MISSING_RATE * 0.75,
        "payment_method": MISSING_RATE * 0.5,
    }
    missing_log: dict[str, list[str]] = {}
    for col, rate in missing_columns.items():
        n_missing = max(1, int(len(df) * rate))
        idx = rng.choice(df.index, size=n_missing, replace=False).tolist()
        df.loc[idx, col] = np.nan
        missing_log[col] = [str(df.loc[i, "transaction_id"]) for i in idx]

    log["missing_values"] = {
        "columns_affected": list(missing_log.keys()),
        "counts": {col: len(ids) for col, ids in missing_log.items()},
        "transaction_ids_sample": {
            col: ids[:10] for col, ids in missing_log.items()
        },
    }

    # -----------------------------------------------------------------------
    # B. DUPLICATE TRANSACTIONS
    # -----------------------------------------------------------------------
    dup_source_idx = rng.choice(df.index, size=DUPLICATE_COUNT, replace=False).tolist()
    dup_rows       = df.loc[dup_source_idx].copy()
    dup_txn_ids    = df.loc[dup_source_idx, "transaction_id"].tolist()

    df = pd.concat([df, dup_rows], ignore_index=True)

    log["duplicates"] = {
        "count": DUPLICATE_COUNT,
        "transaction_ids": dup_txn_ids,
    }

    # -----------------------------------------------------------------------
    # C. INVALID VALUES
    # -----------------------------------------------------------------------
    n_invalid = max(1, int(len(df) * INVALID_RATE))
    inv_idx   = rng.choice(df.index, size=n_invalid, replace=False).tolist()
    invalid_records = []
    for i, idx in enumerate(inv_idx):
        txn_id = df.loc[idx, "transaction_id"]
        # Rotate among several invalid-value types
        problem_type = i % 4
        if problem_type == 0:
            df.loc[idx, "quantity"]    = -int(rng.integers(1, 10))
            issue = "negative_quantity"
        elif problem_type == 1:
            df.loc[idx, "quantity"]    = 0
            issue = "zero_quantity"
        elif problem_type == 2:
            df.loc[idx, "unit_price"]  = _round2(-rng.uniform(1, 100))
            df.loc[idx, "total_amount"] = _round2(
                df.loc[idx, "quantity"] * df.loc[idx, "unit_price"]
            )
            issue = "negative_unit_price"
        else:
            df.loc[idx, "total_amount"] = _round2(-rng.uniform(10, 1000))
            issue = "negative_total_amount"

        invalid_records.append({
            "transaction_id": str(txn_id),
            "row_index":      int(idx),
            "issue":          issue,
        })

    log["invalid_values"] = {
        "count": n_invalid,
        "records": invalid_records,
    }

    # -----------------------------------------------------------------------
    # D. ANOMALOUS TRANSACTIONS (extreme values)
    # -----------------------------------------------------------------------
    anom_idx = rng.choice(df.index, size=ANOMALY_COUNT, replace=False).tolist()
    anomaly_records = []
    for i, idx in enumerate(anom_idx):
        txn_id = df.loc[idx, "transaction_id"]
        if i % 2 == 0:
            # Extreme quantity
            extreme_qty = int(rng.integers(500, 2000))
            df.loc[idx, "quantity"]     = extreme_qty
            df.loc[idx, "total_amount"] = _round2(
                extreme_qty * float(df.loc[idx, "unit_price"])
            )
            anomaly_records.append({
                "transaction_id": str(txn_id),
                "type": "extreme_quantity",
                "value": extreme_qty,
            })
        else:
            # Extreme total_amount (corrupt — does not match qty × price)
            extreme_amt = _round2(rng.uniform(50_000, 250_000))
            df.loc[idx, "total_amount"] = extreme_amt
            anomaly_records.append({
                "transaction_id": str(txn_id),
                "type": "extreme_total_amount",
                "value": extreme_amt,
            })

    log["anomalies"] = {
        "count": ANOMALY_COUNT,
        "records": anomaly_records,
    }

    # -----------------------------------------------------------------------
    # E. CATEGORICAL INCONSISTENCY
    # -----------------------------------------------------------------------
    bad_regions   = ["UNKNOWN_REGION", "N/A", "TBD", "??"]
    bad_payments  = ["Cryptocurrency", "BARTER", "Unknown", "wire transfer"]
    bad_channels  = ["LEGACY", "Fax", "Unknown Channel", "walk-in"]

    cat_idx = rng.choice(df.index, size=CATEGORICAL_BAD, replace=False).tolist()
    cat_records = []
    for i, idx in enumerate(cat_idx):
        txn_id = df.loc[idx, "transaction_id"]
        cycle  = i % 3
        if cycle == 0:
            bad_val = random.choice(bad_regions)
            df.loc[idx, "region"] = bad_val
            col = "region"
        elif cycle == 1:
            bad_val = random.choice(bad_payments)
            df.loc[idx, "payment_method"] = bad_val
            col = "payment_method"
        else:
            bad_val = random.choice(bad_channels)
            df.loc[idx, "sales_channel"] = bad_val
            col = "sales_channel"

        cat_records.append({
            "transaction_id": str(txn_id),
            "column": col,
            "bad_value": bad_val,
        })

    log["categorical_issues"] = {
        "count": CATEGORICAL_BAD,
        "records": cat_records,
    }

    # -----------------------------------------------------------------------
    # F. DATA TYPE / FORMAT INCONSISTENCY
    # -----------------------------------------------------------------------
    # Cast mixed-type columns to object so string injection doesn't raise FutureWarning
    df["quantity"]         = df["quantity"].astype(object)
    df["transaction_date"] = df["transaction_date"].astype(object)
    df["unit_price"]       = df["unit_price"].astype(object)

    fmt_idx = rng.choice(df.index, size=FORMAT_BAD, replace=False).tolist()
    fmt_records = []
    for i, idx in enumerate(fmt_idx):
        txn_id = df.loc[idx, "transaction_id"]
        cycle  = i % 3
        if cycle == 0:
            # quantity stored as a string
            df.loc[idx, "quantity"] = str(df.loc[idx, "quantity"]) + ".0 units"
            issue = "quantity_as_string"
        elif cycle == 1:
            # inconsistent date format (MM/DD/YYYY instead of YYYY-MM-DD)
            raw_date = df.loc[idx, "transaction_date"]
            try:
                parsed = pd.to_datetime(raw_date)
                df.loc[idx, "transaction_date"] = parsed.strftime("%m/%d/%Y")
                issue = "date_format_inconsistency"
            except Exception:
                issue = "date_parse_error"
        else:
            # unit_price stored as a string with currency symbol
            df.loc[idx, "unit_price"] = f"${df.loc[idx, 'unit_price']}"
            issue = "unit_price_as_currency_string"

        fmt_records.append({
            "transaction_id": str(txn_id),
            "issue": issue,
            "row_index": int(idx),
        })

    log["format_issues"] = {
        "count": FORMAT_BAD,
        "records": fmt_records,
    }

    return df, log


# ---------------------------------------------------------------------------
# 5. Revenue anomaly computation
# ---------------------------------------------------------------------------

def compute_revenue_stats(
    clean_df: pd.DataFrame,
    prob_df:  pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute reported vs. expected revenue statistics.

    'Reported' revenue = sum of total_amount in the problematic dataset
    after converting to numeric (non-parseable values become NaN → excluded).
    """
    clean_revenue    = clean_df["total_amount"].astype(float).sum()

    # Coerce to numeric — format issues and strings will become NaN
    prob_revenue_raw = pd.to_numeric(prob_df["total_amount"], errors="coerce")
    prob_revenue     = prob_revenue_raw.sum()

    diff_abs     = prob_revenue - clean_revenue
    diff_pct     = (diff_abs / clean_revenue) * 100 if clean_revenue else 0.0

    return {
        "clean_total_revenue":       _round2(clean_revenue),
        "problematic_total_revenue": _round2(prob_revenue),
        "absolute_difference":       _round2(diff_abs),
        "percentage_difference":     _round2(diff_pct),
        "direction":                 "over" if diff_pct > 0 else "under",
    }


# ---------------------------------------------------------------------------
# 6. Focused test datasets
# ---------------------------------------------------------------------------

def generate_test_datasets(
    clean_df: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:
    """Create small, focused test datasets under data/test/."""
    rng = _rng()

    # -----------------------------------------------------------------------
    # missing_values.csv — 50 records, several columns set to NaN
    # -----------------------------------------------------------------------
    sample_mv = clean_df.sample(n=50, random_state=RANDOM_SEED).copy().reset_index(drop=True)
    mv_col_indices = {
        "customer_id":    rng.choice(50, size=8, replace=False).tolist(),
        "region":         rng.choice(50, size=6, replace=False).tolist(),
        "quantity":       rng.choice(50, size=5, replace=False).tolist(),
        "payment_method": rng.choice(50, size=4, replace=False).tolist(),
    }
    for col, idxs in mv_col_indices.items():
        sample_mv.loc[idxs, col] = np.nan

    sample_mv.to_csv(TEST_DIR / "missing_values.csv", index=False)

    # -----------------------------------------------------------------------
    # duplicates.csv — 40 base records + 20 exact duplicates
    # -----------------------------------------------------------------------
    base_dup = clean_df.sample(n=40, random_state=RANDOM_SEED + 1).copy().reset_index(drop=True)
    dup_rows = base_dup.sample(n=20, random_state=RANDOM_SEED + 2).copy()
    sample_dup = pd.concat([base_dup, dup_rows], ignore_index=True)
    sample_dup.to_csv(TEST_DIR / "duplicates.csv", index=False)

    # -----------------------------------------------------------------------
    # anomaly.csv — 30 records with a few extreme values injected
    # -----------------------------------------------------------------------
    sample_anom = clean_df.sample(n=30, random_state=RANDOM_SEED + 3).copy().reset_index(drop=True)
    anom_rows   = rng.choice(30, size=5, replace=False).tolist()
    for idx in anom_rows:
        sample_anom.loc[idx, "total_amount"] = _round2(rng.uniform(100_000, 500_000))
    sample_anom.to_csv(TEST_DIR / "anomaly.csv", index=False)

    # -----------------------------------------------------------------------
    # schema_issue.csv — 25 records with type/format inconsistencies
    # -----------------------------------------------------------------------
    sample_schema = clean_df.sample(n=25, random_state=RANDOM_SEED + 4).copy().reset_index(drop=True)
    # Cast to object dtype before injecting mixed-type string values
    sample_schema["quantity"]         = sample_schema["quantity"].astype(object)
    sample_schema["transaction_date"] = sample_schema["transaction_date"].astype(object)
    schema_rows   = rng.choice(25, size=5, replace=False).tolist()
    for idx in schema_rows:
        sample_schema.loc[idx, "quantity"] = str(sample_schema.loc[idx, "quantity"]) + " units"
    fmt_rows = rng.choice(25, size=4, replace=False).tolist()
    for idx in fmt_rows:
        raw = sample_schema.loc[idx, "transaction_date"]
        try:
            parsed = pd.to_datetime(raw)
            sample_schema.loc[idx, "transaction_date"] = parsed.strftime("%m/%d/%Y")
        except Exception:
            pass
    sample_schema.to_csv(TEST_DIR / "schema_issue.csv", index=False)

    # -----------------------------------------------------------------------
    # combined_issues.csv — 60 records with multiple issue types
    # -----------------------------------------------------------------------
    sample_comb = clean_df.sample(n=60, random_state=RANDOM_SEED + 5).copy().reset_index(drop=True)
    # missing
    sample_comb.loc[rng.choice(60, size=4, replace=False).tolist(), "customer_id"] = np.nan
    # duplicate rows
    extra = sample_comb.sample(n=10, random_state=RANDOM_SEED + 6).copy()
    sample_comb = pd.concat([sample_comb, extra], ignore_index=True)
    # invalid
    sample_comb.loc[rng.choice(len(sample_comb), size=3, replace=False).tolist(), "quantity"] = -5
    # anomaly
    sample_comb.loc[rng.choice(len(sample_comb), size=2, replace=False).tolist(), "total_amount"] = 999_999.99
    # bad category
    sample_comb.loc[rng.choice(len(sample_comb), size=2, replace=False).tolist(), "region"] = "UNKNOWN_REGION"
    sample_comb.to_csv(TEST_DIR / "combined_issues.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Ground-truth JSON
# ---------------------------------------------------------------------------

def build_ground_truth(
    injection_log:  dict[str, Any],
    revenue_stats:  dict[str, Any],
    clean_df:       pd.DataFrame,
    prob_df:        pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the ground_truth.json structure from recorded injections."""

    # Missing value column list
    mv_cols = injection_log["missing_values"]["columns_affected"]

    # Duplicate transaction IDs
    dup_txn_ids = injection_log["duplicates"]["transaction_ids"]

    # Invalid records
    invalid_records = [
        {"transaction_id": r["transaction_id"], "issue": r["issue"]}
        for r in injection_log["invalid_values"]["records"]
    ]

    # Anomalous records
    anom_records = [
        {"transaction_id": r["transaction_id"], "type": r["type"], "value": r["value"]}
        for r in injection_log["anomalies"]["records"]
    ]

    # Schema/format issues
    schema_issues = [
        {"transaction_id": r["transaction_id"], "issue": r["issue"]}
        for r in injection_log["format_issues"]["records"]
    ]

    # Categorical issues
    cat_issues = [
        {
            "transaction_id": r["transaction_id"],
            "column": r["column"],
            "bad_value": r["bad_value"],
        }
        for r in injection_log["categorical_issues"]["records"]
    ]

    ground_truth = {
        "_metadata": {
            "description": (
                "Ground truth for Phase 2 data-quality injection. "
                "Describes every intentionally introduced problem."
            ),
            "clean_dataset_rows": len(clean_df),
            "problematic_dataset_rows": len(prob_df),
            "random_seed": RANDOM_SEED,
        },
        "missing_values": {
            "expected": True,
            "columns": mv_cols,
            "counts_per_column": injection_log["missing_values"]["counts"],
            "sample_transaction_ids": injection_log["missing_values"]["transaction_ids_sample"],
        },
        "duplicates": {
            "expected": True,
            "count": injection_log["duplicates"]["count"],
            "transaction_ids": dup_txn_ids,
        },
        "invalid_values": {
            "expected": True,
            "count": injection_log["invalid_values"]["count"],
            "records": invalid_records,
        },
        "anomalies": {
            "expected": True,
            "count": ANOMALY_COUNT,
            "records": anom_records,
        },
        "categorical_issues": {
            "expected": True,
            "count": CATEGORICAL_BAD,
            "affected_columns": ["region", "payment_method", "sales_channel"],
            "records": cat_issues,
        },
        "schema_issues": {
            "expected": True,
            "count": FORMAT_BAD,
            "issue_types": [
                "quantity_as_string",
                "date_format_inconsistency",
                "unit_price_as_currency_string",
            ],
            "issues": schema_issues,
        },
        "revenue_anomaly": {
            "expected": True,
            "clean_total_revenue":       revenue_stats["clean_total_revenue"],
            "problematic_total_revenue": revenue_stats["problematic_total_revenue"],
            "absolute_difference":       revenue_stats["absolute_difference"],
            "expected_difference_percent": revenue_stats["percentage_difference"],
            "direction": revenue_stats["direction"],
            "contributing_factors": [
                "duplicate_transactions_inflate_revenue",
                "anomalous_extreme_total_amounts",
                "records_with_missing_or_invalid_quantity_excluded_from_sum",
                "negative_total_amounts_reduce_sum",
            ],
        },
        "issue_categories": [
            "missing_values",
            "duplicate_transactions",
            "invalid_values",
            "anomalous_values",
            "categorical_inconsistency",
            "data_type_format_inconsistency",
        ],
    }

    return ground_truth


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 2 - Dataset Generation")
    print("=" * 60)

    # 1. Customers
    print("\n[1/6] Generating customers...")
    customers = generate_customers()
    customers.to_csv(RAW_DIR / "customers.csv", index=False)
    print(f"      Saved {len(customers)} customer records -> data/raw/customers.csv")

    # 2. Products
    print("\n[2/6] Generating products...")
    products = generate_products()
    products.to_csv(RAW_DIR / "products.csv", index=False)
    print(f"      Saved {len(products)} product records -> data/raw/products.csv")

    # 3. Clean sales
    print(f"\n[3/6] Generating {N_TRANSACTIONS:,} clean sales transactions...")
    clean_sales = generate_clean_sales(customers, products)
    clean_sales.to_csv(RAW_DIR / "sales_clean.csv", index=False)
    clean_revenue = clean_sales["total_amount"].sum()
    print(f"      Saved {len(clean_sales):,} rows -> data/raw/sales_clean.csv")
    print(f"      Clean total revenue: ${clean_revenue:,.2f}")

    # 4. Problematic sales
    print("\n[4/6] Generating problematic sales dataset...")
    prob_sales, injection_log = generate_problematic_sales(clean_sales)
    prob_sales.to_csv(RAW_DIR / "sales_problematic.csv", index=False)
    revenue_stats = compute_revenue_stats(clean_sales, prob_sales)
    print(f"      Saved {len(prob_sales):,} rows -> data/raw/sales_problematic.csv")
    print(f"      Problematic revenue: ${revenue_stats['problematic_total_revenue']:,.2f}")
    print(f"      Revenue difference:  {revenue_stats['percentage_difference']:+.2f}%")

    # 5. Test datasets
    print("\n[5/6] Generating focused test datasets...")
    generate_test_datasets(clean_sales, products, customers)
    for fname in ["missing_values.csv", "duplicates.csv", "anomaly.csv",
                  "schema_issue.csv", "combined_issues.csv"]:
        rows = len(pd.read_csv(TEST_DIR / fname))
        print(f"      data/test/{fname}: {rows} rows")

    # 6. Ground truth
    print("\n[6/6] Building ground_truth.json...")
    ground_truth = build_ground_truth(injection_log, revenue_stats, clean_sales, prob_sales)
    with open(TEST_DIR / "ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, indent=2, default=str)
    print("      Saved -> data/test/ground_truth.json")

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  customers.csv          : {len(customers):,} rows")
    print(f"  products.csv           : {len(products):,} rows")
    print(f"  sales_clean.csv        : {len(clean_sales):,} rows")
    print(f"  sales_problematic.csv  : {len(prob_sales):,} rows")
    print(f"  Revenue anomaly        : {revenue_stats['percentage_difference']:+.2f}%")
    print(f"  Missing value columns  : {injection_log['missing_values']['columns_affected']}")
    print(f"  Duplicate rows added   : {injection_log['duplicates']['count']}")
    print(f"  Invalid value records  : {injection_log['invalid_values']['count']}")
    print(f"  Anomalous records      : {injection_log['anomalies']['count']}")
    print(f"  Categorical bad rows   : {injection_log['categorical_issues']['count']}")
    print(f"  Format/type bad rows   : {injection_log['format_issues']['count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
