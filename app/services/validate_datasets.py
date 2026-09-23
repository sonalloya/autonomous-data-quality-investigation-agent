"""
validate_datasets.py
--------------------
Phase 2 — Dataset Validation Utility.

Verifies that all expected Phase 2 datasets exist, can be loaded,
contain the required columns, are non-empty, and satisfy basic
business-logic constraints for the clean sales dataset.

Run:
    python app/services/validate_datasets.py
    -- or --
    python -m app.services.validate_datasets

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = BASE_DIR / "data" / "raw"
TEST_DIR = BASE_DIR / "data" / "test"

# ---------------------------------------------------------------------------
# Expected files and their required columns
# ---------------------------------------------------------------------------

EXPECTED_FILES: dict[str, dict[str, Any]] = {
    str(RAW_DIR / "customers.csv"): {
        "required_columns": [
            "customer_id", "customer_name", "customer_segment", "city", "region"
        ],
        "min_rows": 100,
    },
    str(RAW_DIR / "products.csv"): {
        "required_columns": ["product_id", "product_name", "category", "unit_price"],
        "min_rows": 10,
    },
    str(RAW_DIR / "sales_clean.csv"): {
        "required_columns": [
            "transaction_id", "customer_id", "product_id",
            "transaction_date", "quantity", "unit_price", "total_amount",
            "region", "payment_method", "sales_channel",
        ],
        "min_rows": 1_000,
    },
    str(RAW_DIR / "sales_problematic.csv"): {
        "required_columns": [
            "transaction_id", "customer_id", "product_id",
            "transaction_date", "quantity", "unit_price", "total_amount",
            "region", "payment_method", "sales_channel",
        ],
        "min_rows": 1_000,
    },
    str(TEST_DIR / "missing_values.csv"): {
        "required_columns": ["transaction_id"],
        "min_rows": 1,
    },
    str(TEST_DIR / "duplicates.csv"): {
        "required_columns": ["transaction_id"],
        "min_rows": 1,
    },
    str(TEST_DIR / "anomaly.csv"): {
        "required_columns": ["transaction_id", "total_amount"],
        "min_rows": 1,
    },
    str(TEST_DIR / "schema_issue.csv"): {
        "required_columns": ["transaction_id"],
        "min_rows": 1,
    },
    str(TEST_DIR / "combined_issues.csv"): {
        "required_columns": ["transaction_id"],
        "min_rows": 1,
    },
}

GROUND_TRUTH_PATH = TEST_DIR / "ground_truth.json"

# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


class CheckResult:
    def __init__(self, name: str, status: str, detail: str = "") -> None:
        self.name   = name
        self.status = status
        self.detail = detail

    def __str__(self) -> str:
        icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(self.status, "[?]")
        line = f"  {icon} {self.name}"
        if self.detail:
            line += f"\n        {self.detail}"
        return line


Results: list[CheckResult] = []


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    if condition:
        Results.append(CheckResult(name, PASS))
        return True
    status = WARN if warn_only else FAIL
    Results.append(CheckResult(name, status, detail))
    return False


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_file_exists(path: str) -> bool:
    p = Path(path)
    return check(
        f"File exists: {p.name}",
        p.exists(),
        detail=f"Missing: {path}",
    )


def check_csv_loadable(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, dtype=str)   # dtype=str: load as-is, no coercion
        check(f"CSV loadable: {Path(path).name}", True)
        return df
    except Exception as exc:
        check(
            f"CSV loadable: {Path(path).name}",
            False,
            detail=str(exc),
        )
        return None


def check_columns(path: str, df: pd.DataFrame, required: list[str]) -> bool:
    missing = [c for c in required if c not in df.columns]
    return check(
        f"Required columns: {Path(path).name}",
        len(missing) == 0,
        detail=f"Missing columns: {missing}",
    )


def check_not_empty(path: str, df: pd.DataFrame, min_rows: int) -> bool:
    return check(
        f"Row count ≥ {min_rows}: {Path(path).name}",
        len(df) >= min_rows,
        detail=f"Only {len(df)} rows found (expected ≥ {min_rows})",
    )


# ---------------------------------------------------------------------------
# Clean-dataset business-logic checks
# ---------------------------------------------------------------------------

def check_clean_sales(df_raw: pd.DataFrame) -> None:
    """Business-logic validation for sales_clean.csv."""
    # Must be able to coerce key columns to numeric
    try:
        df = df_raw.copy()
        df["quantity"]     = pd.to_numeric(df["quantity"],     errors="coerce")
        df["unit_price"]   = pd.to_numeric(df["unit_price"],   errors="coerce")
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    except Exception as exc:
        check("Clean sales — numeric coercion", False, detail=str(exc))
        return

    # Unique transaction IDs
    dup_count = df_raw["transaction_id"].duplicated().sum()
    check(
        "Clean sales — transaction_id unique",
        dup_count == 0,
        detail=f"{dup_count} duplicate transaction_ids found in clean dataset",
    )

    # No negative quantities
    neg_qty = (df["quantity"] < 0).sum()
    check(
        "Clean sales — no negative quantity",
        neg_qty == 0,
        detail=f"{neg_qty} records with negative quantity",
    )

    # No negative unit prices
    neg_price = (df["unit_price"] < 0).sum()
    check(
        "Clean sales — no negative unit_price",
        neg_price == 0,
        detail=f"{neg_price} records with negative unit_price",
    )

    # total_amount ≈ quantity × unit_price (tolerance: $0.02 rounding)
    computed = (df["quantity"] * df["unit_price"]).round(2)
    mismatch = (abs(df["total_amount"] - computed) > 0.02).sum()
    check(
        "Clean sales — total_amount = qty × price",
        mismatch == 0,
        detail=f"{mismatch} records where total_amount ≠ quantity × unit_price",
    )

    # Dates parseable as YYYY-MM-DD
    try:
        parsed = pd.to_datetime(df_raw["transaction_date"], format="%Y-%m-%d", errors="coerce")
        bad_dates = parsed.isna().sum()
        check(
            "Clean sales — dates parseable (YYYY-MM-DD)",
            bad_dates == 0,
            detail=f"{bad_dates} rows with unparseable dates",
        )
    except Exception as exc:
        check("Clean sales — dates parseable", False, detail=str(exc))

    # No missing values in clean dataset
    missing_any = df_raw.isnull().any().any()
    check(
        "Clean sales — no missing values",
        not missing_any,
        detail="Clean dataset contains NaN values (unexpected)",
        warn_only=True,
    )


def check_referential_integrity(
    sales_df:     pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df:  pd.DataFrame,
) -> None:
    """Verify customer_id and product_id in clean sales exist in reference tables."""
    known_customers = set(customers_df["customer_id"].dropna().tolist())
    known_products  = set(products_df["product_id"].dropna().tolist())

    orphan_cust = sales_df["customer_id"].dropna()
    orphan_cust = orphan_cust[~orphan_cust.isin(known_customers)]
    check(
        "Referential integrity — customer_id",
        len(orphan_cust) == 0,
        detail=f"{len(orphan_cust)} sales rows with unknown customer_id",
    )

    orphan_prod = sales_df["product_id"].dropna()
    orphan_prod = orphan_prod[~orphan_prod.isin(known_products)]
    check(
        "Referential integrity — product_id",
        len(orphan_prod) == 0,
        detail=f"{len(orphan_prod)} sales rows with unknown product_id",
    )


def check_problematic_sales(df_raw: pd.DataFrame) -> None:
    """Confirm the problematic dataset actually contains expected issue types."""
    # Should have duplicates
    dup_count = df_raw["transaction_id"].duplicated().sum()
    check(
        "Problematic sales — has duplicate transaction_ids",
        dup_count > 0,
        detail=f"Expected duplicates but found 0",
    )

    # Should have missing values
    has_missing = df_raw.isnull().any().any()
    check(
        "Problematic sales — has missing values",
        has_missing,
        detail="Expected NaN values but found none",
    )

    # Row count > clean (due to duplicates being added)
    clean_path = RAW_DIR / "sales_clean.csv"
    if clean_path.exists():
        clean_rows = len(pd.read_csv(clean_path, dtype=str))
        check(
            "Problematic sales — row count > clean count",
            len(df_raw) > clean_rows,
            detail=f"Problematic: {len(df_raw)}, Clean: {clean_rows}",
        )


def check_ground_truth(gt_path: Path) -> None:
    """Validate structure of ground_truth.json."""
    if not gt_path.exists():
        check("ground_truth.json exists", False, detail=str(gt_path))
        return

    check("ground_truth.json exists", True)

    try:
        with open(gt_path, encoding="utf-8") as fh:
            gt = json.load(fh)
        check("ground_truth.json parseable", True)
    except Exception as exc:
        check("ground_truth.json parseable", False, detail=str(exc))
        return

    required_keys = [
        "missing_values", "duplicates", "invalid_values",
        "anomalies", "schema_issues", "revenue_anomaly",
    ]
    for key in required_keys:
        check(
            f"ground_truth.json has '{key}'",
            key in gt,
            detail=f"Key '{key}' missing from ground_truth.json",
        )

    # Revenue anomaly within expected range
    if "revenue_anomaly" in gt:
        pct = gt["revenue_anomaly"].get("expected_difference_percent", 0)
        check(
            f"Revenue anomaly ≥ 5% (got {pct:+.2f}%)",
            abs(pct) >= 5,
            detail=f"Revenue difference {pct:.2f}% is smaller than expected (≥ 5%)",
            warn_only=True,
        )


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

def run_validation() -> int:
    """Run all validation checks. Returns exit code (0 = pass, 1 = fail)."""
    print("=" * 60)
    print("Phase 2 - Dataset Validation")
    print("=" * 60)

    loaded: dict[str, pd.DataFrame | None] = {}

    # --- File existence + CSV loading ---
    print("\n[1] File existence and loading:")
    for path, spec in EXPECTED_FILES.items():
        if not check_file_exists(path):
            loaded[path] = None
            continue
        loaded[path] = check_csv_loadable(path)

    # --- Column checks ---
    print("\n[2] Column presence checks:")
    for path, spec in EXPECTED_FILES.items():
        df = loaded.get(path)
        if df is not None:
            check_columns(path, df, spec["required_columns"])

    # --- Row count checks ---
    print("\n[3] Row count checks:")
    for path, spec in EXPECTED_FILES.items():
        df = loaded.get(path)
        if df is not None:
            check_not_empty(path, df, spec["min_rows"])

    # --- Business logic: clean sales ---
    clean_path = str(RAW_DIR / "sales_clean.csv")
    print("\n[4] Clean sales business-logic checks:")
    if loaded.get(clean_path) is not None:
        check_clean_sales(loaded[clean_path])

    # --- Referential integrity ---
    cust_path = str(RAW_DIR / "customers.csv")
    prod_path = str(RAW_DIR / "products.csv")
    print("\n[5] Referential integrity checks:")
    if all(loaded.get(p) is not None for p in [clean_path, cust_path, prod_path]):
        check_referential_integrity(
            loaded[clean_path],
            loaded[cust_path],
            loaded[prod_path],
        )
    else:
        check("Referential integrity — skipped (files missing)", False,
              detail="One or more reference files could not be loaded")

    # --- Problematic dataset checks ---
    prob_path = str(RAW_DIR / "sales_problematic.csv")
    print("\n[6] Problematic dataset checks:")
    if loaded.get(prob_path) is not None:
        check_problematic_sales(loaded[prob_path])

    # --- Ground truth ---
    print("\n[7] Ground truth validation:")
    check_ground_truth(GROUND_TRUTH_PATH)

    # --- Summary ---
    passes  = sum(1 for r in Results if r.status == PASS)
    warns   = sum(1 for r in Results if r.status == WARN)
    fails   = sum(1 for r in Results if r.status == FAIL)
    total   = len(Results)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    for r in Results:
        print(str(r))

    print("\n" + "-" * 60)
    print(f"  Total checks : {total}")
    print(f"  [OK] Passed  : {passes}")
    print(f"  [W] Warnings : {warns}")
    print(f"  [F] Failed   : {fails}")
    print("=" * 60)

    if fails > 0:
        print("\nValidation FAILED -- fix the issues above.\n")
        return 1
    if warns > 0:
        print("\nValidation passed with warnings.\n")
        return 0
    print("\nAll validation checks passed.\n")
    return 0


if __name__ == "__main__":
    exit_code = run_validation()
    sys.exit(exit_code)
