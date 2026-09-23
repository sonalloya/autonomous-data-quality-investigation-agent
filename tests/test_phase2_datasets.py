"""
test_phase2_datasets.py
-----------------------
Automated pytest tests for Phase 2 datasets.

Verifies structural integrity, business rules, and expected data-quality
properties across all generated datasets.

Run:
    pytest tests/test_phase2_datasets.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw"
TEST_DIR = BASE_DIR / "data" / "test"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def customers() -> pd.DataFrame:
    path = RAW_DIR / "customers.csv"
    if not path.exists():
        pytest.skip("customers.csv not found — run generate_dataset.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def products() -> pd.DataFrame:
    path = RAW_DIR / "products.csv"
    if not path.exists():
        pytest.skip("products.csv not found — run generate_dataset.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def sales_clean() -> pd.DataFrame:
    path = RAW_DIR / "sales_clean.csv"
    if not path.exists():
        pytest.skip("sales_clean.csv not found — run generate_dataset.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def sales_problematic() -> pd.DataFrame:
    path = RAW_DIR / "sales_problematic.csv"
    if not path.exists():
        pytest.skip("sales_problematic.csv not found — run generate_dataset.py first")
    return pd.read_csv(path, dtype=str)


@pytest.fixture(scope="session")
def ground_truth() -> dict:
    path = TEST_DIR / "ground_truth.json"
    if not path.exists():
        pytest.skip("ground_truth.json not found — run generate_dataset.py first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. customers.csv tests
# ---------------------------------------------------------------------------

class TestCustomers:
    def test_file_exists(self):
        assert (RAW_DIR / "customers.csv").exists(), "customers.csv missing"

    def test_row_count(self, customers):
        assert len(customers) >= 100, f"Expected ≥ 100 customers, got {len(customers)}"

    def test_required_columns(self, customers):
        required = ["customer_id", "customer_name", "customer_segment", "city", "region"]
        for col in required:
            assert col in customers.columns, f"Column '{col}' missing from customers.csv"

    def test_unique_customer_ids(self, customers):
        dups = customers["customer_id"].duplicated().sum()
        assert dups == 0, f"{dups} duplicate customer_ids found"

    def test_no_null_ids(self, customers):
        nulls = customers["customer_id"].isna().sum()
        assert nulls == 0, f"{nulls} null customer_ids found"

    def test_valid_segments(self, customers):
        valid = {"Consumer", "Corporate", "Small Business"}
        found = set(customers["customer_segment"].dropna().unique())
        unknown = found - valid
        assert not unknown, f"Unknown segments: {unknown}"

    def test_valid_regions(self, customers):
        valid = {"North", "South", "East", "West", "Central"}
        found = set(customers["region"].dropna().unique())
        unknown = found - valid
        assert not unknown, f"Unknown regions in customers: {unknown}"


# ---------------------------------------------------------------------------
# 2. products.csv tests
# ---------------------------------------------------------------------------

class TestProducts:
    def test_file_exists(self):
        assert (RAW_DIR / "products.csv").exists(), "products.csv missing"

    def test_row_count(self, products):
        assert len(products) >= 10, f"Expected ≥ 10 products, got {len(products)}"

    def test_required_columns(self, products):
        required = ["product_id", "product_name", "category", "unit_price"]
        for col in required:
            assert col in products.columns, f"Column '{col}' missing from products.csv"

    def test_unique_product_ids(self, products):
        dups = products["product_id"].duplicated().sum()
        assert dups == 0, f"{dups} duplicate product_ids"

    def test_positive_unit_prices(self, products):
        neg = (products["unit_price"] < 0).sum()
        assert neg == 0, f"{neg} products have negative unit_price"

    def test_no_zero_prices(self, products):
        zeros = (products["unit_price"] == 0).sum()
        assert zeros == 0, f"{zeros} products have zero unit_price"

    def test_no_null_prices(self, products):
        nulls = products["unit_price"].isna().sum()
        assert nulls == 0, f"{nulls} products have null unit_price"


# ---------------------------------------------------------------------------
# 3. sales_clean.csv tests
# ---------------------------------------------------------------------------

class TestCleanSales:
    REQUIRED_COLS = [
        "transaction_id", "customer_id", "product_id",
        "transaction_date", "quantity", "unit_price",
        "total_amount", "region", "payment_method", "sales_channel",
    ]

    def test_file_exists(self):
        assert (RAW_DIR / "sales_clean.csv").exists(), "sales_clean.csv missing"

    def test_row_count(self, sales_clean):
        assert len(sales_clean) >= 1_000, f"Expected ≥ 1000 rows, got {len(sales_clean)}"

    def test_required_columns(self, sales_clean):
        for col in self.REQUIRED_COLS:
            assert col in sales_clean.columns, f"Column '{col}' missing"

    def test_no_null_transaction_ids(self, sales_clean):
        nulls = sales_clean["transaction_id"].isna().sum()
        assert nulls == 0, f"{nulls} null transaction_ids in clean data"

    def test_unique_transaction_ids(self, sales_clean):
        dups = sales_clean["transaction_id"].duplicated().sum()
        assert dups == 0, f"{dups} duplicate transaction_ids in clean data"

    def test_no_missing_values(self, sales_clean):
        missing = sales_clean.isnull().sum().sum()
        assert missing == 0, f"Clean dataset has {missing} missing values"

    def test_positive_quantity(self, sales_clean):
        neg = (sales_clean["quantity"] < 1).sum()
        assert neg == 0, f"{neg} rows with quantity < 1"

    def test_positive_unit_price(self, sales_clean):
        neg = (sales_clean["unit_price"] <= 0).sum()
        assert neg == 0, f"{neg} rows with unit_price ≤ 0"

    def test_positive_total_amount(self, sales_clean):
        neg = (sales_clean["total_amount"] <= 0).sum()
        assert neg == 0, f"{neg} rows with total_amount ≤ 0"

    def test_total_amount_equals_qty_times_price(self, sales_clean):
        computed = (sales_clean["quantity"] * sales_clean["unit_price"]).round(2)
        mismatch = (abs(sales_clean["total_amount"] - computed) > 0.02).sum()
        assert mismatch == 0, f"{mismatch} rows where total_amount ≠ qty × price"

    def test_date_format(self, sales_clean):
        parsed = pd.to_datetime(
            sales_clean["transaction_date"], format="%Y-%m-%d", errors="coerce"
        )
        bad = parsed.isna().sum()
        assert bad == 0, f"{bad} rows with non-YYYY-MM-DD dates in clean data"

    def test_valid_regions(self, sales_clean):
        valid = {"North", "South", "East", "West", "Central"}
        found = set(sales_clean["region"].dropna().unique())
        assert found <= valid, f"Unexpected regions: {found - valid}"

    def test_valid_payment_methods(self, sales_clean):
        valid = {"Credit Card", "Debit Card", "Bank Transfer", "PayPal", "Cash"}
        found = set(sales_clean["payment_method"].dropna().unique())
        assert found <= valid, f"Unexpected payment methods: {found - valid}"

    def test_valid_sales_channels(self, sales_clean):
        valid = {"Online", "In-Store", "Phone", "Partner", "Direct Sales"}
        found = set(sales_clean["sales_channel"].dropna().unique())
        assert found <= valid, f"Unexpected sales channels: {found - valid}"

    def test_referential_integrity_customer(self, sales_clean, customers):
        known = set(customers["customer_id"].dropna())
        orphans = set(sales_clean["customer_id"].dropna()) - known
        assert not orphans, f"{len(orphans)} unknown customer_ids in clean sales"

    def test_referential_integrity_product(self, sales_clean, products):
        known = set(products["product_id"].dropna())
        orphans = set(sales_clean["product_id"].dropna()) - known
        assert not orphans, f"{len(orphans)} unknown product_ids in clean sales"


# ---------------------------------------------------------------------------
# 4. sales_problematic.csv tests
# ---------------------------------------------------------------------------

class TestProblematicSales:
    def test_file_exists(self):
        assert (RAW_DIR / "sales_problematic.csv").exists(), "sales_problematic.csv missing"

    def test_row_count_greater_than_clean(self, sales_problematic):
        clean_path = RAW_DIR / "sales_clean.csv"
        if not clean_path.exists():
            pytest.skip("Clean dataset not found")
        clean_count = len(pd.read_csv(clean_path))
        assert len(sales_problematic) > clean_count, (
            f"Problematic dataset ({len(sales_problematic)}) should be larger "
            f"than clean ({clean_count}) due to added duplicates"
        )

    def test_has_duplicate_transaction_ids(self, sales_problematic):
        dups = sales_problematic["transaction_id"].duplicated().sum()
        assert dups > 0, "Expected duplicate transaction_ids but found none"

    def test_has_missing_values(self, sales_problematic):
        missing = sales_problematic.isnull().sum().sum()
        assert missing > 0, "Expected missing values but found none"

    def test_missing_in_expected_columns(self, sales_problematic):
        expected_missing = ["customer_id", "region", "quantity", "payment_method"]
        for col in expected_missing:
            if col in sales_problematic.columns:
                cnt = sales_problematic[col].isna().sum()
                assert cnt > 0, f"Expected missing values in '{col}' but found none"


# ---------------------------------------------------------------------------
# 5. Test datasets
# ---------------------------------------------------------------------------

class TestTestDatasets:
    @pytest.mark.parametrize("filename", [
        "missing_values.csv",
        "duplicates.csv",
        "anomaly.csv",
        "schema_issue.csv",
        "combined_issues.csv",
    ])
    def test_file_exists(self, filename):
        assert (TEST_DIR / filename).exists(), f"{filename} missing"

    @pytest.mark.parametrize("filename", [
        "missing_values.csv",
        "duplicates.csv",
        "anomaly.csv",
        "schema_issue.csv",
        "combined_issues.csv",
    ])
    def test_not_empty(self, filename):
        df = pd.read_csv(TEST_DIR / filename, dtype=str)
        assert len(df) > 0, f"{filename} is empty"

    def test_missing_values_has_nulls(self):
        df = pd.read_csv(TEST_DIR / "missing_values.csv")
        assert df.isnull().sum().sum() > 0, "missing_values.csv has no NaN values"

    def test_duplicates_has_duplicate_ids(self):
        df = pd.read_csv(TEST_DIR / "duplicates.csv", dtype=str)
        dups = df["transaction_id"].duplicated().sum()
        assert dups > 0, "duplicates.csv has no duplicate transaction_ids"

    def test_anomaly_has_extreme_values(self):
        df = pd.read_csv(TEST_DIR / "anomaly.csv")
        # At least one total_amount should be very large
        max_amt = df["total_amount"].max()
        assert max_amt > 10_000, f"Expected extreme total_amount but max is {max_amt}"

    def test_schema_has_mixed_types(self):
        # Read as str to avoid coercion
        df = pd.read_csv(TEST_DIR / "schema_issue.csv", dtype=str)
        # Some quantity values should end with " units"
        qty_col = df["quantity"].dropna()
        has_string = qty_col.str.contains("units", na=False).any()
        # Or some dates should be in MM/DD/YYYY format
        date_col = df["transaction_date"].dropna()
        has_slash_date = date_col.str.match(r"\d{2}/\d{2}/\d{4}").any()
        assert has_string or has_slash_date, (
            "schema_issue.csv should contain at least one type/format inconsistency"
        )


# ---------------------------------------------------------------------------
# 6. Ground truth tests
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_file_exists(self):
        assert (TEST_DIR / "ground_truth.json").exists(), "ground_truth.json missing"

    def test_parseable(self, ground_truth):
        assert isinstance(ground_truth, dict), "ground_truth.json is not a dict"

    def test_required_keys(self, ground_truth):
        required = [
            "missing_values", "duplicates", "invalid_values",
            "anomalies", "schema_issues", "revenue_anomaly",
        ]
        for key in required:
            assert key in ground_truth, f"Key '{key}' missing from ground_truth.json"

    def test_missing_values_documented(self, ground_truth):
        mv = ground_truth["missing_values"]
        assert mv.get("expected") is True
        assert len(mv.get("columns", [])) > 0, "No columns documented for missing values"

    def test_duplicates_documented(self, ground_truth):
        dup = ground_truth["duplicates"]
        assert dup.get("expected") is True
        assert dup.get("count", 0) > 0, "Duplicate count should be > 0"
        assert len(dup.get("transaction_ids", [])) > 0, "No duplicate transaction_ids listed"

    def test_revenue_anomaly_documented(self, ground_truth):
        rev = ground_truth["revenue_anomaly"]
        assert rev.get("expected") is True
        pct = abs(rev.get("expected_difference_percent", 0))
        assert pct >= 5, f"Revenue difference {pct:.2f}% seems too small (expected ≥ 5%)"

    def test_invalid_values_documented(self, ground_truth):
        inv = ground_truth["invalid_values"]
        assert inv.get("expected") is True
        assert inv.get("count", 0) > 0

    def test_anomalies_documented(self, ground_truth):
        anom = ground_truth["anomalies"]
        assert anom.get("expected") is True
        assert anom.get("count", 0) > 0

    def test_schema_issues_documented(self, ground_truth):
        schema = ground_truth["schema_issues"]
        assert schema.get("expected") is True
        assert schema.get("count", 0) > 0
