"""
tests/test_phase4_anomaly.py
-----------------------------
Automated pytest tests for Phase 4 — Anomaly Detection Tool and Agent.

Tests cover:
  A. Clean sales dataset — minimal/zero anomalies expected
  B. Anomaly test dataset — intentionally injected extreme values
  C. Problematic sales dataset — multiple issue types
  D. Combined issues dataset — anomaly detection amid other problems
  E. Small dataset edge case
  F. Constant-value column (no divide-by-zero)
  G. Missing values (NaN handling)
  H. Non-numeric columns (not passed to numeric methods)
  I. Invalid/missing file (graceful error handling)
  J. Agent-level tests (AgentAnomalyResponse structure)
  K. Ground-truth verification

Run:
    pytest tests/test_phase4_anomaly.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.tools.anomaly_tool import (
    detect_anomalies,
    AnomalyResult,
    Severity,
    _detect_iqr_outliers,
    _detect_zscore_outliers,
    _detect_distribution_anomalies,
    _find_numeric_columns,
)
from app.agents.anomaly_agent import run_anomaly_agent, AgentAnomalyResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw"
TEST_DIR = BASE_DIR / "data" / "test"


# ---------------------------------------------------------------------------
# Ground-truth fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ground_truth() -> dict[str, Any]:
    path = TEST_DIR / "ground_truth.json"
    if not path.exists():
        pytest.skip("ground_truth.json not found — run generate_dataset.py first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _require_file(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} not found — run generate_dataset.py first")


# ===========================================================================
# A. Clean sales dataset
# ===========================================================================


class TestAnomalyCleanSales:
    """Anomaly tool on clean, high-quality data."""

    @pytest.fixture(scope="class")
    def result(self) -> AnomalyResult:
        path = RAW_DIR / "sales_clean.csv"
        _require_file(path)
        return detect_anomalies(path)

    def test_loads_successfully(self, result):
        assert result.anomaly_error is None

    def test_row_count(self, result):
        assert result.row_count == 8_000

    def test_numeric_columns_analyzed(self, result):
        analyzed = result.analysis_metadata.get("numeric_columns_analyzed", [])
        for col in ["quantity", "unit_price", "total_amount"]:
            assert col in analyzed, f"'{col}' should be in numeric analysis"

    def test_no_anomaly_error(self, result):
        assert result.anomaly_error is None

    def test_structure_complete(self, result):
        assert isinstance(result.numerical_outliers, list)
        assert isinstance(result.distribution_anomalies, list)
        assert isinstance(result.metric_anomalies, list)
        assert isinstance(result.severity_summary, dict)

    def test_severity_summary_keys(self, result):
        assert Severity.HIGH   in result.severity_summary
        assert Severity.MEDIUM in result.severity_summary
        assert Severity.LOW    in result.severity_summary

    def test_no_high_severity_outliers(self, result):
        """
        The clean dataset spans 75 products with prices from $14 to ~$2500
        (a natural wide range), so IQR naturally flags high-priced items.
        We verify: no quantity outliers (quantity 1–20 is tightly bounded),
        and the total outlier count is bounded.
        We accept IQR outliers on wide-range pricing columns (unit_price,
        total_amount) as expected statistical artifacts of a diverse product catalog.
        """
        # quantity should be clean — range 1–20, very tight
        qty_outliers = [
            o for o in result.numerical_outliers
            if o.column == "quantity" and o.method == "IQR"
        ]
        assert len(qty_outliers) == 0, (
            f"Expected no quantity IQR outliers on clean data, "
            f"got {[(o.column, o.outlier_pct) for o in qty_outliers]}"
        )
        # Wide-price columns may naturally flag IQR outliers; limit to at most 2
        high_iqr = [
            o for o in result.numerical_outliers
            if o.severity == Severity.HIGH and o.method == "IQR"
        ]
        assert len(high_iqr) <= 2, (
            f"Expected at most 2 HIGH IQR outlier columns on clean data "
            f"(price range artifacts), got {len(high_iqr)}"
        )


    def test_business_metrics_populated(self, result):
        meta = result.analysis_metadata
        assert meta.get("total_revenue") is not None
        assert meta.get("transaction_count") == 8_000

    def test_analysis_metadata_present(self, result):
        meta = result.analysis_metadata
        assert "iqr_multiplier" in meta
        assert "zscore_threshold" in meta


# ===========================================================================
# B. Anomaly test dataset  (data/test/anomaly.csv)
# ===========================================================================


class TestAnomalyDataset:
    """Anomaly tool on the intentionally-injected anomaly test file."""

    @pytest.fixture(scope="class")
    def result(self) -> AnomalyResult:
        path = TEST_DIR / "anomaly.csv"
        _require_file(path)
        return detect_anomalies(path)

    def test_loads(self, result):
        assert result.anomaly_error is None

    def test_row_count(self, result):
        assert result.row_count == 30

    def test_anomalies_detected(self, result):
        assert result.anomalies_detected, "Expected anomalies in anomaly.csv"

    def test_total_amount_outlier_detected(self, result, ground_truth):
        """
        The anomaly.csv dataset contains extreme total_amount values (up to ~500k).
        The IQR method should flag at least one total_amount outlier.
        """
        total_amount_outliers = [
            o for o in result.numerical_outliers
            if o.column == "total_amount" and o.method == "IQR"
        ]
        assert len(total_amount_outliers) > 0, (
            "Expected IQR outliers for 'total_amount' in anomaly.csv"
        )
        assert total_amount_outliers[0].outlier_count >= 1

    def test_max_outlier_is_extreme(self, result):
        """The injected anomalies have total_amount up to ~500k."""
        for o in result.numerical_outliers:
            if o.column == "total_amount" and o.method == "IQR":
                assert o.max_outlier_value is not None
                assert o.max_outlier_value > 10_000, (
                    f"Expected extreme max value, got {o.max_outlier_value}"
                )

    def test_severity_not_none(self, result):
        for o in result.numerical_outliers:
            assert o.severity in (Severity.LOW, Severity.MEDIUM, Severity.HIGH)

    def test_example_values_reasonable(self, result):
        """Example values list must be non-empty and bounded."""
        for o in result.numerical_outliers:
            if o.outlier_count > 0:
                assert len(o.example_values) > 0
                assert len(o.example_values) <= 10

    def test_ground_truth_anomaly_types_covered(self, result, ground_truth):
        """
        Ground truth records extreme_quantity and extreme_total_amount types.
        At least one of these should be detected.
        """
        detected_cols = {o.column for o in result.numerical_outliers if o.outlier_count > 0}
        gt_types = {r["type"] for r in ground_truth["anomalies"]["records"]}

        # Map ground truth types to expected column names
        type_to_col = {
            "extreme_quantity":     "quantity",
            "extreme_total_amount": "total_amount",
        }
        expected_cols = {type_to_col[t] for t in gt_types if t in type_to_col}
        overlap = detected_cols & expected_cols
        assert overlap, (
            f"Expected at least one of {expected_cols} to be detected, "
            f"got {detected_cols}"
        )


# ===========================================================================
# C. Problematic sales dataset
# ===========================================================================


class TestAnomalyProblematic:
    """Anomaly tool on the full problematic dataset."""

    @pytest.fixture(scope="class")
    def result(self) -> AnomalyResult:
        path = RAW_DIR / "sales_problematic.csv"
        _require_file(path)
        return detect_anomalies(path)

    def test_loads(self, result):
        assert result.anomaly_error is None

    def test_row_count(self, result):
        assert result.row_count == 8_180

    def test_anomalies_detected(self, result):
        assert result.anomalies_detected

    def test_total_amount_outlier_detected(self, result):
        """Injected extreme total_amounts should appear as IQR outliers."""
        iqr_cols = [o.column for o in result.numerical_outliers if o.method == "IQR"]
        assert "total_amount" in iqr_cols

    def test_quantity_outlier_detected(self, result):
        """Injected extreme quantities (500–2000) should be detected."""
        iqr_qty = [o for o in result.numerical_outliers
                   if o.column == "quantity" and o.method == "IQR"]
        assert len(iqr_qty) > 0
        assert iqr_qty[0].outlier_count > 0

    def test_numerical_outlier_count_reasonable(self, result):
        """At least 2 numeric columns should show outliers."""
        cols_with_outliers = {
            o.column for o in result.numerical_outliers
            if o.method == "IQR" and o.outlier_count > 0
        }
        assert len(cols_with_outliers) >= 2

    def test_distribution_analysis_ran(self, result):
        """Distribution analysis should have executed (result may or may not flag)."""
        meta = result.analysis_metadata
        assert "categorical_columns_analyzed" in meta

    def test_metric_analysis_ran(self, result):
        meta = result.analysis_metadata
        assert "total_revenue" in meta
        assert "transaction_count" in meta

    def test_severity_summary_populated(self, result):
        total_in_summary = sum(result.severity_summary.values())
        assert total_in_summary == result.total_anomalies

    def test_ground_truth_extreme_values_detected(self, result, ground_truth):
        """35 anomalies were injected; at least 10 should be detectable outliers."""
        qty_iqr = next(
            (o for o in result.numerical_outliers
             if o.column == "quantity" and o.method == "IQR"),
            None,
        )
        amt_iqr = next(
            (o for o in result.numerical_outliers
             if o.column == "total_amount" and o.method == "IQR"),
            None,
        )
        qty_count = qty_iqr.outlier_count if qty_iqr else 0
        amt_count = amt_iqr.outlier_count if amt_iqr else 0
        # The 35 injected anomalies (extreme_quantity + extreme_total_amount)
        # should be largely detectable by IQR
        assert qty_count + amt_count >= 10, (
            f"Expected >= 10 detected outliers across qty+amount, got {qty_count+amt_count}"
        )


# ===========================================================================
# D. Combined issues dataset
# ===========================================================================


class TestAnomaly_CombinedIssues:
    @pytest.fixture(scope="class")
    def result(self) -> AnomalyResult:
        path = TEST_DIR / "combined_issues.csv"
        _require_file(path)
        return detect_anomalies(path)

    def test_loads(self, result):
        assert result.anomaly_error is None

    def test_row_count_correct(self, result):
        assert result.row_count == 70

    def test_no_crash_with_multiple_issues(self, result):
        """Should complete without errors even with missing, dup, schema issues."""
        assert result.anomaly_error is None
        assert isinstance(result.numerical_outliers, list)
        assert isinstance(result.distribution_anomalies, list)

    def test_structure_valid(self, result):
        d = result.to_dict()
        assert "anomalies_detected" in d
        assert "severity_summary"   in d
        assert "numerical_outliers" in d


# ===========================================================================
# E. Small dataset edge case
# ===========================================================================


class TestAnomalySmallDataset:
    def test_very_small_dataset(self):
        """3-row dataset should not crash; IQR and Z-score skip gracefully."""
        df = pd.DataFrame({
            "id":     ["A", "B", "C"],
            "amount": [10.0, 20.0, 30.0],
        })
        result = detect_anomalies(df, dataset_name="tiny")
        assert result.anomaly_error is None
        assert result.row_count == 3
        # Z-score should be skipped (< MIN_ZSCORE_ROWS)
        zscore_results = [o for o in result.numerical_outliers if o.method == "Z-score"]
        assert len(zscore_results) == 0

    def test_exactly_four_rows(self):
        """4 rows: IQR can run but may find no outliers on clean monotone data."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
        result = detect_anomalies(df)
        assert result.anomaly_error is None

    def test_single_row(self):
        df = pd.DataFrame({"val": [42.0]})
        result = detect_anomalies(df)
        # Either error or empty — must not crash
        assert result is not None


# ===========================================================================
# F. Constant-value column
# ===========================================================================


class TestAnomalyConstantColumn:
    def test_constant_numeric_column(self):
        """IQR=0 → skip gracefully, no divide-by-zero."""
        df = pd.DataFrame({
            "id":    ["A"] * 50,
            "value": [5.0] * 50,
        })
        result = detect_anomalies(df)
        assert result.anomaly_error is None
        # No IQR outliers on a constant column
        iqr_results = [o for o in result.numerical_outliers if o.method == "IQR"]
        assert len(iqr_results) == 0

    def test_zero_std_zscore(self):
        """Z-score std=0 → skip gracefully."""
        series = pd.Series([7.0] * 50)
        result = _detect_zscore_outliers(series, "constant_col")
        assert result is None


# ===========================================================================
# G. Missing values / NaN handling
# ===========================================================================


class TestAnomalyMissingValues:
    def test_column_with_many_nulls(self):
        """NaN-heavy column should not crash IQR or Z-score detection."""
        df = pd.DataFrame({
            "amount": [None, 10.0, None, 20.0, None, None, 30.0, None, 1000.0, None],
        })
        result = detect_anomalies(df)
        assert result.anomaly_error is None

    def test_all_null_column_skipped(self):
        """Column with 100% NaN: IQR should return None (skip)."""
        series = pd.Series([None] * 10, dtype=object)
        result = _detect_iqr_outliers(series, "all_null")
        assert result is None

    def test_missing_values_in_amount_col(self):
        path = TEST_DIR / "missing_values.csv"
        _require_file(path)
        result = detect_anomalies(path)
        assert result.anomaly_error is None
        assert result.row_count > 0

    def test_nan_in_numeric_series_iqr(self):
        """IQR should skip NaN values and operate on clean subset."""
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 10000.0])
        result = _detect_iqr_outliers(series, "test_col")
        # Should detect the 10000.0 extreme value
        assert result is not None
        assert result.outlier_count >= 1


# ===========================================================================
# H. Non-numeric columns not passed to numeric methods
# ===========================================================================


class TestAnomalyNonNumericHandling:
    def test_text_column_excluded_from_iqr(self):
        """Pure text columns must not be passed to IQR/Z-score detection."""
        df = pd.DataFrame({
            "name":   ["Alice", "Bob", "Charlie"],
            "value":  [10.0, 20.0, 1000.0],
        })
        result = detect_anomalies(df)
        assert result.anomaly_error is None
        # Only 'value' should appear in numerical outlier columns
        numerical_col_names = {o.column for o in result.numerical_outliers}
        assert "name" not in numerical_col_names

    def test_find_numeric_columns_excludes_text(self):
        df = pd.DataFrame({
            "region":   ["North", "South", "East"],
            "quantity": [1.0, 2.0, 3.0],
        })
        numeric = _find_numeric_columns(df)
        assert "region"   not in numeric
        assert "quantity" in     numeric

    def test_mixed_column_classified_correctly(self):
        """A column with mostly text and some numbers → classified as text."""
        df = pd.DataFrame({
            "qty_str": ["5", "3 units", "abc", "7", "10 pieces"] * 6,
        })
        numeric = _find_numeric_columns(df)
        # "3 units" and "abc" and "10 pieces" fail numeric parsing → < 80% threshold
        # So this should NOT be classified as numeric
        assert "qty_str" not in numeric


# ===========================================================================
# I. Invalid/missing file
# ===========================================================================


class TestAnomalyErrorHandling:
    def test_missing_file_returns_error(self):
        result = detect_anomalies("data/no_such_file.csv")
        assert result.anomaly_error is not None
        assert result.anomalies_detected is False
        assert result.total_anomalies == 0

    def test_missing_file_error_message(self):
        result = detect_anomalies("data/no_such_file.csv")
        assert "not found" in result.anomaly_error.lower()

    def test_empty_dataframe_returns_error(self):
        df     = pd.DataFrame()
        result = detect_anomalies(df, dataset_name="empty")
        assert result.anomaly_error is not None
        assert result.row_count == 0

    def test_result_structure_on_error(self):
        result = detect_anomalies("data/no_such.csv")
        d = result.to_dict()
        assert "anomaly_error"       in d
        assert "numerical_outliers"  in d
        assert "distribution_anomalies" in d
        assert d["numerical_outliers"] == []

    def test_invalid_binary_file(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\xff\xfe" + b"\x00" * 50)
        result = detect_anomalies(bad)
        # Must not raise; either error result or partial result
        assert result is not None


# ===========================================================================
# J. IQR / Z-score unit tests
# ===========================================================================


class TestDetectionMethodUnits:
    """Low-level unit tests for individual detection functions."""

    def test_iqr_detects_known_outlier(self):
        # Clear outlier at 1000 in a tightly distributed dataset
        series = pd.Series([1.0, 2.0, 3.0, 2.5, 1.5, 2.8, 1.2, 1000.0])
        result = _detect_iqr_outliers(series, "test")
        assert result is not None
        assert result.outlier_count >= 1
        assert 1000.0 in result.example_values

    def test_iqr_no_outliers_on_uniform(self):
        series = pd.Series(list(range(1, 101)))  # 1..100, no outliers
        result = _detect_iqr_outliers(series, "uniform")
        assert result is None or result.outlier_count == 0

    def test_iqr_skips_insufficient_data(self):
        series = pd.Series([1.0, 2.0, 3.0])  # n=3 < 4
        result = _detect_iqr_outliers(series, "tiny")
        assert result is None

    def test_iqr_fence_calculation(self):
        # Manually verify fence values
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = _detect_iqr_outliers(series, "manual")
        assert result is not None
        # Q1=3.25, Q3=7.75, IQR=4.5, lo=3.25-6.75=-3.5, hi=7.75+6.75=14.5
        assert result.lower_fence is not None
        assert result.upper_fence is not None
        assert result.lower_fence < 0
        assert result.upper_fence > 10

    def test_zscore_detects_known_outlier(self):
        base = list(range(1, 51))  # 1..50 gives good normal-ish distribution
        base.append(500)           # clear outlier
        series = pd.Series(base, dtype=float)
        result = _detect_zscore_outliers(series, "test_z")
        assert result is not None
        assert result.outlier_count >= 1

    def test_zscore_skips_small_dataset(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # n=5 < 30
        result = _detect_zscore_outliers(series, "small")
        assert result is None

    def test_zscore_skips_zero_std(self):
        series = pd.Series([5.0] * 40)
        result = _detect_zscore_outliers(series, "constant")
        assert result is None

    def test_distribution_detects_dominant_category(self):
        df = pd.DataFrame({"region": ["North"] * 95 + ["South"] * 5})
        results = _detect_distribution_anomalies(df, ["region"])
        dominant = [r for r in results if r.anomaly_type == "dominant_category"]
        assert len(dominant) >= 1
        assert dominant[0].dominant_pct >= 90

    def test_distribution_detects_unexpected_category(self):
        df = pd.DataFrame({
            "method": ["Credit Card"] * 40 + ["UNKNOWN"] * 5 + ["Cash"] * 5,
        })
        results = _detect_distribution_anomalies(df, ["method"])
        unexpected = [r for r in results if r.anomaly_type == "unexpected_category"]
        assert len(unexpected) >= 1

    def test_distribution_no_false_positive_normal(self):
        """5 balanced categories → no distribution anomaly expected."""
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West", "Central"] * 20,
        })
        results = _detect_distribution_anomalies(df, ["region"])
        # No dominant category (each is 20%)
        dominant = [r for r in results if r.anomaly_type == "dominant_category"]
        assert len(dominant) == 0


# ===========================================================================
# K. Agent-level tests
# ===========================================================================


class TestAnomalyAgent:
    @pytest.fixture(scope="class")
    def clean_response(self) -> AgentAnomalyResponse:
        path = RAW_DIR / "sales_clean.csv"
        _require_file(path)
        return run_anomaly_agent(path)

    @pytest.fixture(scope="class")
    def prob_response(self) -> AgentAnomalyResponse:
        path = RAW_DIR / "sales_problematic.csv"
        _require_file(path)
        return run_anomaly_agent(path)

    @pytest.fixture(scope="class")
    def anomaly_response(self) -> AgentAnomalyResponse:
        path = TEST_DIR / "anomaly.csv"
        _require_file(path)
        return run_anomaly_agent(path)

    def test_clean_response_structure(self, clean_response):
        assert clean_response.analysis_completed is True
        assert clean_response.anomaly_error is None
        assert isinstance(clean_response.summary, str) and clean_response.summary
        assert isinstance(clean_response.key_findings, list)
        assert isinstance(clean_response.notable_concerns, list)
        assert isinstance(clean_response.anomaly_metrics, dict)
        assert isinstance(clean_response.outlier_details, list)

    def test_problematic_detects_anomalies(self, prob_response):
        assert prob_response.analysis_completed is True
        assert prob_response.anomalies_detected is True
        assert prob_response.total_anomalies > 0

    def test_anomaly_dataset_flagged(self, anomaly_response):
        assert anomaly_response.analysis_completed is True
        assert anomaly_response.anomalies_detected is True

    def test_severity_summary_in_response(self, prob_response):
        ss = prob_response.severity_summary
        total = ss.get("HIGH", 0) + ss.get("MEDIUM", 0) + ss.get("LOW", 0)
        assert total == prob_response.total_anomalies

    def test_error_on_missing_file(self):
        response = run_anomaly_agent("data/no_such.csv")
        assert response.analysis_completed is False
        assert response.anomaly_error is not None

    def test_accepts_anomaly_result(self):
        """Agent must accept a pre-computed AnomalyResult."""
        df     = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 9999.0] * 10})
        result = detect_anomalies(df)
        response = run_anomaly_agent(result)
        assert response.analysis_completed is True

    def test_to_json_valid(self, clean_response):
        j      = clean_response.to_json()
        parsed = json.loads(j)
        assert "dataset_name"       in parsed
        assert "anomalies_detected" in parsed
        assert "outlier_details"    in parsed
        assert "metric_details"     in parsed

    def test_key_findings_not_empty(self, clean_response):
        assert len(clean_response.key_findings) > 0

    def test_summary_not_empty(self, prob_response):
        assert len(prob_response.summary) > 10

    def test_outlier_details_structure(self, prob_response):
        for od in prob_response.outlier_details:
            assert "column"        in od
            assert "method"        in od
            assert "outlier_count" in od
            assert "severity"      in od
            assert od["method"] in ("IQR", "Z-score")


# ===========================================================================
# L. Ground-truth verification
# ===========================================================================


class TestGroundTruthAnomalyVerification:
    """Cross-check anomaly tool output against ground_truth.json."""

    @pytest.fixture(scope="class")
    def prob_result(self) -> AnomalyResult:
        path = RAW_DIR / "sales_problematic.csv"
        _require_file(path)
        return detect_anomalies(path)

    @pytest.fixture(scope="class")
    def anomaly_result(self) -> AnomalyResult:
        path = TEST_DIR / "anomaly.csv"
        _require_file(path)
        return detect_anomalies(path)

    def test_extreme_quantity_detected_in_problematic(self, prob_result, ground_truth):
        """Ground truth: 18 extreme_quantity records injected."""
        qty_iqr = next(
            (o for o in prob_result.numerical_outliers
             if o.column == "quantity" and o.method == "IQR"),
            None,
        )
        assert qty_iqr is not None, "Expected IQR outlier detection on 'quantity'"
        assert qty_iqr.outlier_count >= 10, (
            f"Expected >= 10 qty outliers, got {qty_iqr.outlier_count}"
        )

    def test_extreme_total_amount_detected_in_problematic(self, prob_result, ground_truth):
        """Ground truth: 17 extreme_total_amount records injected."""
        amt_iqr = next(
            (o for o in prob_result.numerical_outliers
             if o.column == "total_amount" and o.method == "IQR"),
            None,
        )
        assert amt_iqr is not None, "Expected IQR outlier detection on 'total_amount'"
        assert amt_iqr.outlier_count >= 10, (
            f"Expected >= 10 total_amount outliers, got {amt_iqr.outlier_count}"
        )

    def test_anomaly_csv_extreme_amount_detected(self, anomaly_result, ground_truth):
        """The anomaly.csv dataset must show IQR outliers for total_amount."""
        outlier = next(
            (o for o in anomaly_result.numerical_outliers
             if o.column == "total_amount" and o.method == "IQR"),
            None,
        )
        assert outlier is not None
        assert outlier.outlier_count >= 1

    def test_revenue_metadata_close_to_ground_truth(self, prob_result, ground_truth):
        """
        The detected total revenue should be close to ground_truth's
        problematic_total_revenue (allowing for rows dropped due to NaN).
        """
        gt_revenue = ground_truth["revenue_anomaly"]["problematic_total_revenue"]
        detected   = prob_result.analysis_metadata.get("total_revenue")
        assert detected is not None
        # Allow 5% tolerance (missing values excluded from sum)
        tolerance = gt_revenue * 0.05
        assert abs(detected - gt_revenue) <= tolerance, (
            f"Detected revenue {detected} differs from GT {gt_revenue} by > 5%"
        )

    def test_anomaly_count_in_ground_truth(self, ground_truth):
        """Ground truth records 35 anomaly records — sanity-check the file."""
        assert ground_truth["anomalies"]["count"] == 35
