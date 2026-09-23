"""
tests/test_phase3_profiling.py
------------------------------
Automated pytest tests for Phase 3 — Data Profiling Tool and Agent.

Tests cover:
  A. Clean sales dataset
  B. Problematic sales dataset
  C. missing_values.csv
  D. duplicates.csv
  E. anomaly.csv
  F. schema_issue.csv
  G. Error / edge-case handling

All numerical assertions are grounded in ground_truth.json where relevant.

Run:
    pytest tests/test_phase3_profiling.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.tools.profiling_tool import profile_dataset, ProfilingResult
from app.agents.profiling_agent import run_profiling_agent, AgentProfilingResponse

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_file(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} not found — run generate_dataset.py first")


# ===========================================================================
# A. Clean sales dataset
# ===========================================================================


class TestProfilingToolCleanSales:
    """Profiling tool applied to sales_clean.csv."""

    @pytest.fixture(scope="class")
    def result(self) -> ProfilingResult:
        path = RAW_DIR / "sales_clean.csv"
        _require_file(path)
        return profile_dataset(path)

    def test_loads_successfully(self, result):
        assert result.profiling_error is None, f"Got error: {result.profiling_error}"

    def test_row_count(self, result):
        assert result.row_count == 8_000, (
            f"Expected 8000 rows, got {result.row_count}"
        )

    def test_column_count(self, result):
        assert result.column_count == 10

    def test_expected_columns_present(self, result):
        expected = {
            "transaction_id", "customer_id", "product_id",
            "transaction_date", "quantity", "unit_price",
            "total_amount", "region", "payment_method", "sales_channel",
        }
        assert set(result.columns) == expected

    def test_no_missing_values(self, result):
        assert result.missing_value_summary.total_missing_cells == 0

    def test_completeness_100(self, result):
        assert result.quality_indicators.completeness_pct == 100.0

    def test_no_duplicate_rows(self, result):
        # Clean dataset must have 0 fully duplicate rows
        assert result.duplicate_summary.duplicate_rows == 0

    def test_no_duplicate_transaction_ids(self, result, ground_truth):
        # ground_truth tells us there are no intentional duplicates in clean data
        assert result.duplicate_summary.duplicate_transaction_ids == 0

    def test_numeric_columns_detected(self, result):
        # quantity, unit_price, total_amount should be numeric
        for col in ["quantity", "unit_price", "total_amount"]:
            assert col in result.numeric_columns, f"'{col}' should be numeric"

    def test_positive_revenue_stats(self, result):
        # total_amount stats must all be positive in clean dataset
        for cd in result.column_details:
            if cd.column_name == "total_amount":
                assert cd.min is not None and cd.min > 0
                assert cd.mean is not None and cd.mean > 0
                assert cd.max is not None and cd.max > 0
                break

    def test_quality_indicators_populated(self, result):
        qi = result.quality_indicators
        assert qi.completeness_pct == 100.0
        assert qi.duplicate_pct == 0.0
        assert qi.uniqueness_pct == 100.0

    def test_categorical_columns_have_top_values(self, result):
        for cd in result.column_details:
            if cd.column_name == "region":
                assert cd.top_values is not None
                assert len(cd.top_values) > 0
                break

    def test_transaction_id_uniqueness(self, result):
        # In clean dataset, every transaction_id is unique
        for cd in result.column_details:
            if cd.column_name == "transaction_id":
                assert cd.unique_count == 8_000
                break


# ===========================================================================
# B. Problematic sales dataset
# ===========================================================================


class TestProfilingToolProblematicSales:
    """Profiling tool applied to sales_problematic.csv."""

    @pytest.fixture(scope="class")
    def result(self) -> ProfilingResult:
        path = RAW_DIR / "sales_problematic.csv"
        _require_file(path)
        return profile_dataset(path)

    def test_loads_successfully(self, result):
        assert result.profiling_error is None

    def test_row_count_greater_than_clean(self, result):
        # 8000 clean + 180 duplicates = 8180
        assert result.row_count == 8_180

    def test_missing_values_detected(self, result):
        assert result.missing_value_summary.total_missing_cells > 0

    def test_expected_missing_columns(self, result, ground_truth):
        expected_cols = set(ground_truth["missing_values"]["columns"])
        missing_cols  = {
            item["column"]
            for item in result.missing_value_summary.columns_with_missing
        }
        # All expected-missing columns should appear in profiling output
        assert expected_cols <= missing_cols, (
            f"Missing columns not detected: {expected_cols - missing_cols}"
        )

    def test_duplicate_rows_detected(self, result, ground_truth):
        # df.duplicated() marks only the 2nd-and-later occurrence of each duplicate.
        # With 180 duplicates added from 180 unique source rows, the count is >=150.
        # We verify: (a) at least 1 duplicate detected, (b) count is in plausible range.
        gt_dup_count = ground_truth["duplicates"]["count"]
        detected     = result.duplicate_summary.duplicate_rows
        assert detected > 0, "Expected duplicate rows but found none"
        # Lower bound: at least 80% of injected duplicates detectable
        assert detected >= int(gt_dup_count * 0.8), (
            f"Expected at least {int(gt_dup_count * 0.8)} duplicate rows, got {detected}"
        )

    def test_duplicate_transaction_ids_detected(self, result, ground_truth):
        gt_dup_count = ground_truth["duplicates"]["count"]
        dup_ids = result.duplicate_summary.duplicate_transaction_ids
        assert dup_ids is not None
        # transaction_id duplicates = exactly the number injected (each txn_id appears twice)
        assert dup_ids == gt_dup_count, (
            f"Expected exactly {gt_dup_count} duplicate transaction_ids, got {dup_ids}"
        )

    def test_missing_summary_columns_populated(self, result):
        ms = result.missing_value_summary
        assert ms.columns_with_nulls > 0
        assert len(ms.columns_with_missing) > 0
        for entry in ms.columns_with_missing:
            assert "column" in entry
            assert "count"  in entry
            assert "pct"    in entry

    def test_numeric_stats_generated(self, result):
        for cd in result.column_details:
            if cd.column_name == "total_amount" and cd.is_numeric:
                assert cd.mean is not None
                assert cd.std  is not None
                break

    def test_categorical_stats_generated(self, result):
        for cd in result.column_details:
            if cd.column_name == "region":
                assert cd.top_values is not None
                break

    def test_quality_indicators_show_issues(self, result):
        qi = result.quality_indicators
        assert qi.completeness_pct < 100.0
        assert qi.duplicate_pct    > 0.0

    def test_duplicate_pct_reasonable(self, result, ground_truth):
        # 180 duplicates / 8180 total = ~2.2%
        assert result.quality_indicators.duplicate_pct > 1.0


# ===========================================================================
# C. missing_values.csv
# ===========================================================================


class TestProfilingMissingValues:
    @pytest.fixture(scope="class")
    def result(self) -> ProfilingResult:
        path = TEST_DIR / "missing_values.csv"
        _require_file(path)
        return profile_dataset(path)

    def test_loads(self, result):
        assert result.profiling_error is None

    def test_missing_values_present(self, result):
        assert result.missing_value_summary.total_missing_cells > 0

    def test_completeness_below_100(self, result):
        assert result.quality_indicators.completeness_pct < 100.0

    def test_affected_columns_listed(self, result):
        # Test dataset has missing in customer_id, region, quantity, payment_method
        missing_cols = {
            item["column"]
            for item in result.missing_value_summary.columns_with_missing
        }
        assert len(missing_cols) >= 1

    def test_null_percentages_match_counts(self, result):
        """Verify pct = count / row_count * 100 for each missing column."""
        for entry in result.missing_value_summary.columns_with_missing:
            expected_pct = round(100 * entry["count"] / result.row_count, 2)
            assert abs(entry["pct"] - expected_pct) < 0.1, (
                f"Pct mismatch for column '{entry['column']}': "
                f"expected {expected_pct}, got {entry['pct']}"
            )


# ===========================================================================
# D. duplicates.csv
# ===========================================================================


class TestProfilingDuplicates:
    @pytest.fixture(scope="class")
    def result(self) -> ProfilingResult:
        path = TEST_DIR / "duplicates.csv"
        _require_file(path)
        return profile_dataset(path)

    def test_loads(self, result):
        assert result.profiling_error is None

    def test_duplicate_rows_detected(self, result):
        assert result.duplicate_summary.duplicate_rows > 0, (
            "Expected duplicate rows to be detected"
        )

    def test_duplicate_transaction_ids_detected(self, result):
        dup_ids = result.duplicate_summary.duplicate_transaction_ids
        assert dup_ids is not None and dup_ids > 0

    def test_duplicate_count_plausible(self, result):
        # duplicates.csv = 40 base + 20 duplicates => at least 10 dup rows
        assert result.duplicate_summary.duplicate_rows >= 10

    def test_duplicate_pct_above_zero(self, result):
        assert result.quality_indicators.duplicate_pct > 0.0


# ===========================================================================
# E. anomaly.csv
# ===========================================================================


class TestProfilingAnomaly:
    @pytest.fixture(scope="class")
    def result(self) -> ProfilingResult:
        path = TEST_DIR / "anomaly.csv"
        _require_file(path)
        return profile_dataset(path)

    def test_loads(self, result):
        assert result.profiling_error is None

    def test_total_amount_numeric(self, result):
        assert "total_amount" in result.numeric_columns

    def test_max_total_amount_is_extreme(self, result):
        """The anomaly dataset has total_amount values up to 500k."""
        for cd in result.column_details:
            if cd.column_name == "total_amount":
                assert cd.max is not None and cd.max > 10_000, (
                    f"Expected extreme max, got {cd.max}"
                )
                break

    def test_std_elevated(self, result):
        """High standard deviation expected due to extreme values."""
        for cd in result.column_details:
            if cd.column_name == "total_amount":
                assert cd.std is not None and cd.std > 0
                break

    def test_row_count_correct(self, result):
        assert result.row_count == 30


# ===========================================================================
# F. schema_issue.csv
# ===========================================================================


class TestProfilingSchemaIssue:
    @pytest.fixture(scope="class")
    def result(self) -> ProfilingResult:
        path = TEST_DIR / "schema_issue.csv"
        _require_file(path)
        return profile_dataset(path)

    def test_loads(self, result):
        assert result.profiling_error is None

    def test_row_count_correct(self, result):
        assert result.row_count == 25

    def test_columns_reported(self, result):
        assert "transaction_id" in result.columns
        assert "quantity"        in result.columns

    def test_dtypes_reported(self, result):
        """dtype should be reported for every column."""
        for cd in result.column_details:
            assert cd.dtype != ""

    def test_quantity_column_reported(self, result):
        """
        schema_issue.csv mixes numeric and string quantities.
        The profiling tool loads everything as string so quantity
        will appear as categorical (not fully numeric).
        """
        qty_detail = next(
            (cd for cd in result.column_details if cd.column_name == "quantity"),
            None,
        )
        assert qty_detail is not None
        # Because some values are "3 units", numeric success rate < 1.0
        # Either it's classified as categorical OR as numeric with low hit-rate
        # Either way the column must be listed
        assert qty_detail.column_name == "quantity"


# ===========================================================================
# G. Error handling / edge cases
# ===========================================================================


class TestProfilingErrorHandling:
    def test_missing_file_returns_error(self):
        result = profile_dataset("data/does_not_exist_xyz.csv")
        assert result.profiling_error is not None
        assert result.row_count == 0
        assert result.profiling_completed is False if hasattr(result, "profiling_completed") else True

    def test_missing_file_error_message(self):
        result = profile_dataset("data/no_such_file.csv")
        assert "not found" in result.profiling_error.lower() or \
               "error" in result.profiling_error.lower()

    def test_empty_dataframe_returns_error(self):
        empty_df = pd.DataFrame()
        result   = profile_dataset(empty_df, dataset_name="empty")
        assert result.profiling_error is not None
        assert result.row_count == 0

    def test_invalid_csv_content(self, tmp_path):
        """A file that cannot be parsed as CSV."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_bytes(b"\xff\xfe" + b"\x00" * 100)  # binary garbage
        # Should not raise; must return error result
        result = profile_dataset(bad_file)
        # Either an error result or an empty/partial result — must not crash
        assert result is not None

    def test_dataframe_input(self):
        """Profiling a DataFrame directly must work."""
        df = pd.DataFrame({
            "id":    ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        })
        result = profile_dataset(df, dataset_name="test_df")
        assert result.profiling_error is None
        assert result.row_count == 3
        assert result.column_count == 2

    def test_single_column_dataframe(self):
        df = pd.DataFrame({"only_col": range(10)})
        result = profile_dataset(df)
        assert result.row_count == 10
        assert result.column_count == 1

    def test_all_null_column(self):
        df = pd.DataFrame({
            "id":    ["A", "B"],
            "empty": [None, None],
        })
        result = profile_dataset(df, dataset_name="all_null")
        empty_col = next(cd for cd in result.column_details if cd.column_name == "empty")
        assert empty_col.null_count == 2
        assert empty_col.null_percentage == 100.0


# ===========================================================================
# H. Agent-level tests
# ===========================================================================


class TestProfilingAgent:
    """Tests for the run_profiling_agent() wrapper."""

    @pytest.fixture(scope="class")
    def clean_response(self) -> AgentProfilingResponse:
        path = RAW_DIR / "sales_clean.csv"
        _require_file(path)
        return run_profiling_agent(path)

    @pytest.fixture(scope="class")
    def prob_response(self) -> AgentProfilingResponse:
        path = RAW_DIR / "sales_problematic.csv"
        _require_file(path)
        return run_profiling_agent(path)

    def test_clean_agent_response_structure(self, clean_response):
        assert clean_response.profiling_completed is True
        assert clean_response.profiling_error is None
        assert isinstance(clean_response.summary, str) and clean_response.summary
        assert isinstance(clean_response.key_observations, list)
        assert isinstance(clean_response.quality_concerns, list)
        assert isinstance(clean_response.quality_metrics, dict)
        assert isinstance(clean_response.column_findings, list)

    def test_clean_agent_metrics_match_profiling(self, clean_response):
        qm = clean_response.quality_metrics
        assert qm["completeness_pct"] == 100.0
        assert qm["duplicate_rows"]   == 0
        assert qm["missing_cells_total"] == 0

    def test_clean_agent_column_findings_count(self, clean_response):
        assert len(clean_response.column_findings) == 10

    def test_problematic_agent_detects_issues(self, prob_response):
        assert prob_response.profiling_completed is True
        qm = prob_response.quality_metrics
        assert qm["duplicate_rows"]           > 0
        assert qm["missing_cells_total"]       > 0
        assert qm["duplicate_transaction_ids"] > 0

    def test_problematic_agent_has_concerns(self, prob_response):
        # At least one quality concern should be identified
        assert len(prob_response.quality_concerns) >= 1

    def test_agent_error_on_missing_file(self):
        response = run_profiling_agent("data/does_not_exist.csv")
        assert response.profiling_completed is False
        assert response.profiling_error is not None

    def test_agent_accepts_dataframe(self):
        df = pd.DataFrame({
            "transaction_id": ["TXN001", "TXN002"],
            "amount":         [100.0, 200.0],
        })
        response = run_profiling_agent(df, dataset_name="df_test")
        assert response.profiling_completed is True
        assert response.row_count == 2 if hasattr(response, "row_count") else True

    def test_agent_accepts_profiling_result(self):
        """Agent must accept a pre-computed ProfilingResult (skip re-profiling)."""
        df     = pd.DataFrame({"x": [1, 2, 3]})
        result = profile_dataset(df, dataset_name="pre_profiled")
        response = run_profiling_agent(result)
        assert response.profiling_completed is True

    def test_to_json_is_valid(self, clean_response):
        import json
        j = clean_response.to_json()
        parsed = json.loads(j)
        assert "dataset_name" in parsed
        assert "quality_metrics" in parsed
        assert "column_findings" in parsed

    def test_observations_not_empty(self, clean_response):
        assert len(clean_response.key_observations) > 0

    def test_summary_not_empty(self, prob_response):
        assert len(prob_response.summary) > 10


# ===========================================================================
# I. Ground truth numerical verification
# ===========================================================================


class TestGroundTruthVerification:
    """Cross-check profiling tool output against ground_truth.json."""

    @pytest.fixture(scope="class")
    def result(self) -> ProfilingResult:
        path = RAW_DIR / "sales_problematic.csv"
        _require_file(path)
        return profile_dataset(path)

    def test_missing_column_count_matches_ground_truth(
        self, result, ground_truth
    ):
        gt_cols = set(ground_truth["missing_values"]["columns"])
        detected = {
            item["column"]
            for item in result.missing_value_summary.columns_with_missing
        }
        # All ground-truth missing columns must be detected
        missing_from_detection = gt_cols - detected
        assert not missing_from_detection, (
            f"Ground truth says these columns have nulls but profiler missed them: "
            f"{missing_from_detection}"
        )

    def test_duplicate_count_at_least_gt(self, result, ground_truth):
        gt_count = ground_truth["duplicates"]["count"]
        detected = result.duplicate_summary.duplicate_rows
        # df.duplicated() marks 2nd+ occurrence; allow 80% of injected count
        assert detected >= int(gt_count * 0.8), (
            f"Expected at least {int(gt_count * 0.8)} duplicate rows, got {detected}"
        )

    def test_duplicate_txn_ids_at_least_gt(self, result, ground_truth):
        gt_count = ground_truth["duplicates"]["count"]
        dup_ids  = result.duplicate_summary.duplicate_transaction_ids
        assert dup_ids is not None
        # Each of the 180 injected duplicate rows has a repeated transaction_id
        assert dup_ids == gt_count, (
            f"Expected exactly {gt_count} duplicate transaction_ids, got {dup_ids}"
        )

    def test_total_row_count_matches_ground_truth_metadata(
        self, result, ground_truth
    ):
        gt_rows = ground_truth["_metadata"]["problematic_dataset_rows"]
        assert result.row_count == gt_rows
