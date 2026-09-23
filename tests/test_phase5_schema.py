"""
tests/test_phase5_schema.py
----------------------------
Phase 5 — Schema Analysis Agent & Tool Test Suite.

Tests scenarios A through L specified in requirements:
  A. Clean dataset (sales_clean.csv)
  B. Problematic dataset (sales_problematic.csv)
  C. schema_issue.csv
  D. combined_issues.csv
  E. Missing column
  F. Extra column
  G. Compatible numeric types
  H. Invalid dates
  I. Nullability violation
  J. Constraint violation
  K. Empty dataset
  L. Missing file / invalid input
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from app.config.schema_config import ColumnSchema, SchemaDefinition, get_sales_schema, load_schema
from app.tools.schema_tool import analyze_schema, SchemaAnalysisResult
from app.agents.schema_agent import run_schema_agent, AgentSchemaResponse

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "raw" / "sales_clean.csv"
PROBLEMATIC_PATH = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"
SCHEMA_ISSUE_PATH = PROJECT_ROOT / "data" / "test" / "schema_issue.csv"
COMBINED_ISSUES_PATH = PROJECT_ROOT / "data" / "test" / "combined_issues.csv"


# =============================================================================
# Test Configuration & Schema Loading
# =============================================================================

def test_sales_schema_loading():
    """Verify sales schema loads correctly from JSON and helper function."""
    schema = get_sales_schema()
    assert schema.dataset_name == "sales"
    assert len(schema.columns) == 10
    col_names = [c.name for c in schema.columns]
    assert "transaction_id" in col_names
    assert "total_amount" in col_names


# =============================================================================
# Scenario A: Clean Dataset
# =============================================================================

class TestScenarioACleanDataset:
    """A. Clean dataset must strictly match expected schema."""

    @pytest.fixture(scope="class")
    def clean_result(self):
        df = pd.read_csv(CLEAN_PATH)
        return analyze_schema(df, schema="sales", dataset_name="sales_clean.csv")

    def test_clean_schema_validity(self, clean_result):
        assert clean_result.schema_valid is True

    def test_clean_no_missing_or_unexpected(self, clean_result):
        assert len(clean_result.missing_columns) == 0
        assert len(clean_result.unexpected_columns) == 0

    def test_clean_no_type_or_format_issues(self, clean_result):
        assert len(clean_result.type_mismatches) == 0
        assert len(clean_result.format_issues) == 0

    def test_clean_no_nullability_or_constraint_violations(self, clean_result):
        assert len(clean_result.nullability_violations) == 0
        assert len(clean_result.constraint_violations) == 0


# =============================================================================
# Scenario B: Problematic Dataset
# =============================================================================

class TestScenarioBProblematicDataset:
    """B. Problematic dataset should trigger schema analysis issues."""

    @pytest.fixture(scope="class")
    def prob_result(self):
        df = pd.read_csv(PROBLEMATIC_PATH)
        return analyze_schema(df, schema="sales", dataset_name="sales_problematic.csv")

    def test_problematic_schema_invalid(self, prob_result):
        assert prob_result.schema_valid is False

    def test_problematic_detects_issues(self, prob_result):
        total_issues = prob_result.summary["total_issues"]
        assert total_issues > 0
        # Problematic sales has dirty quantity strings ('units'), negative total amounts, missing fields, format issues
        has_type_or_format_or_constraint = (
            len(prob_result.type_mismatches) > 0
            or len(prob_result.format_issues) > 0
            or len(prob_result.constraint_violations) > 0
            or len(prob_result.nullability_violations) > 0
        )
        assert has_type_or_format_or_constraint is True


# =============================================================================
# Scenario C: schema_issue.csv
# =============================================================================

class TestScenarioCSchemaIssueDataset:
    """C. schema_issue.csv contains known type/format issues."""

    @pytest.fixture(scope="class")
    def schema_issue_result(self):
        df = pd.read_csv(SCHEMA_ISSUE_PATH)
        return analyze_schema(df, schema="sales", dataset_name="schema_issue.csv")

    def test_schema_issue_detected(self, schema_issue_result):
        assert schema_issue_result.schema_valid is False
        assert len(schema_issue_result.type_mismatches) > 0 or len(schema_issue_result.format_issues) > 0
        # Quantity column contains strings like "3 units"
        type_mismatch_cols = [t["column"] for t in schema_issue_result.type_mismatches]
        assert "quantity" in type_mismatch_cols or "transaction_date" in [f["column"] for f in schema_issue_result.format_issues]


# =============================================================================
# Scenario D: combined_issues.csv
# =============================================================================

class TestScenarioDCombinedIssuesDataset:
    """D. combined_issues.csv should run without errors and report issues."""

    def test_combined_issues_analysis(self):
        df = pd.read_csv(COMBINED_ISSUES_PATH)
        res = analyze_schema(df, schema="sales", dataset_name="combined_issues.csv")
        assert res.dataset_name == "combined_issues.csv"
        assert isinstance(res.summary, dict)


# =============================================================================
# Scenario E: Missing Column Detection
# =============================================================================

def test_missing_column_detection():
    """E. Missing required column should be flagged with HIGH/CRITICAL severity."""
    df = pd.read_csv(CLEAN_PATH).drop(columns=["customer_id"])
    res = analyze_schema(df, schema="sales")
    assert res.schema_valid is False
    missing_names = [m["column"] for m in res.missing_columns]
    assert "customer_id" in missing_names


# =============================================================================
# Scenario F: Extra Column Detection
# =============================================================================

def test_extra_column_detection():
    """F. Extra column in actual dataset should be reported as LOW severity."""
    df = pd.read_csv(CLEAN_PATH)
    df["extra_discount_code"] = "PROMO2024"
    res = analyze_schema(df, schema="sales")
    unexpected_names = [u["column"] for u in res.unexpected_columns]
    assert "extra_discount_code" in unexpected_names
    for u in res.unexpected_columns:
        if u["column"] == "extra_discount_code":
            assert u["severity"] == "LOW"


# =============================================================================
# Scenario G: Compatible Numeric Types
# =============================================================================

def test_compatible_numeric_types():
    """G. Equivalent numeric representations (e.g. float64 with whole integers) should not flag false type mismatches."""
    df = pd.read_csv(CLEAN_PATH)
    # Cast quantity (int) to float64 (10.0, 5.0, etc.)
    df["quantity"] = df["quantity"].astype("float64")
    res = analyze_schema(df, schema="sales")
    type_mismatch_cols = [t["column"] for t in res.type_mismatches]
    assert "quantity" not in type_mismatch_cols


# =============================================================================
# Scenario H: Invalid Date Format Analysis
# =============================================================================

def test_invalid_dates_detection():
    """H. Invalid date values and format inconsistencies should be reported."""
    df = pd.read_csv(CLEAN_PATH)
    df.loc[0, "transaction_date"] = "not-a-valid-date"
    df.loc[1, "transaction_date"] = "05/26/2024"  # Non-ISO format
    res = analyze_schema(df, schema="sales")
    format_cols = [f["column"] for f in res.format_issues]
    assert "transaction_date" in format_cols
    date_finding = next(f for f in res.format_issues if f["column"] == "transaction_date")
    assert date_finding["evidence"]["invalid_count"] >= 1


# =============================================================================
# Scenario I: Nullability Violation Detection
# =============================================================================

def test_nullability_violation_detection():
    """I. Null values in non-nullable fields should be flagged."""
    df = pd.read_csv(CLEAN_PATH)
    df.loc[0:4, "transaction_id"] = None
    res = analyze_schema(df, schema="sales")
    null_cols = [n["column"] for n in res.nullability_violations]
    assert "transaction_id" in null_cols
    txn_null = next(n for n in res.nullability_violations if n["column"] == "transaction_id")
    assert txn_null["evidence"]["null_count"] == 5
    assert txn_null["severity"] == "CRITICAL"


# =============================================================================
# Scenario J: Constraint Violation Detection
# =============================================================================

def test_constraint_violation_detection():
    """J. Out-of-bounds numerical and categorical values should be detected."""
    df = pd.read_csv(CLEAN_PATH)
    df.loc[0, "quantity"] = -5  # Violates quantity > 0
    df.loc[1, "total_amount"] = -100.0  # Violates total_amount >= 0
    df.loc[2, "region"] = "InvalidRegion"  # Violates allowed values
    res = analyze_schema(df, schema="sales")
    viol_cols = [c["column"] for c in res.constraint_violations]
    assert "quantity" in viol_cols
    assert "total_amount" in viol_cols
    assert "region" in viol_cols


# =============================================================================
# Scenario K: Empty Dataset Handling
# =============================================================================

def test_empty_dataset_handling():
    """K. Empty DataFrame should be handled gracefully without crashing."""
    df = pd.read_csv(CLEAN_PATH).iloc[0:0]
    res = analyze_schema(df, schema="sales", dataset_name="empty.csv")
    assert res.total_rows == 0
    assert res.total_columns == 10
    assert res.schema_valid is True


# =============================================================================
# Scenario L: Missing File / Invalid Input Handling
# =============================================================================

def test_missing_file_invalid_input():
    """L. Missing file paths and invalid inputs raise appropriate errors."""
    with pytest.raises(FileNotFoundError):
        run_schema_agent("non_existent_file_xyz_123.csv")

    with pytest.raises(ValueError):
        run_schema_agent(12345)


# =============================================================================
# Agent & Gemini Integration Tests
# =============================================================================

def test_schema_agent_execution_clean():
    """Test full Schema Agent execution on clean dataset with fallback interpretation."""
    agent_resp = run_schema_agent(CLEAN_PATH, use_llm=False)
    assert isinstance(agent_resp, AgentSchemaResponse)
    assert agent_resp.schema_valid is True
    assert "SUMMARY:" in agent_resp.llm_interpretation
    assert agent_resp.llm_used is False


def test_schema_agent_execution_problematic():
    """Test full Schema Agent execution on problematic dataset."""
    agent_resp = run_schema_agent(PROBLEMATIC_PATH, use_llm=False)
    assert isinstance(agent_resp, AgentSchemaResponse)
    assert agent_resp.schema_valid is False
    assert "KEY FINDINGS:" in agent_resp.llm_interpretation
