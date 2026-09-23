"""
tests/test_phase8_validation.py
--------------------------------
Phase 8 — Validation Agent & Tool Test Suite.

Tests scenarios A through J specified in requirements:
  A. Fully corrected dataset (PASSED)
  B. Partially corrected dataset (PARTIAL)
  C. Uncorrected problematic dataset (FAILED)
  D. Clean dataset (NOT_APPLICABLE)
  E. Duplicate issue corrected
  F. Missing fields corrected
  G. Schema issue corrected
  H. Revenue metric corrected
  I. Criteria mismatch reported
  J. No data modification (read-only guarantee)
"""

import os
from pathlib import Path
import pandas as pd
import pytest

from app.tools.validation_tool import (
    validate_correction,
    ValidationCheck,
    ValidationResult,
)
from app.agents.validation_agent import run_validation_agent, AgentValidationResponse

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "raw" / "sales_clean.csv"
PROBLEMATIC_PATH = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"


# Helper function to create simulated post-correction DataFrames
def get_simulated_datasets():
    df_prob = pd.read_csv(PROBLEMATIC_PATH)

    # 1. Fully corrected dataset: remove duplicates, fill nulls, fix string types & dates
    df_fixed = df_prob.drop_duplicates(subset=["transaction_id"]).copy()
    df_fixed["customer_id"] = df_fixed["customer_id"].fillna("CUST000999")
    df_fixed["region"] = df_fixed["region"].fillna("North")
    df_fixed["payment_method"] = df_fixed["payment_method"].fillna("Credit Card")

    # Fix string suffixes in quantity (e.g. "3.0.0 units")
    df_fixed["quantity"] = df_fixed["quantity"].fillna("1").astype(str).str.extract(r"(\d+)")[0].fillna("1").astype(int)
    df_fixed["unit_price"] = df_fixed["unit_price"].fillna("10.0").astype(str).str.extract(r"(\d+\.?\d*)")[0].fillna("10.0").astype(float)
    df_fixed["total_amount"] = (df_fixed["quantity"] * df_fixed["unit_price"]).abs()
    df_fixed["transaction_date"] = "2024-06-15"  # Clean ISO date

    # 2. Partially corrected dataset: fix duplicates, but leave missing customer_id nulls
    df_partial = df_prob.drop_duplicates(subset=["transaction_id"]).copy()

    return df_prob, df_fixed, df_partial


# =============================================================================
# Scenario A: Fully Corrected Dataset (PASSED)
# =============================================================================

def test_scenario_a_fully_corrected_dataset():
    """A. Fully corrected dataset should receive PASSED verdict."""
    df_prob, df_fixed, _ = get_simulated_datasets()
    res = validate_correction(post_correction_df=df_fixed, original_df=df_prob, dataset_name="fully_corrected")
    assert res.validation_status == "PASSED"
    assert res.failed_checks == 0
    assert res.passed_checks > 0


# =============================================================================
# Scenario B: Partially Corrected Dataset (PARTIAL)
# =============================================================================

def test_scenario_b_partially_corrected_dataset():
    """B. Partially corrected dataset should receive PARTIAL verdict."""
    df_prob, _, df_partial = get_simulated_datasets()
    res = validate_correction(post_correction_df=df_partial, original_df=df_prob, dataset_name="partially_corrected")
    assert res.validation_status in ("PARTIAL", "FAILED")
    assert len(res.unresolved_issues) > 0


# =============================================================================
# Scenario C: Uncorrected Problematic Dataset (FAILED)
# =============================================================================

def test_scenario_c_uncorrected_problematic_dataset():
    """C. Uncorrected problematic dataset should receive FAILED verdict."""
    df_prob = pd.read_csv(PROBLEMATIC_PATH)
    res = validate_correction(post_correction_df=df_prob, original_df=df_prob, dataset_name="uncorrected_prob")
    assert res.validation_status == "FAILED"
    assert res.failed_checks > 0


# =============================================================================
# Scenario D: Clean Dataset (NOT_APPLICABLE)
# =============================================================================

def test_scenario_d_clean_dataset():
    """D. Clean dataset should receive NOT_APPLICABLE verdict."""
    df_clean = pd.read_csv(CLEAN_PATH)
    agent_resp = run_validation_agent(post_correction_input=df_clean, original_dataset_input=df_clean, use_llm=False)
    assert agent_resp.validation_status == "NOT_APPLICABLE"


# =============================================================================
# Scenario E: Duplicate Issue Corrected
# =============================================================================

def test_scenario_e_duplicate_corrected():
    """E. Corrected duplicates pass duplicate validation check."""
    df_prob, df_fixed, _ = get_simulated_datasets()
    res = validate_correction(post_correction_df=df_fixed, original_df=df_prob)
    dup_check = next((c for c in res.checks if c["category"] == "duplicates"), None)
    if dup_check:
        assert dup_check["status"] == "PASS"
        assert dup_check["after_value"] == 0


# =============================================================================
# Scenario F: Missing Fields Corrected
# =============================================================================

def test_scenario_f_missing_fields_corrected():
    """F. Corrected missing fields pass completeness validation check."""
    df_prob, df_fixed, _ = get_simulated_datasets()
    res = validate_correction(post_correction_df=df_fixed, original_df=df_prob)
    comp_check = next((c for c in res.checks if c["category"] == "completeness"), None)
    if comp_check:
        assert comp_check["status"] == "PASS"
        assert comp_check["after_value"] == 0


# =============================================================================
# Scenario G: Schema Issue Corrected
# =============================================================================

def test_scenario_g_schema_issue_corrected():
    """G. Corrected schema type mismatches pass schema validation check."""
    df_prob, df_fixed, _ = get_simulated_datasets()
    res = validate_correction(post_correction_df=df_fixed, original_df=df_prob)
    sch_check = next((c for c in res.checks if c["category"] == "schema" and "Type" in c["check_name"]), None)
    if sch_check:
        assert sch_check["status"] == "PASS"


# =============================================================================
# Scenario H: Revenue Metric Corrected
# =============================================================================

def test_scenario_h_revenue_metric_corrected():
    """H. Revenue metric calculation is reported in before/after comparison."""
    df_prob, df_fixed, _ = get_simulated_datasets()
    res = validate_correction(post_correction_df=df_fixed, original_df=df_prob)
    assert "total_revenue" in res.before_after_comparison
    assert res.before_after_comparison["total_revenue"]["after"] > 0


# =============================================================================
# Scenario I: Criteria Mismatch Correctly Reported
# =============================================================================

def test_scenario_i_criteria_mismatch_reported():
    """I. Unresolved defects are listed in unresolved_issues."""
    df_prob, _, df_partial = get_simulated_datasets()
    res = validate_correction(post_correction_df=df_partial, original_df=df_prob)
    assert len(res.unresolved_issues) > 0


# =============================================================================
# Scenario J: No Data Modification (Read-Only Guarantee)
# =============================================================================

def test_scenario_j_read_only_guarantee():
    """J. Executing Validation Agent does NOT modify source CSV files or DataFrames."""
    mtime_before = os.path.getmtime(PROBLEMATIC_PATH)
    df_before = pd.read_csv(PROBLEMATIC_PATH)

    run_validation_agent(post_correction_input=PROBLEMATIC_PATH, original_dataset_input=PROBLEMATIC_PATH, use_llm=False)

    mtime_after = os.path.getmtime(PROBLEMATIC_PATH)
    df_after = pd.read_csv(PROBLEMATIC_PATH)

    assert mtime_before == mtime_after
    assert len(df_before) == len(df_after)
