"""
tests/test_phase7_correction.py
--------------------------------
Phase 7 — Correction Recommendation Agent & Tool Test Suite.

Tests scenarios A through J specified in requirements:
  A. Duplicate root cause
  B. Missing-data root cause
  C. Type-conversion root cause
  D. Date-format root cause
  E. Domain violation
  F. Multiple root causes
  G. Clean dataset (sales_clean.csv)
  H. Undetermined root cause
  I. Human approval
  J. No data modification
"""

import os
from pathlib import Path
import pandas as pd
import pytest

from app.tools.root_cause_tool import RootCauseResult
from app.tools.correction_tool import (
    generate_correction_plan,
    CorrectionRecommendation,
    CorrectionPlanResult,
)
from app.agents.correction_agent import run_correction_agent, AgentCorrectionResponse

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "raw" / "sales_clean.csv"
PROBLEMATIC_PATH = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"
COMBINED_ISSUES_PATH = PROJECT_ROOT / "data" / "test" / "combined_issues.csv"


# =============================================================================
# Scenario A: Duplicate Root Cause Recommendation
# =============================================================================

def test_scenario_a_duplicate_recommendation():
    """A. Duplicate root cause generates duplicate remediation recommendation."""
    rc_res = RootCauseResult(
        dataset_name="dup_test",
        investigation_status="completed",
        primary_root_cause={
            "cause": "Duplicate transaction ingestion in data pipeline",
            "confidence": "HIGH",
            "evidence_count": 2,
            "supporting_evidence": [{"description": "180 duplicate records"}],
        },
    )
    plan = generate_correction_plan(rc_res, dataset_name="dup_test")
    assert plan.status == "recommendations_generated"
    assert plan.total_recommendations >= 1
    rec = plan.recommendations[0]
    assert rec["human_approval_required"] is True
    assert any("Quarantine" in s or "quarantine" in s for s in rec["action_steps"])


# =============================================================================
# Scenario B: Missing-data Root Cause Recommendation
# =============================================================================

def test_scenario_b_missing_data_recommendation():
    """B. Missing data root cause generates missing-data remediation recommendation."""
    rc_res = RootCauseResult(
        dataset_name="missing_test",
        investigation_status="completed",
        primary_root_cause={
            "cause": "Upstream data loss or missing required fields during ingestion",
            "confidence": "HIGH",
            "supporting_evidence": [{"column": "customer_id", "description": "320 nulls"}],
        },
    )
    plan = generate_correction_plan(rc_res, dataset_name="missing_test")
    assert plan.total_recommendations >= 1
    rec = plan.recommendations[0]
    assert rec["human_approval_required"] is True
    assert "customer_id" in rec["affected_columns"]


# =============================================================================
# Scenario C: Type-conversion Root Cause Recommendation
# =============================================================================

def test_scenario_c_type_conversion_recommendation():
    """C. Type-conversion root cause generates type parsing recommendation."""
    rc_res = RootCauseResult(
        dataset_name="type_test",
        investigation_status="completed",
        primary_root_cause={
            "cause": "Data transformation and type-conversion parsing failure",
            "confidence": "HIGH",
            "supporting_evidence": [{"column": "quantity", "description": "Type mismatch on quantity"}],
        },
    )
    plan = generate_correction_plan(rc_res, dataset_name="type_test")
    assert plan.total_recommendations >= 1
    rec = plan.recommendations[0]
    assert rec["human_approval_required"] is True
    assert "quantity" in rec["affected_columns"]


# =============================================================================
# Scenario D: Date-format Root Cause Recommendation
# =============================================================================

def test_scenario_d_date_format_recommendation():
    """D. Date format root cause generates date normalization recommendation."""
    rc_res = RootCauseResult(
        dataset_name="date_test",
        investigation_status="completed",
        primary_root_cause={
            "cause": "Date parsing and timestamp format inconsistency",
            "confidence": "MEDIUM",
            "supporting_evidence": [{"column": "transaction_date", "description": "Format inconsistency"}],
        },
    )
    plan = generate_correction_plan(rc_res, dataset_name="date_test")
    assert plan.total_recommendations >= 1
    rec = plan.recommendations[0]
    assert "transaction_date" in rec["affected_columns"]


# =============================================================================
# Scenario E: Domain Violation Recommendation
# =============================================================================

def test_scenario_e_domain_violation_recommendation():
    """E. Categorical domain root cause generates domain mapping recommendation."""
    rc_res = RootCauseResult(
        dataset_name="domain_test",
        investigation_status="completed",
        primary_root_cause={
            "cause": "Invalid categorical values or domain entry error",
            "confidence": "MEDIUM",
            "supporting_evidence": [{"column": "region", "description": "Domain constraint violation"}],
        },
    )
    plan = generate_correction_plan(rc_res, dataset_name="domain_test")
    assert plan.total_recommendations >= 1
    rec = plan.recommendations[0]
    assert "region" in rec["affected_columns"]


# =============================================================================
# Scenario F: Multiple Root Causes
# =============================================================================

def test_scenario_f_multiple_root_causes():
    """F. Multiple root causes generate separate recommendations."""
    rc_res = RootCauseResult(
        dataset_name="multi_test",
        investigation_status="completed",
        primary_root_cause={
            "cause": "Duplicate transaction ingestion in data pipeline",
            "confidence": "HIGH",
        },
        contributing_causes=[
            {
                "cause": "Upstream data loss or missing required fields during ingestion",
                "confidence": "HIGH",
                "supporting_evidence": [{"column": "region"}],
            }
        ],
    )
    plan = generate_correction_plan(rc_res, dataset_name="multi_test")
    assert plan.total_recommendations == 2
    causes_in_recs = [r["root_cause"] for r in plan.recommendations]
    assert "Duplicate transaction ingestion in data pipeline" in causes_in_recs
    assert "Upstream data loss or missing required fields during ingestion" in causes_in_recs


# =============================================================================
# Scenario G: Clean Dataset (sales_clean.csv)
# =============================================================================

def test_scenario_g_clean_dataset():
    """G. Clean dataset returns 'no_action_required'."""
    agent_resp = run_correction_agent(CLEAN_PATH, use_llm=False)
    assert agent_resp.status == "no_action_required"
    assert agent_resp.total_recommendations == 0
    assert "No corrective action required" in agent_resp.summary


# =============================================================================
# Scenario H: Undetermined Root Cause
# =============================================================================

def test_scenario_h_undetermined_root_cause():
    """H. Undetermined root cause recommends investigation/escalation without fabricating data fixes."""
    rc_res = RootCauseResult(
        dataset_name="undetermined_test",
        investigation_status="inconclusive",
        primary_root_cause={"cause": "Undetermined", "confidence": "LOW"},
    )
    plan = generate_correction_plan(rc_res, dataset_name="undetermined_test")
    assert plan.status == "escalation_required"
    assert plan.total_recommendations == 1
    rec = plan.recommendations[0]
    assert rec["human_approval_required"] is False  # Safe analysis action
    assert "Collect additional" in rec["action_steps"][0]


# =============================================================================
# Scenario I: Human Approval Enforcement
# =============================================================================

def test_scenario_i_human_approval_enforcement():
    """I. Data-altering or quarantining recommendations require human approval."""
    agent_resp = run_correction_agent(PROBLEMATIC_PATH, use_llm=False)
    assert agent_resp.status == "recommendations_generated"
    assert agent_resp.human_approval_summary["required"] >= 1
    for rec in agent_resp.recommendations:
        if any("Quarantine" in s or "quarantine" in s or "Reprocess" in s for s in rec["action_steps"]):
            assert rec["human_approval_required"] is True


# =============================================================================
# Scenario J: No Data Modification Guarantee
# =============================================================================

def test_scenario_j_no_data_modification():
    """J. Executing Correction Agent does NOT modify source dataset files."""
    mtime_before = os.path.getmtime(PROBLEMATIC_PATH)
    df_before = pd.read_csv(PROBLEMATIC_PATH)

    agent_resp = run_correction_agent(PROBLEMATIC_PATH, use_llm=False)

    mtime_after = os.path.getmtime(PROBLEMATIC_PATH)
    df_after = pd.read_csv(PROBLEMATIC_PATH)

    assert mtime_before == mtime_after
    assert len(df_before) == len(df_after)
    assert list(df_before.columns) == list(df_after.columns)
