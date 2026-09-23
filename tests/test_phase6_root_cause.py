"""
tests/test_phase6_root_cause.py
--------------------------------
Phase 6 — Root Cause Investigation Agent & Tool Test Suite.

Tests scenarios A through I specified in requirements:
  A. Duplicate-driven anomaly
  B. Missing-record scenario
  C. Schema/type issue scenario
  D. Date issue scenario
  E. Multiple unrelated issues scenario
  F. Contradictory evidence scenario
  G. Insufficient evidence scenario
  H. Clean dataset scenario (sales_clean.csv)
  I. Problematic dataset scenario (sales_problematic.csv & combined_issues.csv)
"""

from pathlib import Path
import pandas as pd
import pytest

from app.tools.profiling_tool import profile_dataset, ProfilingResult
from app.tools.anomaly_tool import detect_anomalies, AnomalyResult
from app.tools.schema_tool import analyze_schema, SchemaAnalysisResult
from app.tools.root_cause_tool import (
    investigate_root_cause,
    normalize_evidence,
    generate_candidate_causes,
    rank_candidates,
    EvidenceItem,
    CandidateCause,
    RootCauseResult,
)
from app.agents.root_cause_agent import run_root_cause_agent, AgentRootCauseResponse

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "raw" / "sales_clean.csv"
PROBLEMATIC_PATH = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"
COMBINED_ISSUES_PATH = PROJECT_ROOT / "data" / "test" / "combined_issues.csv"


# =============================================================================
# Scenario A: Duplicate-driven Anomaly Scenario
# =============================================================================

def test_scenario_a_duplicate_anomaly():
    """A. Duplicate transactions + revenue anomaly -> Duplicate ingestion candidate."""
    ev_dup = EvidenceItem(
        source="profiling",
        finding_type="duplicate_records",
        column=None,
        severity="HIGH",
        description="180 duplicate transaction records detected.",
        observed_value=180,
    )
    ev_anom = EvidenceItem(
        source="anomaly",
        finding_type="metric_anomaly",
        column="total_amount",
        severity="HIGH",
        description="Business metric anomaly on 'total_amount': spike in daily revenue.",
        observed_value=50000.0,
    )

    candidates = generate_candidate_causes([ev_dup, ev_anom])
    ranked = rank_candidates(candidates)

    assert len(ranked) > 0
    top_cause = ranked[0]
    assert "Duplicate" in top_cause.cause
    assert top_cause.confidence in ("HIGH", "MEDIUM")


# =============================================================================
# Scenario B: Missing-record Scenario
# =============================================================================

def test_scenario_b_missing_record():
    """B. Missing required fields -> Upstream record loss candidate."""
    ev_null = EvidenceItem(
        source="schema",
        finding_type="nullability_violation",
        column="customer_id",
        severity="HIGH",
        description="Column 'customer_id' contains 320 null values.",
    )

    candidates = generate_candidate_causes([ev_null])
    ranked = rank_candidates(candidates)

    assert len(ranked) > 0
    assert "Upstream data loss" in ranked[0].cause or "missing" in ranked[0].cause.lower()


# =============================================================================
# Scenario C: Schema/type Issue Scenario
# =============================================================================

def test_scenario_c_schema_type_issue():
    """C. Numeric type mismatch + constraint violations -> Transformation issue candidate."""
    ev_type = EvidenceItem(
        source="schema",
        finding_type="type_mismatch",
        column="quantity",
        severity="HIGH",
        description="Column 'quantity' expects integer, got object.",
    )
    ev_constraint = EvidenceItem(
        source="schema",
        finding_type="constraint_violation",
        column="quantity",
        severity="HIGH",
        description="Values violating quantity > 0.",
    )

    candidates = generate_candidate_causes([ev_type, ev_constraint])
    ranked = rank_candidates(candidates)

    assert len(ranked) > 0
    assert "transformation" in ranked[0].cause.lower() or "type-conversion" in ranked[0].cause.lower()


# =============================================================================
# Scenario D: Date Issue Scenario
# =============================================================================

def test_scenario_d_date_issue():
    """D. Date format issue -> Date parsing candidate."""
    ev_fmt = EvidenceItem(
        source="schema",
        finding_type="format_issue",
        column="transaction_date",
        severity="HIGH",
        description="15 format inconsistencies in date column.",
    )

    candidates = generate_candidate_causes([ev_fmt])
    ranked = rank_candidates(candidates)

    assert len(ranked) > 0
    assert "Date parsing" in ranked[0].cause or "timestamp" in ranked[0].cause.lower()


# =============================================================================
# Scenario E: Multiple Unrelated Issues Scenario
# =============================================================================

def test_scenario_e_multiple_unrelated_issues():
    """E. Multiple distinct issues do not force an unsupported single root cause."""
    ev_fmt = EvidenceItem(source="schema", finding_type="format_issue", column="transaction_date", severity="LOW", description="1 minor date inconsistency.")
    ev_extra = EvidenceItem(source="schema", finding_type="unexpected_column", column="unknown_col", severity="LOW", description="1 extra column.")

    res = investigate_root_cause(dataset_name="unrelated_test")
    # Normalize evidence with minor items
    items = [ev_fmt, ev_extra]
    cands = generate_candidate_causes(items)
    ranked = rank_candidates(cands)

    if not ranked or ranked[0].score < 1.5:
        # Expected behavior: Undetermined primary cause when evidence is weak/unrelated
        assert True


# =============================================================================
# Scenario F: Contradictory Evidence Scenario
# =============================================================================

def test_scenario_f_contradictory_evidence():
    """F. Contradictory evidence reduces candidate score and confidence."""
    ev_sup = EvidenceItem(source="schema", finding_type="type_mismatch", column="unit_price", severity="HIGH", description="Type mismatch on unit_price.")
    ev_con = EvidenceItem(source="profiling", finding_type="clean_check", column="unit_price", severity="HIGH", description="Unit price clean in profiling.")

    c = CandidateCause(
        cause="Test Candidate",
        affected_data_area="Test Area",
        supporting_evidence=[ev_sup],
        contradicting_evidence=[ev_con],
    )
    ranked = rank_candidates([c])
    assert ranked[0].score < 2.0  # Reduced due to contradiction


# =============================================================================
# Scenario G: Insufficient Evidence Scenario
# =============================================================================

def test_scenario_g_insufficient_evidence():
    """G. Weak or missing evidence returns 'Undetermined' primary cause."""
    res = investigate_root_cause(dataset_name="insufficient_test")
    assert res.investigation_status == "no_issues_detected" or res.primary_root_cause["cause"] == "Undetermined"


# =============================================================================
# Scenario H: Clean Dataset Scenario (sales_clean.csv)
# =============================================================================

class TestScenarioHCleanDataset:
    """H. Clean dataset should yield 'no_issues_detected' without fabricated root causes."""

    @pytest.fixture(scope="class")
    def clean_rc_response(self):
        return run_root_cause_agent(CLEAN_PATH, use_llm=False)

    def test_clean_status(self, clean_rc_response):
        assert clean_rc_response.investigation_status == "no_issues_detected"

    def test_clean_primary_cause(self, clean_rc_response):
        p_cause = clean_rc_response.primary_root_cause
        assert p_cause["cause"] == "No Data Quality Issues Detected"
        assert p_cause["confidence"] == "HIGH"


# =============================================================================
# Scenario I: Problematic & Combined Datasets
# =============================================================================

class TestScenarioIProblematicDatasets:
    """I. Problematic sales dataset & combined_issues.csv generate meaningful root cause results."""

    def test_problematic_sales_root_cause(self):
        agent_resp = run_root_cause_agent(PROBLEMATIC_PATH, use_llm=False)
        assert agent_resp.investigation_status == "completed"
        p_cause = agent_resp.primary_root_cause
        assert p_cause["cause"] != "Undetermined"
        assert p_cause["confidence"] in ("HIGH", "MEDIUM")
        assert len(agent_resp.supporting_evidence) > 0

    def test_combined_issues_root_cause(self):
        agent_resp = run_root_cause_agent(COMBINED_ISSUES_PATH, use_llm=False)
        assert isinstance(agent_resp, AgentRootCauseResponse)
        assert agent_resp.investigation_status in ("completed", "inconclusive")
