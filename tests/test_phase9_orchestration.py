"""
tests/test_phase9_orchestration.py
----------------------------------
Comprehensive automated tests for Phase 9 — LangGraph Multi-Agent Orchestration.

Covers:
  1. Complete problematic-data workflow (pending, approved, rejected)
  2. Clean dataset workflow (skips correction & validation)
  3. Approved correction workflow
  4. Rejected correction workflow
  5. Pending approval workflow
  6. Validation with corrected dataset
  7. Missing corrected dataset handling
  8. Node failure handling & error preservation
  9. Execution trace correctness and ordering
 10. Final report structure and completeness
 11. Immutability guarantee: original raw datasets are never mutated
 12. State pass-through: agent outputs correctly passed across nodes
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest

from app.graph.state import InvestigationState, create_initial_state
from app.graph.nodes import (
    profiling_node,
    anomaly_node,
    schema_node,
    root_cause_node,
    correction_node,
    approval_node,
    validation_node,
    final_report_node,
)
from app.graph.report import generate_final_report, FinalWorkflowReport
from app.graph.workflow import (
    run_data_quality_workflow,
    route_after_profiling,
    route_after_anomaly,
    route_after_schema,
    route_after_root_cause,
    route_after_correction,
    route_after_approval,
    create_workflow_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_problematic_path() -> str:
    path = Path("data/raw/sales_problematic.csv")
    assert path.exists(), "Problematic dataset fixture must exist."
    return str(path)


@pytest.fixture
def raw_clean_path() -> str:
    path = Path("data/raw/sales_clean.csv")
    assert path.exists(), "Clean dataset fixture must exist."
    return str(path)


@pytest.fixture
def corrected_dataset_fixture(tmp_path: Path) -> str:
    """Creates a temporary corrected CSV dataset for post-correction validation."""
    clean_csv = Path("data/raw/sales_clean.csv")
    df = pd.read_csv(clean_csv)
    target = tmp_path / "temp_corrected_sales.csv"
    df.to_csv(target, index=False)
    return str(target)


# ---------------------------------------------------------------------------
# 1. State Initialization & Contract Tests
# ---------------------------------------------------------------------------

def test_initial_state_creation():
    """Verify create_initial_state properly initializes all fields."""
    state = create_initial_state(
        dataset_path="data/raw/sales_problematic.csv",
        dataset_name="custom_sales",
        schema_name="sales",
        approval_status="APPROVED",
        corrected_dataset_path="data/test/fixed.csv",
        use_llm=False,
    )

    assert state["dataset_path"] == "data/raw/sales_problematic.csv"
    assert state["dataset_name"] == "custom_sales"
    assert state["schema_name"] == "sales"
    assert state["approval_status"] == "APPROVED"
    assert state["corrected_dataset_path"] == "data/test/fixed.csv"
    assert state["use_llm"] is False
    assert state["workflow_status"] == "RUNNING"
    assert state["errors"] == []
    assert state["execution_trace"] == []
    assert state["profiling_result"] is None
    assert state["anomaly_result"] is None
    assert state["schema_result"] is None
    assert state["root_cause_result"] is None
    assert state["correction_result"] is None
    assert state["validation_result"] is None
    assert state["final_report"] is None


# ---------------------------------------------------------------------------
# 2. Individual Nodes Tests
# ---------------------------------------------------------------------------

def test_profiling_node(raw_clean_path: str):
    """Test profiling_node standalone execution."""
    state = create_initial_state(dataset_path=raw_clean_path, use_llm=False)
    res = profiling_node(state)

    assert "profiling_result" in res
    assert res["profiling_result"] is not None
    assert len(res["execution_trace"]) == 1
    assert res["execution_trace"][0]["node"] == "profiling_node"
    assert res["execution_trace"][0]["status"] == "completed"


def test_anomaly_node(raw_clean_path: str):
    """Test anomaly_node standalone execution."""
    state = create_initial_state(dataset_path=raw_clean_path, use_llm=False)
    res = anomaly_node(state)

    assert "anomaly_result" in res
    assert res["anomaly_result"] is not None
    assert len(res["execution_trace"]) == 1
    assert res["execution_trace"][0]["node"] == "anomaly_node"


def test_schema_node(raw_clean_path: str):
    """Test schema_node standalone execution."""
    state = create_initial_state(dataset_path=raw_clean_path, use_llm=False)
    res = schema_node(state)

    assert "schema_result" in res
    assert res["schema_result"] is not None
    assert len(res["execution_trace"]) == 1
    assert res["execution_trace"][0]["node"] == "schema_node"


def test_approval_node_statuses():
    """Test approval_node handles APPROVED, REJECTED, PENDING."""
    # Test APPROVED
    s1 = create_initial_state(dataset_path="dummy.csv", approval_status="APPROVED")
    r1 = approval_node(s1)
    assert r1["approval_status"] == "APPROVED"
    assert r1["workflow_status"] == "RUNNING"
    assert r1["execution_trace"][-1]["status"] == "approved"

    # Test REJECTED
    s2 = create_initial_state(dataset_path="dummy.csv", approval_status="REJECTED")
    r2 = approval_node(s2)
    assert r2["approval_status"] == "REJECTED"
    assert r2["workflow_status"] == "REJECTED"
    assert r2["execution_trace"][-1]["status"] == "rejected"

    # Test PENDING
    s3 = create_initial_state(dataset_path="dummy.csv", approval_status="PENDING")
    r3 = approval_node(s3)
    assert r3["approval_status"] == "PENDING"
    assert r3["workflow_status"] == "PENDING_APPROVAL"
    assert r3["execution_trace"][-1]["status"] == "pending"


# ---------------------------------------------------------------------------
# 3. Clean Dataset Workflow Test (Skips Correction & Validation)
# ---------------------------------------------------------------------------

def test_clean_dataset_workflow_skips_correction(raw_clean_path: str):
    """Clean dataset must route directly from root cause to final report."""
    final_state = run_data_quality_workflow(
        dataset_path=raw_clean_path,
        approval_status="PENDING",
        use_llm=False,
    )

    assert final_state["workflow_status"] == "NO_CORRECTION_REQUIRED"
    assert final_state["correction_result"] is None
    assert final_state["validation_result"] is None

    report = final_state.get("final_report", {})
    assert report["workflow_status"] == "NO_CORRECTION_REQUIRED"
    assert "clean" in report["final_recommendation"].lower()

    # Check execution trace contains 5 nodes: profiling, anomaly, schema, root_cause, final_report
    nodes_executed = [step["node"] for step in final_state["execution_trace"]]
    assert nodes_executed == [
        "profiling_node",
        "anomaly_node",
        "schema_node",
        "root_cause_node",
        "final_report_node",
    ]


# ---------------------------------------------------------------------------
# 4. Problematic Workflow — Pending Approval
# ---------------------------------------------------------------------------

def test_problematic_workflow_pending_approval(raw_problematic_path: str):
    """Problematic dataset with PENDING approval generates correction plan then stops."""
    final_state = run_data_quality_workflow(
        dataset_path=raw_problematic_path,
        approval_status="PENDING",
        use_llm=False,
    )

    assert final_state["workflow_status"] == "PENDING_APPROVAL"
    assert final_state["correction_result"] is not None
    assert final_state["validation_result"] is None

    report = final_state.get("final_report", {})
    assert report["workflow_status"] == "PENDING_APPROVAL"
    assert report["approval_status"] == "PENDING"
    assert report["correction_recommendation"]["status"] == "recommendations_generated"
    assert report["validation_status"]["status"] == "not_executed"

    nodes_executed = [step["node"] for step in final_state["execution_trace"]]
    assert nodes_executed == [
        "profiling_node",
        "anomaly_node",
        "schema_node",
        "root_cause_node",
        "correction_node",
        "approval_node",
        "final_report_node",
    ]


# ---------------------------------------------------------------------------
# 5. Problematic Workflow — Rejected Approval
# ---------------------------------------------------------------------------

def test_problematic_workflow_rejected_approval(raw_problematic_path: str):
    """Problematic dataset with REJECTED approval stops before validation."""
    final_state = run_data_quality_workflow(
        dataset_path=raw_problematic_path,
        approval_status="REJECTED",
        use_llm=False,
    )

    assert final_state["workflow_status"] == "REJECTED"
    assert final_state["correction_result"] is not None
    assert final_state["validation_result"] is None

    report = final_state.get("final_report", {})
    assert report["workflow_status"] == "REJECTED"
    assert "REJECTED" in report["final_recommendation"]

    nodes_executed = [step["node"] for step in final_state["execution_trace"]]
    assert nodes_executed == [
        "profiling_node",
        "anomaly_node",
        "schema_node",
        "root_cause_node",
        "correction_node",
        "approval_node",
        "final_report_node",
    ]


# ---------------------------------------------------------------------------
# 6. Problematic Workflow — Approved with Corrected Dataset
# ---------------------------------------------------------------------------

def test_problematic_workflow_approved_with_validation(
    raw_problematic_path: str,
    corrected_dataset_fixture: str,
):
    """Approved workflow with supplied corrected dataset completes full loop through validation."""
    final_state = run_data_quality_workflow(
        dataset_path=raw_problematic_path,
        approval_status="APPROVED",
        corrected_dataset_path=corrected_dataset_fixture,
        use_llm=False,
    )

    assert final_state["workflow_status"] == "COMPLETED"
    assert final_state["correction_result"] is not None
    assert final_state["validation_result"] is not None

    val_resp = final_state["validation_result"]
    assert val_resp.validation_status == "PASSED"

    report = final_state.get("final_report", {})
    assert report["workflow_status"] == "COMPLETED"
    assert report["validation_status"]["status"] == "completed"
    assert report["validation_status"]["verdict"] == "PASSED"

    nodes_executed = [step["node"] for step in final_state["execution_trace"]]
    assert nodes_executed == [
        "profiling_node",
        "anomaly_node",
        "schema_node",
        "root_cause_node",
        "correction_node",
        "approval_node",
        "validation_node",
        "final_report_node",
    ]


# ---------------------------------------------------------------------------
# 7. Problematic Workflow — Approved WITHOUT Corrected Dataset
# ---------------------------------------------------------------------------

def test_problematic_workflow_approved_missing_corrected_dataset(raw_problematic_path: str):
    """Approved workflow without static fixture dynamically generates corrected copy and executes validation."""
    final_state = run_data_quality_workflow(
        dataset_path=raw_problematic_path,
        approval_status="APPROVED",
        corrected_dataset_path=None,
        use_llm=False,
    )

    # In Phase 12, a dynamic corrected copy is generated and validated
    assert final_state["validation_result"] is not None
    nodes_executed = [step["node"] for step in final_state["execution_trace"]]
    assert "validation_node" in nodes_executed
    assert nodes_executed[-1] == "final_report_node"


# ---------------------------------------------------------------------------
# 8. Node Failure Handling & Error Preservation
# ---------------------------------------------------------------------------

def test_node_failure_handling_and_recovery():
    """Simulated failure in profiling_node preserves error and gracefully compiles final report."""
    with patch("app.graph.nodes.run_profiling_agent", side_effect=RuntimeError("Simulated engine fault")):
        final_state = run_data_quality_workflow(
            dataset_path="data/raw/sales_problematic.csv",
            approval_status="APPROVED",
            use_llm=False,
        )

    assert final_state["workflow_status"] == "FAILED"
    assert len(final_state["errors"]) > 0
    assert "Simulated engine fault" in final_state["errors"][0]["error"]

    report = final_state.get("final_report", {})
    assert report["workflow_status"] == "FAILED"
    assert len(report["errors"]) > 0

    trace = final_state["execution_trace"]
    assert any(step.get("status") == "failed" for step in trace)
    assert trace[-1]["node"] == "final_report_node"


# ---------------------------------------------------------------------------
# 9. Execution Trace Integrity
# ---------------------------------------------------------------------------

def test_execution_trace_structure(raw_problematic_path: str, corrected_dataset_fixture: str):
    """Verify execution trace entries have valid keys and status progression."""
    final_state = run_data_quality_workflow(
        dataset_path=raw_problematic_path,
        approval_status="APPROVED",
        corrected_dataset_path=corrected_dataset_fixture,
        use_llm=False,
    )

    trace = final_state["execution_trace"]
    assert len(trace) == 8
    for step in trace:
        assert "node" in step
        assert "status" in step
        assert isinstance(step["node"], str)
        assert isinstance(step["status"], str)


# ---------------------------------------------------------------------------
# 10. Final Report Completeness
# ---------------------------------------------------------------------------

def test_final_report_structure(raw_problematic_path: str, corrected_dataset_fixture: str):
    """Verify final report conforms to expected structure and contains all sections."""
    final_state = run_data_quality_workflow(
        dataset_path=raw_problematic_path,
        approval_status="APPROVED",
        corrected_dataset_path=corrected_dataset_fixture,
        use_llm=False,
    )

    report = final_state["final_report"]
    required_keys = [
        "dataset",
        "workflow_status",
        "detected_issues",
        "root_cause",
        "correction_recommendation",
        "approval_status",
        "validation_status",
        "execution_trace",
        "unresolved_issues",
        "errors",
        "final_recommendation",
    ]
    for key in required_keys:
        assert key in report, f"Missing key in final report: {key}"

    assert report["detected_issues"]["missing_values_count"] > 0
    assert report["detected_issues"]["duplicate_rows_count"] > 0
    assert report["root_cause"]["confidence"] == "HIGH"
    assert report["correction_recommendation"]["total_actions"] > 0


# ---------------------------------------------------------------------------
# 11. Safety & Immutability Guarantee
# ---------------------------------------------------------------------------

def test_original_dataset_is_never_mutated(raw_problematic_path: str, corrected_dataset_fixture: str):
    """Verify that the raw problematic dataset hash remains unchanged after workflow execution."""
    raw_path = Path(raw_problematic_path)
    initial_bytes = raw_path.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    run_data_quality_workflow(
        dataset_path=raw_problematic_path,
        approval_status="APPROVED",
        corrected_dataset_path=corrected_dataset_fixture,
        use_llm=False,
    )

    post_bytes = raw_path.read_bytes()
    post_hash = hashlib.sha256(post_bytes).hexdigest()

    assert initial_hash == post_hash, "Raw dataset file was mutated during workflow execution!"


# ---------------------------------------------------------------------------
# 12. Conditional Edge Routing Unit Tests
# ---------------------------------------------------------------------------

def test_routing_functions_standalone():
    """Test conditional edge router branches directly."""
    failed_state: InvestigationState = {"workflow_status": "FAILED"}
    assert route_after_profiling(failed_state) == "final_report_node"
    assert route_after_anomaly(failed_state) == "final_report_node"
    assert route_after_schema(failed_state) == "final_report_node"
    assert route_after_root_cause(failed_state) == "final_report_node"
    assert route_after_correction(failed_state) == "final_report_node"
    assert route_after_approval(failed_state) == "final_report_node"

    running_state: InvestigationState = {"workflow_status": "RUNNING"}
    assert route_after_profiling(running_state) == "anomaly_node"
    assert route_after_anomaly(running_state) == "schema_node"
    assert route_after_schema(running_state) == "root_cause_node"
    assert route_after_correction(running_state) == "approval_node"

    appr_state: InvestigationState = {"workflow_status": "RUNNING", "approval_status": "APPROVED", "corrected_dataset_path": "fixed.csv"}
    assert route_after_approval(appr_state) == "validation_node"

    rej_state: InvestigationState = {"workflow_status": "RUNNING", "approval_status": "REJECTED"}
    assert route_after_approval(rej_state) == "final_report_node"
