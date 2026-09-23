"""
tests/test_phase10_api.py
-------------------------
Phase 10 — FastAPI Endpoints & End-to-End Workflow Test Suite.

Tests:
  1. GET /health status and response metadata.
  2. POST /investigate on problematic dataset.
  3. POST /investigate with PENDING approval (validation not executed).
  4. POST /investigate with REJECTED approval.
  5. POST /investigate with APPROVED + corrected fixture (validation PASSED).
  6. POST /investigate on clean dataset (skips correction & validation).
  7. POST /investigate with nonexistent dataset (validation error).
  8. POST /investigate with invalid approval status (validation error).
  9. POST /investigate with missing required fields (validation error).
 10. POST /investigate with invalid file extension (non-CSV).
 11. Response schema completeness and contract verification.
 12. Immutability guarantee: raw dataset remains identical before and after API invocation.
 13. End-to-end multi-agent integration verification.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBLEMATIC_CSV = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "raw" / "sales_clean.csv"
CORRECTED_CSV = PROJECT_ROOT / "data" / "test" / "fully_corrected.csv"


# ---------------------------------------------------------------------------
# 1. Health Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_health_endpoint():
    """Verify GET /health returns 200 OK and expected system metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Autonomous Data Quality Investigation Agent" in data["service"]
    assert "Phase 10" in data["phase"]
    assert data["agents_active"] is False  # Preserves Phase 1 contract


# ---------------------------------------------------------------------------
# 2. Problematic Dataset Investigation (Default / Pending)
# ---------------------------------------------------------------------------

def test_investigate_problematic_dataset_default_pending():
    """Verify POST /investigate with default pending approval generates correction plan."""
    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "PENDING",
        "corrected_dataset_path": None,
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["workflow_status"] == "PENDING_APPROVAL"
    assert data["approval_status"] == "PENDING"
    assert data["detected_issues"]["missing_values_count"] > 0
    assert data["detected_issues"]["duplicate_rows_count"] > 0
    assert data["root_cause"]["primary_cause"] != "None"
    assert data["correction_recommendation"]["total_actions"] > 0
    assert data["validation_status"]["verdict"] in ("NOT_VALIDATED", "not_executed")


# ---------------------------------------------------------------------------
# 3. Explicit PENDING Approval
# ---------------------------------------------------------------------------

def test_investigate_pending_approval_bypasses_validation():
    """Verify POST /investigate with explicit PENDING does not run validation."""
    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "PENDING",
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["workflow_status"] == "PENDING_APPROVAL"
    nodes = [s["node"] for s in data["execution_trace"]]
    assert "validation_node" not in nodes
    assert "final_report_node" in nodes


# ---------------------------------------------------------------------------
# 4. REJECTED Approval
# ---------------------------------------------------------------------------

def test_investigate_rejected_approval_terminates_safely():
    """Verify POST /investigate with REJECTED approval halts without validation."""
    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "REJECTED",
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["workflow_status"] == "REJECTED"
    assert data["approval_status"] == "REJECTED"
    assert "REJECTED" in data["final_recommendation"]
    nodes = [s["node"] for s in data["execution_trace"]]
    assert "validation_node" not in nodes


# ---------------------------------------------------------------------------
# 5. APPROVED Approval with Corrected Dataset Fixture
# ---------------------------------------------------------------------------

def test_investigate_approved_with_corrected_fixture():
    """Verify POST /investigate with APPROVED + fully_corrected.csv yields PASSED validation."""
    assert CORRECTED_CSV.exists(), "fully_corrected.csv test fixture must exist."

    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "APPROVED",
        "corrected_dataset_path": str(CORRECTED_CSV),
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["workflow_status"] == "COMPLETED"
    assert data["approval_status"] == "APPROVED"
    assert data["validation_status"]["status"] == "completed"
    assert data["validation_status"]["verdict"] == "PASSED"
    assert data["validation_status"]["passed_checks"] > 0
    assert data["validation_status"]["failed_checks"] == 0

    nodes = [s["node"] for s in data["execution_trace"]]
    assert "validation_node" in nodes
    assert "final_report_node" in nodes


# ---------------------------------------------------------------------------
# 6. Clean Dataset Investigation
# ---------------------------------------------------------------------------

def test_investigate_clean_dataset():
    """Verify POST /investigate on clean dataset transitions to NO_CORRECTION_REQUIRED."""
    payload = {
        "dataset_path": str(CLEAN_CSV),
        "approval_status": "PENDING",
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["workflow_status"] == "NO_CORRECTION_REQUIRED"
    assert data["detected_issues"]["missing_values_count"] == 0
    assert data["detected_issues"]["duplicate_rows_count"] == 0
    assert "clean" in data["final_recommendation"].lower()


# ---------------------------------------------------------------------------
# 7. Request Validation: Missing Dataset File
# ---------------------------------------------------------------------------

def test_investigate_nonexistent_dataset():
    """Verify POST /investigate rejects nonexistent dataset paths with 422 Unprocessable Entity."""
    payload = {
        "dataset_path": "data/raw/nonexistent_file_xyz.csv",
        "approval_status": "PENDING",
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any("does not exist" in str(err) for err in data["detail"])


# ---------------------------------------------------------------------------
# 8. Request Validation: Invalid Approval Status
# ---------------------------------------------------------------------------

def test_investigate_invalid_approval_status():
    """Verify POST /investigate rejects invalid approval status with 422."""
    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "MAYBE",
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("approval_status" in str(err) for err in data["detail"])


# ---------------------------------------------------------------------------
# 9. Request Validation: Missing Required Field
# ---------------------------------------------------------------------------

def test_investigate_missing_required_field():
    """Verify POST /investigate rejects request without dataset_path with 422."""
    payload = {
        "approval_status": "APPROVED"
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 10. Request Validation: Unsupported File Extension
# ---------------------------------------------------------------------------

def test_investigate_unsupported_file_extension(tmp_path: Path):
    """Verify POST /investigate rejects non-CSV files."""
    text_file = tmp_path / "data.txt"
    text_file.write_text("sample content")

    payload = {
        "dataset_path": str(text_file),
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("Unsupported file type" in str(err) for err in data["detail"])


# ---------------------------------------------------------------------------
# 11. Response Schema Completeness
# ---------------------------------------------------------------------------

def test_investigate_response_structure():
    """Verify POST /investigate response contains all required top-level contract keys."""
    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "PENDING",
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    required_fields = [
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
    for field in required_fields:
        assert field in data, f"Missing required field in API response: {field}"


# ---------------------------------------------------------------------------
# 12. Immutability Guarantee: Raw Dataset Never Modified
# ---------------------------------------------------------------------------

def test_investigate_raw_dataset_hash_unchanged():
    """Verify SHA-256 hash of raw dataset is identical before and after investigation."""
    initial_bytes = PROBLEMATIC_CSV.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "APPROVED",
        "corrected_dataset_path": str(CORRECTED_CSV),
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200

    post_bytes = PROBLEMATIC_CSV.read_bytes()
    post_hash = hashlib.sha256(post_bytes).hexdigest()

    assert initial_hash == post_hash, "Raw dataset file was modified by API execution!"


# ---------------------------------------------------------------------------
# 13. Complete Multi-Agent End-to-End Workflow Integration Test
# ---------------------------------------------------------------------------

def test_end_to_end_complete_workflow_integration():
    """
    Assert full end-to-end pipeline execution:
    sales_problematic.csv -> Profiling -> Anomaly -> Schema -> Root Cause -> Correction -> APPROVED -> Validation -> Final Report
    """
    payload = {
        "dataset_path": str(PROBLEMATIC_CSV),
        "approval_status": "APPROVED",
        "corrected_dataset_path": str(CORRECTED_CSV),
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 1. Assert all expected workflow stages execute in order
    executed_nodes = [step["node"] for step in data["execution_trace"]]
    expected_nodes = [
        "profiling_node",
        "anomaly_node",
        "schema_node",
        "root_cause_node",
        "correction_node",
        "approval_node",
        "validation_node",
        "final_report_node",
    ]
    assert executed_nodes == expected_nodes

    # 2. Assert approval is recorded
    assert data["approval_status"] == "APPROVED"

    # 3. Assert validation executed and passed
    assert data["validation_status"]["status"] == "completed"
    assert data["validation_status"]["verdict"] == "PASSED"
    assert data["validation_status"]["passed_checks"] == 5
    assert data["validation_status"]["failed_checks"] == 0

    # 4. Assert final recommendation exists
    assert "PASSED" in data["final_recommendation"]
