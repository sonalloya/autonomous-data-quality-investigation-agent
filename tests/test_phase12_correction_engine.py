"""
tests/test_phase12_correction_engine.py
---------------------------------------
Phase 12 — Generic Deterministic Correction Engine & Secure Download Test Suite.

Covers:
  1. Generic CSV correction with arbitrary columns & rows.
  2. Duplicate record removal (canonical retention, copy only).
  3. Safe type conversion & unit suffix stripping.
  4. Date format normalization to ISO %Y-%m-%d without guessing.
  5. Safe categorical domain mapping & alias normalization.
  6. Missing-value handling (computable calculation vs reporting unresolved).
  7. Unresolved issue tracking (never silently guessing missing IDs or bad dates).
  8. Human approval gating (PENDING vs REJECTED vs APPROVED).
  9. Corrected file creation (distinct new file).
 10. Immutability guarantee (original file hash unchanged).
 11. Post-correction validation on dynamically generated copy.
 12. Corrected CSV download endpoint (GET /download/corrected).
 13. Path traversal and unauthorized filesystem access protections.
 14. Clean dataset behavior (no unnecessary correction or download).
 15. Arbitrary uploaded CSV investigation and download.
 16. Partial / failed validation behavior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from app.tools.correction_engine import apply_corrections, CorrectionExecutionResult
from app.graph.workflow import run_data_quality_workflow

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PROBLEMATIC = DATA_DIR / "raw" / "sales_problematic.csv"
RAW_CLEAN = DATA_DIR / "raw" / "sales_clean.csv"


# ---------------------------------------------------------------------------
# 1. Generic CSV Correction
# ---------------------------------------------------------------------------

def test_generic_csv_correction_arbitrary_schema(tmp_path: Path):
    """Verify generic correction engine works on an arbitrary non-sales dataset."""
    df_sample = pd.DataFrame({
        "employee_id": ["E01", "E02", "E02", "E03"],
        "name": [" Alice ", "Bob", "Bob", "Charlie"],
        "join_date": ["2025/01/15", "02/20/2025", "02/20/2025", "2025-03-10"],
        "salary": ["$75,000.00", "80000 USD", "80000 USD", "$92,500.50"],
    })
    raw_file = tmp_path / "employees.csv"
    df_sample.to_csv(raw_file, index=False)

    df_corr, res = apply_corrections(
        dataset_input=raw_file,
        schema=None,
        dataset_name="employees",
    )

    assert isinstance(res, CorrectionExecutionResult)
    assert len(df_corr) == 3  # 1 duplicate removed
    assert df_corr["employee_id"].tolist() == ["E01", "E02", "E03"]
    assert df_corr["name"].tolist() == ["Alice", "Bob", "Charlie"]
    assert df_corr["join_date"].iloc[0] == "2025-01-15"
    assert df_corr["join_date"].iloc[1] == "2025-02-20"
    assert res.corrections_applied > 0
    assert raw_file.exists()


# ---------------------------------------------------------------------------
# 2. Duplicate Record Removal
# ---------------------------------------------------------------------------

def test_duplicate_removal_retains_canonical(tmp_path: Path):
    """Verify exact duplicates and ID duplicates are removed in copy only."""
    df = pd.DataFrame({
        "id": ["A1", "A2", "A2", "A3"],
        "val": [10, 20, 20, 30]
    })
    raw_file = tmp_path / "dup_test.csv"
    df.to_csv(raw_file, index=False)

    df_corr, res = apply_corrections(raw_file, schema=None)
    assert len(df_corr) == 3
    assert df_corr["id"].tolist() == ["A1", "A2", "A3"]
    assert any(c["correction_type"] == "duplicate_removal" for c in res.corrections)


# ---------------------------------------------------------------------------
# 3. Safe Type Conversion & Suffix Stripping
# ---------------------------------------------------------------------------

def test_safe_numeric_type_conversion():
    """Verify safe conversion of currency, units, and formatted strings."""
    df = pd.DataFrame({
        "quantity": ["10 units", " 5 ", "12 items", "20"],
        "price": ["$10.50", "€25.00", " 15.75 ", "1,000.00"],
    })

    df_corr, res = apply_corrections(df, schema=None)
    assert pd.api.types.is_numeric_dtype(df_corr["quantity"])
    assert pd.api.types.is_numeric_dtype(df_corr["price"])
    assert df_corr["quantity"].tolist() == [10, 5, 12, 20]
    assert df_corr["price"].tolist() == [10.5, 25.0, 15.75, 1000.0]


# ---------------------------------------------------------------------------
# 4. Date Normalization to ISO %Y-%m-%d
# ---------------------------------------------------------------------------

def test_date_format_normalization():
    """Verify dates with slash, dot, or dash are normalized to ISO format."""
    df = pd.DataFrame({
        "event_date": ["2026/07/05", "07/05/2026", "2026-07-05", "2026.07.05"]
    })

    df_corr, res = apply_corrections(df, schema=None)
    for d_val in df_corr["event_date"]:
        assert str(d_val).startswith("2026-")


# ---------------------------------------------------------------------------
# 5. Safe Categorical Domain Normalization
# ---------------------------------------------------------------------------

def test_safe_categorical_normalization():
    """Verify categorical values are normalized against approved domain values."""
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "payment_method": [" credit card ", "paypal", "in-store"],
    })

    df_corr, res = apply_corrections(df, schema="sales")
    assert df_corr["payment_method"].iloc[0] == "Credit Card"
    assert df_corr["payment_method"].iloc[1] == "PayPal"


# ---------------------------------------------------------------------------
# 6. Missing Value Handling & Deterministic Computation
# ---------------------------------------------------------------------------

def test_computed_field_missing_value_handling():
    """Verify computable fields like total_amount = quantity * unit_price are deterministically filled."""
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "quantity": [2, 5],
        "unit_price": [10.0, 20.0],
        "total_amount": [None, 100.0],
    })

    df_corr, res = apply_corrections(df, schema="sales")
    assert df_corr["total_amount"].iloc[0] == 20.0
    assert any(c["correction_type"] == "computed_field" for c in res.corrections)


# ---------------------------------------------------------------------------
# 7. Unresolved Issue Preservation (No Guessing)
# ---------------------------------------------------------------------------

def test_unresolved_issues_preservation_no_guessing():
    """Verify system never invents missing IDs or guesses invalid dates."""
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "customer_id": [None, "CU002"],  # Missing mandatory ID
        "transaction_date": ["2026-05-10", "invalid_date_xyz"],  # Bad date
    })

    df_corr, res = apply_corrections(df, schema="sales")
    # Original invalid date and None customer_id must not be invented
    assert pd.isna(df_corr["customer_id"].iloc[0])
    assert df_corr["transaction_date"].iloc[1] == "invalid_date_xyz"
    assert len(res.unresolved_issues) >= 2


# ---------------------------------------------------------------------------
# 8. Human Approval Gating
# ---------------------------------------------------------------------------

def test_human_approval_gating(tmp_path: Path):
    """Verify workflow generates corrected file ONLY when approved."""
    df = pd.DataFrame({
        "id": ["1", "1", "2"],
        "val": [" $10 ", " $10 ", " $20 "],
    })
    raw_file = tmp_path / "approval_gate_test.csv"
    df.to_csv(raw_file, index=False)

    # 1. PENDING: No corrected file created
    state_pending = run_data_quality_workflow(
        dataset_path=str(raw_file),
        approval_status="PENDING",
        use_llm=False,
    )
    assert state_pending["workflow_status"] == "PENDING_APPROVAL"
    assert state_pending.get("corrected_dataset_path") is None

    # 2. REJECTED: No corrected file created
    state_rejected = run_data_quality_workflow(
        dataset_path=str(raw_file),
        approval_status="REJECTED",
        use_llm=False,
    )
    assert state_rejected["workflow_status"] == "REJECTED"
    assert state_rejected.get("corrected_dataset_path") is None

    # 3. APPROVED: Corrected file dynamically created and validated
    state_approved = run_data_quality_workflow(
        dataset_path=str(raw_file),
        approval_status="APPROVED",
        use_llm=False,
    )
    assert state_approved["workflow_status"] in ("COMPLETED", "APPROVED")
    corr_path = state_approved.get("corrected_dataset_path")
    assert corr_path is not None
    assert Path(corr_path).exists()


# ---------------------------------------------------------------------------
# 9. Corrected File Creation
# ---------------------------------------------------------------------------

def test_corrected_file_creation(tmp_path: Path):
    """Verify apply_corrections saves copy with dynamic filename."""
    df = pd.DataFrame({"id": ["A", "B"], "val": ["10", "20"]})
    raw_file = tmp_path / "orders_2026.csv"
    df.to_csv(raw_file, index=False)

    df_corr, res = apply_corrections(raw_file, schema=None)
    assert res.corrected_dataset_path is not None
    assert "orders_2026_corrected.csv" in res.corrected_dataset_path
    assert Path(res.corrected_dataset_path).is_file()


# ---------------------------------------------------------------------------
# 10. Immutability Guarantee
# ---------------------------------------------------------------------------

def test_original_dataset_immutability(tmp_path: Path):
    """Verify original file SHA-256 hash remains strictly unchanged."""
    df = pd.DataFrame({
        "id": ["1", "1", "2"],
        "num": [" $10 ", " $10 ", " $20 "]
    })
    raw_file = tmp_path / "immutability_test.csv"
    df.to_csv(raw_file, index=False)

    initial_bytes = raw_file.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    # Run correction
    apply_corrections(raw_file, schema=None)

    post_bytes = raw_file.read_bytes()
    post_hash = hashlib.sha256(post_bytes).hexdigest()

    assert initial_hash == post_hash, "Original dataset file was modified!"


# ---------------------------------------------------------------------------
# 11. Post-Correction Validation of Generated Copy
# ---------------------------------------------------------------------------

def test_validation_of_dynamically_generated_copy(tmp_path: Path):
    """Verify workflow validates newly generated file rather than predefined static fixture."""
    df_problem = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T2"],  # Has duplicate
        "customer_id": ["C1", "C2", "C2"],
        "product_id": ["P1", "P2", "P2"],
        "transaction_date": ["2026/05/01", "2026-05-02", "2026-05-02"],
        "quantity": [2, 3, 3],
        "unit_price": [10.0, 15.0, 15.0],
        "total_amount": [20.0, 45.0, 45.0],
        "region": ["North", "South", "South"],
        "payment_method": ["Credit Card", "Debit Card", "Debit Card"],
        "sales_channel": ["Online", "In-Store", "In-Store"],
    })
    raw_file = tmp_path / "custom_sales_test.csv"
    df_problem.to_csv(raw_file, index=False)

    state = run_data_quality_workflow(
        dataset_path=str(raw_file),
        approval_status="APPROVED",
        schema_name="sales",
        use_llm=False,
    )

    assert state["workflow_status"] == "COMPLETED"
    assert state.get("validation_result") is not None
    val_status = state["validation_result"].validation_status
    assert val_status == "PASSED"
    assert state["final_report"]["corrected_dataset"]["download_available"] is True


# ---------------------------------------------------------------------------
# 12. Corrected CSV Download Endpoint
# ---------------------------------------------------------------------------

def test_download_corrected_csv_endpoint(tmp_path: Path):
    """Verify GET /download/corrected serves the generated corrected dataset."""
    # Create test corrected file in data/corrected
    test_dir = PROJECT_ROOT / "data" / "corrected"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test_download_sample_corrected.csv"
    test_file.write_text("id,value\n1,100\n2,200\n", encoding="utf-8")

    # Download by filename
    resp = client.get(f"/download/corrected?filename={test_file.name}")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert "1,100" in resp.text

    # Download by path
    resp_path = client.get(f"/download/corrected?path={str(test_file)}")
    assert resp_path.status_code == 200


# ---------------------------------------------------------------------------
# 13. Path Traversal Protection on Download Endpoint
# ---------------------------------------------------------------------------

def test_download_path_traversal_protection():
    """Verify download endpoint rejects path traversal attacks with 403 or 404."""
    # Attempt path traversal
    resp1 = client.get("/download/corrected?path=../../main.py")
    assert resp1.status_code in (400, 403, 404)

    resp2 = client.get("/download/corrected?filename=../../../etc/passwd")
    assert resp2.status_code in (400, 403, 404)

    resp3 = client.get("/download/corrected?path=C:/Windows/win.ini")
    assert resp3.status_code in (400, 403, 404)


# ---------------------------------------------------------------------------
# 14. Clean Dataset Behavior
# ---------------------------------------------------------------------------

def test_clean_dataset_no_unnecessary_correction():
    """Verify clean dataset triggers NO_CORRECTION_REQUIRED without generating copy."""
    state = run_data_quality_workflow(
        dataset_path=str(RAW_CLEAN),
        approval_status="APPROVED",
        use_llm=False,
    )
    assert state["workflow_status"] == "NO_CORRECTION_REQUIRED"
    report = state["final_report"]
    assert "clean" in report["final_recommendation"].lower()


# ---------------------------------------------------------------------------
# 15. Arbitrary Uploaded CSV Investigation via API
# ---------------------------------------------------------------------------

def test_api_investigate_arbitrary_uploaded_csv(tmp_path: Path):
    """Verify POST /investigate on an arbitrary uploaded CSV generates and validates copy."""
    df_custom = pd.DataFrame({
        "id": ["1", "2", "2"],
        "date": ["2026/01/01", "2026/01/02", "2026/01/02"],
        "metric": ["$10.00", "$20.00", "$20.00"]
    })
    custom_csv = tmp_path / "custom_upload.csv"
    df_custom.to_csv(custom_csv, index=False)

    payload = {
        "dataset_path": str(custom_csv),
        "approval_status": "APPROVED",
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["workflow_status"] == "COMPLETED"
    assert data["approval_status"] == "APPROVED"
    assert data["corrected_dataset"] is not None
    assert data["corrected_dataset"]["corrections_applied"] >= 1
    assert data["corrected_dataset"]["download_available"] is True


# ---------------------------------------------------------------------------
# 16. Partial / Unresolved Issues Validation Behavior
# ---------------------------------------------------------------------------

def test_partial_validation_with_unresolved_issues(tmp_path: Path):
    """Verify dataset with unresolvable null customer_id is labeled with unresolved count."""
    df_partial = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "customer_id": [None, "C2"],  # Cannot be invented
        "product_id": ["P1", "P2"],
        "transaction_date": ["2026-01-01", "2026-01-02"],
        "quantity": [1, 2],
        "unit_price": [10.0, 20.0],
        "total_amount": [10.0, 40.0],
        "region": ["North", "South"],
        "payment_method": ["Credit Card", "Debit Card"],
        "sales_channel": ["Online", "In-Store"],
    })
    raw_file = tmp_path / "partial_sales.csv"
    df_partial.to_csv(raw_file, index=False)

    state = run_data_quality_workflow(
        dataset_path=str(raw_file),
        approval_status="APPROVED",
        schema_name="sales",
        use_llm=False,
    )

    report = state["final_report"]
    assert report["corrected_dataset"] is not None
    assert report["corrected_dataset"]["issues_remaining"] > 0
