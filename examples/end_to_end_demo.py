"""
examples/end_to_end_demo.py
----------------------------
Phase 10 — Autonomous Data Quality Investigation Agent: End-to-End Demonstration.

Demonstrates two primary production workflows:

SCENARIO A: Problematic Dataset with Human Approval & Corrected Fixture
  Workflow executes:
    Profiling -> Anomaly -> Schema -> Root Cause -> Correction -> APPROVED -> Validation -> Final Report
  Post-correction validation status: PASSED.

SCENARIO B: Problematic Dataset with Pending / Rejected Human Approval
  Workflow executes:
    Profiling -> Anomaly -> Schema -> Root Cause -> Correction -> PENDING / REJECTED -> Final Report
  Validation is bypassed; workflow safely pauses requiring operator approval.

GUARANTEES:
  - Raw datasets (data/raw/sales_problematic.csv) are NEVER overwritten or modified.
  - Corrected data is read from a separate test fixture (data/test/fully_corrected.csv).
  - Validation verdict is derived deterministically.
"""

import argparse
import hashlib
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.graph.workflow import run_data_quality_workflow
from app.graph.report import format_final_report_text


def get_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for immutability verification."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def run_scenario_a(verbose: bool = True):
    """
    Scenario A: Problematic dataset with approved correction and safe fixture validation.
    """
    raw_path = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"
    corrected_path = PROJECT_ROOT / "data" / "test" / "fully_corrected.csv"

    print("\n" + "=" * 80)
    print("SCENARIO A — PROBLEMATIC DATASET WITH HUMAN APPROVAL & VALIDATION")
    print("=" * 80)
    print(f"Target Dataset: {raw_path}")
    print(f"Corrected Fixture: {corrected_path}")
    print("Approval Decision: APPROVED")
    print("-" * 80)

    # Record initial hash for read-only guarantee verification
    initial_hash = get_file_hash(raw_path)

    final_state = run_data_quality_workflow(
        dataset_path=str(raw_path),
        dataset_name="sales_problematic",
        schema_name="sales",
        approval_status="APPROVED",
        corrected_dataset_path=str(corrected_path),
        use_llm=False,
    )

    # Verify immutability
    post_hash = get_file_hash(raw_path)
    assert initial_hash == post_hash, "CRITICAL ERROR: Raw dataset was modified during execution!"

    report = final_state.get("final_report", {})
    formatted_text = format_final_report_text(report)
    print(formatted_text)
    print(f"\n[Safety Check] Raw dataset SHA-256 unchanged: {post_hash[:16]}... (READ-ONLY VERIFIED)")
    return final_state


def run_scenario_b(verbose: bool = True):
    """
    Scenario B: Problematic dataset with pending human approval (validation bypassed).
    """
    raw_path = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"

    print("\n" + "=" * 80)
    print("SCENARIO B — PROBLEMATIC DATASET WITH PENDING HUMAN APPROVAL")
    print("=" * 80)
    print(f"Target Dataset: {raw_path}")
    print("Approval Decision: PENDING")
    print("-" * 80)

    initial_hash = get_file_hash(raw_path)

    final_state = run_data_quality_workflow(
        dataset_path=str(raw_path),
        dataset_name="sales_problematic",
        schema_name="sales",
        approval_status="PENDING",
        corrected_dataset_path=None,
        use_llm=False,
    )

    post_hash = get_file_hash(raw_path)
    assert initial_hash == post_hash, "CRITICAL ERROR: Raw dataset was modified during execution!"

    report = final_state.get("final_report", {})
    formatted_text = format_final_report_text(report)
    print(formatted_text)
    print(f"\n[Safety Check] Raw dataset SHA-256 unchanged: {post_hash[:16]}... (READ-ONLY VERIFIED)")
    print("[HITL Gate] Validation did NOT execute because human approval is PENDING.")
    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Quality Investigation Agent — End-to-End Demonstration"
    )
    parser.add_argument(
        "--scenario",
        choices=["a", "b", "all"],
        default="all",
        help="Choose scenario to run: 'a' (Approved + Validated), 'b' (Pending Approval), or 'all' (default)."
    )
    args = parser.parse_args()

    if args.scenario in ("a", "all"):
        run_scenario_a()

    if args.scenario in ("b", "all"):
        run_scenario_b()

    print("\n" + "=" * 80)
    print("END-TO-END DEMONSTRATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
