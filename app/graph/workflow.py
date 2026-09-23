"""
app/graph/workflow.py
---------------------
Phase 9 — LangGraph Workflow Orchestration.

Builds and compiles the state graph for the Autonomous Data Quality Investigation Agent.
Provides the primary public entry point `run_data_quality_workflow()` and CLI interface.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional, Union

from langgraph.graph import StateGraph, START, END

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional Routing Functions
# ---------------------------------------------------------------------------

def route_after_profiling(state: InvestigationState) -> str:
    """Route after profiling node."""
    if state.get("workflow_status") == "FAILED":
        return "final_report_node"
    return "anomaly_node"


def route_after_anomaly(state: InvestigationState) -> str:
    """Route after anomaly node."""
    if state.get("workflow_status") == "FAILED":
        return "final_report_node"
    return "schema_node"


def route_after_schema(state: InvestigationState) -> str:
    """Route after schema node."""
    if state.get("workflow_status") == "FAILED":
        return "final_report_node"
    return "root_cause_node"


def route_after_root_cause(state: InvestigationState) -> str:
    """
    Route after root cause node.
    If dataset is clean (no issues detected), bypass correction/validation.
    """
    if state.get("workflow_status") == "FAILED":
        return "final_report_node"

    rc_resp = state.get("root_cause_result")
    is_clean = False

    if rc_resp:
        inv_status = getattr(rc_resp, "investigation_status", "")
        primary_cause = getattr(rc_resp, "primary_root_cause", {}).get("cause", "")
        raw_res = getattr(rc_resp, "raw_result", None)

        if inv_status == "no_issues_detected" or primary_cause.lower().startswith("no data quality issues"):
            is_clean = True
        elif raw_res and getattr(raw_res, "investigation_status", "") == "no_issues_detected":
            is_clean = True
        elif not getattr(rc_resp, "supporting_evidence", []):
            prof_resp = state.get("profiling_result")
            anom_resp = state.get("anomaly_result")
            sch_resp = state.get("schema_result")

            nulls = getattr(prof_resp, "quality_metrics", {}).get("missing_cells_total", 0) if prof_resp else 0
            dups = getattr(prof_resp, "quality_metrics", {}).get("duplicate_rows", 0) if prof_resp else 0
            anoms = getattr(anom_resp, "total_anomalies", 0) if anom_resp else 0
            violations = sum(getattr(sch_resp, "summary", {}).values()) if (sch_resp and isinstance(getattr(sch_resp, "summary", {}), dict)) else 0

            if nulls == 0 and dups == 0 and anoms == 0 and violations == 0:
                is_clean = True

    if is_clean:
        return "final_report_node"

    return "correction_node"


def route_after_correction(state: InvestigationState) -> str:
    """Route after correction node."""
    if state.get("workflow_status") == "FAILED":
        return "final_report_node"
    return "approval_node"


def route_after_approval(state: InvestigationState) -> str:
    """
    Route after approval node based on human approval status.
    - APPROVED with corrected_dataset_path -> validation_node
    - APPROVED without corrected_dataset_path -> final_report_node
    - REJECTED or PENDING -> final_report_node
    """
    if state.get("workflow_status") == "FAILED":
        return "final_report_node"

    approval = state.get("approval_status", "PENDING").upper()
    if approval == "APPROVED":
        if state.get("corrected_dataset_path"):
            return "validation_node"
        return "final_report_node"
    return "final_report_node"


# ---------------------------------------------------------------------------
# Workflow Construction
# ---------------------------------------------------------------------------

def create_workflow_graph():
    """
    Build and compile the LangGraph StateGraph for Data Quality Investigation.
    """
    builder = StateGraph(InvestigationState)

    # Register Nodes
    builder.add_node("profiling_node", profiling_node)
    builder.add_node("anomaly_node", anomaly_node)
    builder.add_node("schema_node", schema_node)
    builder.add_node("root_cause_node", root_cause_node)
    builder.add_node("correction_node", correction_node)
    builder.add_node("approval_node", approval_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("final_report_node", final_report_node)

    # Add Entry Edge
    builder.add_edge(START, "profiling_node")

    # Add Conditional Edges
    builder.add_conditional_edges(
        "profiling_node",
        route_after_profiling,
        {"anomaly_node": "anomaly_node", "final_report_node": "final_report_node"}
    )
    builder.add_conditional_edges(
        "anomaly_node",
        route_after_anomaly,
        {"schema_node": "schema_node", "final_report_node": "final_report_node"}
    )
    builder.add_conditional_edges(
        "schema_node",
        route_after_schema,
        {"root_cause_node": "root_cause_node", "final_report_node": "final_report_node"}
    )
    builder.add_conditional_edges(
        "root_cause_node",
        route_after_root_cause,
        {"correction_node": "correction_node", "final_report_node": "final_report_node"}
    )
    builder.add_conditional_edges(
        "correction_node",
        route_after_correction,
        {"approval_node": "approval_node", "final_report_node": "final_report_node"}
    )
    builder.add_conditional_edges(
        "approval_node",
        route_after_approval,
        {"validation_node": "validation_node", "final_report_node": "final_report_node"}
    )

    builder.add_edge("validation_node", "final_report_node")
    builder.add_edge("final_report_node", END)

    return builder.compile()


# Compiled Singleton Graph Instance
data_quality_graph = create_workflow_graph()


# ---------------------------------------------------------------------------
# Public Execution Entry Point
# ---------------------------------------------------------------------------

def run_data_quality_workflow(
    dataset_path: str,
    dataset_name: Optional[str] = None,
    schema_name: Union[str, Dict[str, Any]] = "sales",
    approval_status: str = "PENDING",
    corrected_dataset_path: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Primary Python API to execute the full LangGraph Data Quality Investigation workflow.

    Parameters
    ----------
    dataset_path : str
        Path to input CSV dataset.
    dataset_name : str, optional
        Custom dataset name label.
    schema_name : str or Dict
        Schema definition preset or custom dict (default 'sales').
    approval_status : str
        Human approval status: 'APPROVED', 'REJECTED', or 'PENDING'.
    corrected_dataset_path : str, optional
        Path to post-correction dataset for validation.
    use_llm : bool
        Whether to attempt LLM interpretation/synthesis.

    Returns
    -------
    Dict[str, Any]
        The compiled final workflow state dictionary containing final_report, execution_trace, etc.
    """
    initial_state = create_initial_state(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        schema_name=schema_name,
        approval_status=approval_status,
        corrected_dataset_path=corrected_dataset_path,
        use_llm=use_llm,
    )

    final_state = data_quality_graph.invoke(initial_state)
    return final_state


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Quality Investigation Agent — LangGraph Orchestration CLI"
    )
    parser.add_argument(
        "dataset_path",
        help="Path to raw/problematic CSV dataset to investigate."
    )
    parser.add_argument(
        "--approval-status",
        choices=["APPROVED", "REJECTED", "PENDING"],
        default="PENDING",
        help="Human correction approval status (default: PENDING)."
    )
    parser.add_argument(
        "--corrected-dataset-path",
        default=None,
        help="Optional path to post-correction CSV dataset for Phase 8 validation."
    )
    parser.add_argument(
        "--schema",
        default="sales",
        help="Schema preset or definition file path (default: sales)."
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable Gemini LLM interpretation."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed execution trace to stderr."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    final_state = run_data_quality_workflow(
        dataset_path=args.dataset_path,
        schema_name=args.schema,
        approval_status=args.approval_status,
        corrected_dataset_path=args.corrected_dataset_path,
        use_llm=not args.no_llm,
    )

    report = final_state.get("final_report", {})

    print(json.dumps(report, indent=2, default=str))

    if args.verbose:
        sys.stderr.write("\n=== Execution Trace ===\n")
        for step in final_state.get("execution_trace", []):
            sys.stderr.write(f"{step}\n")


if __name__ == "__main__":
    main()
