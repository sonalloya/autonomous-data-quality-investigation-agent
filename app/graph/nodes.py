"""
app/graph/nodes.py
------------------
Phase 9 — LangGraph Workflow Nodes.

Implements all specialized nodes for the LangGraph orchestration loop.
Each node consumes shared state, executes the corresponding Phase 3–8 agent/tool,
and records its execution step in execution_trace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.agents import (
    run_profiling_agent,
    run_anomaly_agent,
    run_schema_agent,
    run_root_cause_agent,
    run_correction_agent,
    run_validation_agent,
)
from app.tools import (
    profile_dataset,
    detect_anomalies,
    analyze_schema,
    investigate_root_cause,
    generate_correction_plan,
    validate_correction,
    apply_corrections,
)
from app.graph.state import InvestigationState
from app.graph.report import generate_final_report

logger = logging.getLogger(__name__)


def profiling_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute Phase 3 Data Profiling Agent."""
    if state.get("workflow_status") == "FAILED":
        return {}

    dataset_path = state.get("dataset_path", "")
    dataset_name = state.get("dataset_name")
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    try:
        res = run_profiling_agent(
            source=dataset_path,
            dataset_name=dataset_name,
        )
        trace.append({"node": "profiling_node", "status": "completed"})
        return {
            "profiling_result": res,
            "execution_trace": trace,
        }
    except Exception as e:
        logger.error(f"Error in profiling_node: {e}", exc_info=True)
        errors.append({"node": "profiling_node", "error": str(e)})
        trace.append({"node": "profiling_node", "status": "failed", "error": str(e)})
        return {
            "workflow_status": "FAILED",
            "errors": errors,
            "execution_trace": trace,
        }


def anomaly_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute Phase 4 Anomaly Detection Agent."""
    if state.get("workflow_status") == "FAILED":
        return {}

    dataset_path = state.get("dataset_path", "")
    dataset_name = state.get("dataset_name")
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    try:
        res = run_anomaly_agent(
            source=dataset_path,
            dataset_name=dataset_name,
        )
        trace.append({"node": "anomaly_node", "status": "completed"})
        return {
            "anomaly_result": res,
            "execution_trace": trace,
        }
    except Exception as e:
        logger.error(f"Error in anomaly_node: {e}", exc_info=True)
        errors.append({"node": "anomaly_node", "error": str(e)})
        trace.append({"node": "anomaly_node", "status": "failed", "error": str(e)})
        return {
            "workflow_status": "FAILED",
            "errors": errors,
            "execution_trace": trace,
        }


def schema_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute Phase 5 Schema Analysis Agent."""
    if state.get("workflow_status") == "FAILED":
        return {}

    dataset_path = state.get("dataset_path", "")
    schema_name = state.get("schema_name", "sales")
    use_llm = state.get("use_llm", True)
    dataset_name = state.get("dataset_name")
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    try:
        res = run_schema_agent(
            dataset_input=dataset_path,
            schema=schema_name,
            use_llm=use_llm,
            dataset_name=dataset_name,
        )
        trace.append({"node": "schema_node", "status": "completed"})
        return {
            "schema_result": res,
            "execution_trace": trace,
        }
    except Exception as e:
        logger.error(f"Error in schema_node: {e}", exc_info=True)
        errors.append({"node": "schema_node", "error": str(e)})
        trace.append({"node": "schema_node", "status": "failed", "error": str(e)})
        return {
            "workflow_status": "FAILED",
            "errors": errors,
            "execution_trace": trace,
        }


def root_cause_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute Phase 6 Root Cause Investigation Agent."""
    if state.get("workflow_status") == "FAILED":
        return {}

    dataset_path = state.get("dataset_path", "")
    schema_name = state.get("schema_name", "sales")
    use_llm = state.get("use_llm", True)
    dataset_name = state.get("dataset_name")
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    sch_resp = state.get("schema_result")
    sch_res = getattr(sch_resp, "raw_result", None) if sch_resp else None

    try:
        res = run_root_cause_agent(
            dataset_input=dataset_path,
            schema=schema_name,
            use_llm=use_llm,
            dataset_name=dataset_name,
            schema_result=sch_res,
        )
        trace.append({"node": "root_cause_node", "status": "completed"})
        return {
            "root_cause_result": res,
            "execution_trace": trace,
        }
    except Exception as e:
        logger.error(f"Error in root_cause_node: {e}", exc_info=True)
        errors.append({"node": "root_cause_node", "error": str(e)})
        trace.append({"node": "root_cause_node", "status": "failed", "error": str(e)})
        return {
            "workflow_status": "FAILED",
            "errors": errors,
            "execution_trace": trace,
        }


def correction_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute Phase 7 Correction Recommendation Agent."""
    if state.get("workflow_status") == "FAILED":
        return {}

    dataset_path = state.get("dataset_path", "")
    schema_name = state.get("schema_name", "sales")
    use_llm = state.get("use_llm", True)
    dataset_name = state.get("dataset_name")
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    rc_resp = state.get("root_cause_result")
    rc_res = getattr(rc_resp, "raw_result", None) if rc_resp else None

    try:
        res = run_correction_agent(
            dataset_input=dataset_path,
            schema=schema_name,
            use_llm=use_llm,
            dataset_name=dataset_name,
            root_cause_result=rc_res,
        )
        trace.append({"node": "correction_node", "status": "completed"})
        return {
            "correction_result": res,
            "execution_trace": trace,
        }
    except Exception as e:
        logger.error(f"Error in correction_node: {e}", exc_info=True)
        errors.append({"node": "correction_node", "error": str(e)})
        trace.append({"node": "correction_node", "status": "failed", "error": str(e)})
        return {
            "workflow_status": "FAILED",
            "errors": errors,
            "execution_trace": trace,
        }


def approval_node(state: InvestigationState) -> Dict[str, Any]:
    """Process Human Correction Approval Status."""
    if state.get("workflow_status") == "FAILED":
        return {}

    trace = list(state.get("execution_trace", []))
    status_raw = state.get("approval_status", "PENDING").upper()
    corrected_path = state.get("corrected_dataset_path")
    corr_exec_dict = state.get("correction_execution")
    errors = list(state.get("errors", []))

    from datetime import datetime, timezone

    if status_raw == "APPROVED":
        status_str = "approved"
        new_workflow_status = state.get("workflow_status", "RUNNING")
        
        correction_result = state.get("correction_result")
        actions = getattr(correction_result, "actions", []) if correction_result else []
        action_recs = []
        for a in actions:
            action_recs.append({
                "action_type": getattr(a, "action_type", str(a)),
                "description": getattr(a, "description", ""),
                "priority": getattr(a, "priority", "HIGH"),
                "risk": getattr(a, "risk", "LOW"),
                "status": "APPROVED"
            })
            
        approval_record = {
            "status": "APPROVED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "approver": "Human Operator",
            "approved_actions": action_recs
        }

        # If no corrected dataset path was provided or doesn't exist, dynamically generate corrected copy
        dataset_path = state.get("dataset_path", "")
        if (not corrected_path or not Path(corrected_path).is_file()) and dataset_path and Path(dataset_path).is_file():
            schema_name = state.get("schema_name", "sales")
            dataset_name = state.get("dataset_name")

            try:
                _, exec_result = apply_corrections(
                    dataset_input=dataset_path,
                    schema=schema_name,
                    dataset_name=dataset_name,
                )
                corrected_path = exec_result.corrected_dataset_path
                corr_exec_dict = exec_result.to_dict()
                logger.info(f"Dynamically generated corrected dataset copy: {corrected_path}")
            except Exception as exc:
                logger.error(f"Failed to generate corrected dataset copy: {exc}", exc_info=True)
                errors.append({"node": "approval_node", "error": f"Failed to generate corrected dataset: {exc}"})
                trace.append({"node": "approval_node", "status": "failed", "error": str(exc)})
                return {
                    "approval_status": status_raw,
                    "workflow_status": "FAILED",
                    "errors": errors,
                    "execution_trace": trace,
                }
    elif status_raw == "REJECTED":
        status_str = "rejected"
        new_workflow_status = "REJECTED"
    else:
        status_str = "pending"
        new_workflow_status = "PENDING_APPROVAL"

    data_change_summary = corr_exec_dict.get("data_change_summary") if corr_exec_dict else None

    trace.append({"node": "approval_node", "status": status_str})
    return {
        "approval_status": status_raw,
        "workflow_status": new_workflow_status,
        "corrected_dataset_path": corrected_path,
        "correction_execution": corr_exec_dict,
        "approval_record": approval_record if status_raw == "APPROVED" else None,
        "data_change_summary": data_change_summary,
        "execution_trace": trace,
    }


def validation_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute Phase 8 Independent Validation Agent."""
    if state.get("workflow_status") == "FAILED":
        return {}

    corrected_path = state.get("corrected_dataset_path")
    original_path = state.get("dataset_path", "")
    schema_name = state.get("schema_name", "sales")
    use_llm = state.get("use_llm", True)
    dataset_name = state.get("dataset_name")
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    corr_resp = state.get("correction_result")
    plan = getattr(corr_resp, "raw_result", None) if corr_resp else None

    if not corrected_path:
        error_msg = "Validation attempted without a corrected_dataset_path."
        logger.error(error_msg)
        errors.append({"node": "validation_node", "error": error_msg})
        trace.append({"node": "validation_node", "status": "failed", "error": error_msg})
        return {
            "workflow_status": "FAILED",
            "errors": errors,
            "execution_trace": trace,
        }

    try:
        res = run_validation_agent(
            post_correction_input=corrected_path,
            original_dataset_input=original_path,
            correction_plan=plan,
            schema=schema_name,
            use_llm=use_llm,
            dataset_name=dataset_name,
        )
        trace.append({"node": "validation_node", "status": "completed"})
        return {
            "validation_result": res,
            "execution_trace": trace,
        }
    except Exception as e:
        logger.error(f"Error in validation_node: {e}", exc_info=True)
        errors.append({"node": "validation_node", "error": str(e)})
        trace.append({"node": "validation_node", "status": "failed", "error": str(e)})
        return {
            "workflow_status": "FAILED",
            "errors": errors,
            "execution_trace": trace,
        }


def final_report_node(state: InvestigationState) -> Dict[str, Any]:
    """Synthesize final workflow state and generate final report."""
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "final_report_node", "status": "completed"})
    current_status = state.get("workflow_status", "RUNNING")

    # Check if this was a clean data early termination
    rc_resp = state.get("root_cause_result")
    if current_status in ("RUNNING", "APPROVED"):
        if rc_resp:
            inv_status = getattr(rc_resp, "investigation_status", "")
            primary_cause = getattr(rc_resp, "primary_root_cause", {}).get("cause", "")
            if inv_status == "no_issues_detected" or primary_cause == "No data quality issues detected":
                current_status = "NO_CORRECTION_REQUIRED"
            else:
                current_status = "COMPLETED"
        else:
            current_status = "COMPLETED"

    # Merge current_status & trace into state for report generation
    temp_state = dict(state)
    temp_state["workflow_status"] = current_status
    temp_state["execution_trace"] = trace

    report_dict = generate_final_report(temp_state)

    return {
        "final_report": report_dict,
        "workflow_status": current_status,
        "execution_trace": trace,
    }
