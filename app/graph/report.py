"""
app/graph/report.py
-------------------
Phase 9 — Final Workflow Report Compiler.

Generates a structured, evidence-based final workflow report synthesizing outputs
from all executed agents in the LangGraph orchestration loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.graph.state import InvestigationState

logger = logging.getLogger(__name__)


@dataclass
class FinalWorkflowReport:
    """Structured final workflow report for LangGraph orchestration."""
    dataset: Dict[str, Any]
    workflow_status: str
    detected_issues: Dict[str, Any]
    root_cause: Dict[str, Any]
    correction_recommendation: Dict[str, Any]
    approval_status: str
    validation_status: Dict[str, Any]
    execution_trace: List[Dict[str, Any]]
    corrected_dataset: Optional[Dict[str, Any]] = None
    approval_record: Optional[Dict[str, Any]] = None
    data_change_summary: Optional[Dict[str, Any]] = None
    unresolved_issues: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    final_recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_final_report(state: InvestigationState) -> Dict[str, Any]:
    """
    Compile a structured final workflow report from the LangGraph investigation state.
    Factually derives findings strictly from executed diagnostic agents.
    """
    dataset_info = {
        "path": state.get("dataset_path", ""),
        "name": state.get("dataset_name", "sales_dataset"),
    }

    workflow_status = state.get("workflow_status", "UNKNOWN")
    approval_status = state.get("approval_status", "PENDING")
    execution_trace = state.get("execution_trace", [])
    errors = state.get("errors", [])

    # 1. Detected Issues Summary
    prof_resp = state.get("profiling_result")
    anom_resp = state.get("anomaly_result")
    sch_resp = state.get("schema_result")

    total_missing_values = 0
    total_duplicate_rows = 0
    anomalous_dates_count = 0
    schema_violations_count = 0

    if prof_resp:
        q_metrics = getattr(prof_resp, "quality_metrics", {})
        total_missing_values = q_metrics.get("missing_cells_total", 0)
        total_duplicate_rows = q_metrics.get("duplicate_rows", 0)

    if anom_resp:
        anomalous_dates_count = getattr(anom_resp, "total_anomalies", 0)

    if sch_resp:
        sch_summary = getattr(sch_resp, "summary", {})
        schema_violations_count = sum(sch_summary.values()) if isinstance(sch_summary, dict) else 0

    detected_issues = {
        "missing_values_count": total_missing_values,
        "duplicate_rows_count": total_duplicate_rows,
        "anomalous_dates_count": anomalous_dates_count,
        "schema_violations_count": schema_violations_count,
        "profiling_summary": getattr(prof_resp, "summary", "Not executed"),
        "anomaly_summary": getattr(anom_resp, "summary", "Not executed"),
        "schema_summary": getattr(sch_resp, "llm_interpretation", None) or ("Valid schema" if getattr(sch_resp, "schema_valid", False) else "Schema violations detected") if sch_resp else "Not executed",
    }

    # 2. Root Cause Summary
    rc_resp = state.get("root_cause_result")
    root_cause = {
        "investigation_status": "not_executed",
        "primary_cause": "Not investigated",
        "confidence": "NONE",
        "narrative": "Not executed",
    }
    unresolved_questions: List[str] = []

    if rc_resp:
        rc_res = getattr(rc_resp, "raw_result", None)
        primary = getattr(rc_resp, "primary_root_cause", {})
        root_cause = {
            "investigation_status": getattr(rc_resp, "investigation_status", "unknown"),
            "primary_cause": primary.get("cause", "Undetermined"),
            "confidence": primary.get("confidence", "LOW"),
            "score": primary.get("score", 0.0),
            "reasoning": primary.get("reasoning", ""),
            "narrative": getattr(rc_resp, "llm_interpretation", "") or primary.get("reasoning", ""),
        }
        if rc_res and hasattr(rc_res, "unresolved_questions"):
            unresolved_questions = list(rc_res.unresolved_questions)
        elif hasattr(rc_resp, "unresolved_questions"):
            unresolved_questions = list(rc_resp.unresolved_questions)

    # 3. Correction Recommendation Summary
    corr_resp = state.get("correction_result")
    correction_rec = {
        "status": "not_executed",
        "total_actions": 0,
        "summary": "Not executed",
        "actions": [],
    }

    if corr_resp:
        recs = getattr(corr_resp, "recommendations", [])
        correction_rec = {
            "status": getattr(corr_resp, "status", "recommended"),
            "total_actions": len(recs),
            "summary": getattr(corr_resp, "summary", ""),
            "actions": recs,
            "recommendation_narrative": getattr(corr_resp, "llm_interpretation", "") or getattr(corr_resp, "summary", ""),
        }

    # 4. Validation Summary
    val_resp = state.get("validation_result")
    default_verdict = "NOT_APPLICABLE" if workflow_status == "NO_CORRECTION_REQUIRED" else "NOT_VALIDATED"
    validation_status = {
        "status": "not_executed",
        "verdict": default_verdict,
        "passed_checks": 0,
        "failed_checks": 0,
        "partial_checks": 0,
        "checks": [],
        "before_after_comparison": {},
        "corrected_dataset_path": state.get("corrected_dataset_path"),
        "issues_resolved": [],
        "issues_unresolved": [],
        "new_issues_introduced": [],
        "summary": "Validation was not performed." if workflow_status != "NO_CORRECTION_REQUIRED" else "Validation not required for clean dataset.",
    }

    if val_resp:
        raw_val = getattr(val_resp, "raw_result", None)
        v_status = getattr(val_resp, "validation_status", "UNKNOWN")
        passed_c = getattr(val_resp, "passed_checks", 0)
        failed_c = getattr(val_resp, "failed_checks", 0)
        partial_c = getattr(val_resp, "partial_checks", 0)
        issues_res = []
        issues_unres = []
        new_issues = []
        if raw_val:
            passed_c = getattr(raw_val, "passed_checks", passed_c)
            failed_c = getattr(raw_val, "failed_checks", failed_c)
            partial_c = getattr(raw_val, "partial_checks", partial_c)
            issues_res = getattr(raw_val, "issues_resolved", [])
            issues_unres = getattr(raw_val, "issues_unresolved", [])
            new_issues = getattr(raw_val, "new_issues_introduced", [])

        checks = getattr(val_resp, "checks", []) or (getattr(raw_val, "checks", []) if raw_val else [])
        before_after = getattr(val_resp, "before_after_comparison", {}) or (getattr(raw_val, "before_after_comparison", {}) if raw_val else {})

        validation_status = {
            "status": "completed",
            "verdict": v_status,
            "passed_checks": passed_c,
            "failed_checks": failed_c,
            "partial_checks": partial_c,
            "checks": checks,
            "before_after_comparison": before_after,
            "corrected_dataset_path": state.get("corrected_dataset_path"),
            "issues_resolved": issues_res,
            "issues_unresolved": issues_unres,
            "new_issues_introduced": new_issues,
            "summary": getattr(val_resp, "llm_interpretation", "") or f"Validation completed with status: {v_status}",
        }
        unresolved_val = getattr(val_resp, "unresolved_issues", [])
        for issue in unresolved_val:
            if issue not in unresolved_questions:
                unresolved_questions.append(issue)

    # 5. Synthesis Final Recommendation
    if workflow_status == "NO_CORRECTION_REQUIRED":
        final_rec = "Dataset was verified as clean. No data quality corrections are required."
    elif workflow_status == "REJECTED":
        final_rec = "Correction plan was REJECTED by human operator. No validation or data modification performed."
    elif workflow_status == "PENDING_APPROVAL":
        final_rec = "Correction plan generated. Awaiting human approval before applying post-correction validation."
    elif workflow_status == "COMPLETED":
        if val_resp:
            verdict = getattr(val_resp, "validation_status", "COMPLETED")
            final_rec = f"Workflow completed successfully. Post-correction validation verdict: {verdict}."
        else:
            final_rec = "Workflow completed successfully."
    elif workflow_status == "FAILED":
        err_msgs = [e.get("error", "Unknown error") for e in errors]
        final_rec = f"Workflow failed due to errors: {err_msgs}."
    else:
        final_rec = f"Workflow status: {workflow_status}."

    # 6. Corrected Dataset Summary
    corr_exec = state.get("correction_execution") or {}
    corrected_path = state.get("corrected_dataset_path")
    val_verdict = validation_status.get("verdict", "NOT_VALIDATED")

    corrections_applied_count = corr_exec.get("corrections_applied", 0)
    if corrections_applied_count == 0 and val_resp:
        corrections_applied_count = getattr(val_resp, "passed_checks", 0)

    issues_remaining_count = corr_exec.get("issues_remaining", len(unresolved_questions))
    can_download = (approval_status == "APPROVED" and corrected_path is not None and val_verdict in ("PASSED", "PARTIAL"))

    corrected_dataset_info = None
    if corrected_path or corr_exec:
        orig_filename = Path(state.get("dataset_path", "")).name if state.get("dataset_path") else ""
        corr_filename = Path(corrected_path).name if corrected_path else ""
        corrected_dataset_info = {
            "original_dataset_path": state.get("dataset_path", ""),
            "original_filename": orig_filename,
            "corrected_dataset_path": corrected_path,
            "corrected_filename": corr_filename,
            "corrections_applied": corrections_applied_count,
            "issues_remaining": issues_remaining_count,
            "validation_verdict": val_verdict,
            "download_available": can_download,
            "download_url": f"/download/corrected?filename={corr_filename}" if (can_download and corr_filename) else None,
            "summary": corr_exec.get("summary", ""),
        }

    report = FinalWorkflowReport(
        dataset=dataset_info,
        workflow_status=workflow_status,
        detected_issues=detected_issues,
        root_cause=root_cause,
        correction_recommendation=correction_rec,
        approval_status=approval_status,
        validation_status=validation_status,
        execution_trace=execution_trace,
        corrected_dataset=corrected_dataset_info,
        approval_record=state.get("approval_record"),
        data_change_summary=state.get("data_change_summary"),
        unresolved_issues=unresolved_questions,
        errors=errors,
        final_recommendation=final_rec,
    )

    return report.to_dict()


def format_final_report_text(report: Dict[str, Any]) -> str:
    """
    Format the final report dictionary into a clean, human-readable terminal text view.
    Conforms directly to Phase 10 & 12 specification requirements.
    """
    dataset = report.get("dataset", {})
    ds_name = dataset.get("name", "Unknown Dataset")
    ds_path = dataset.get("path", "")
    wf_status = report.get("workflow_status", "UNKNOWN")

    findings = report.get("detected_issues", {})
    missing_val = findings.get("missing_values_count", 0)
    dup_recs = findings.get("duplicate_rows_count", 0)
    sch_issues = findings.get("schema_violations_count", 0)
    anomalies = findings.get("anomalous_dates_count", 0)

    rc = report.get("root_cause", {})
    primary_cause = rc.get("primary_cause", "None")
    confidence = rc.get("confidence", "NONE")
    evidence = rc.get("reasoning") or rc.get("narrative") or "None documented"

    corr = report.get("correction_recommendation", {})
    actions = corr.get("actions", [])
    if actions:
        top_action = actions[0] if isinstance(actions[0], dict) else {"description": str(actions[0]), "priority": "HIGH"}
        action_desc = top_action.get("description") or top_action.get("action_type") or top_action.get("target_column") or "Automated remediation plan"
        priority = top_action.get("priority", "HIGH")
    else:
        action_desc = "No correction actions required" if wf_status == "NO_CORRECTION_REQUIRED" else "None"
        priority = "N/A"

    appr_status = report.get("approval_status", "PENDING")

    val = report.get("validation_status", {})
    val_status = val.get("verdict") or val.get("status", "NOT_VALIDATED")
    passed_checks = val.get("passed_checks", 0)
    failed_checks = val.get("failed_checks", 0)
    unresolved = report.get("unresolved_issues", [])

    corr_ds = report.get("corrected_dataset") or {}

    trace = report.get("execution_trace", [])
    trace_map: Dict[str, str] = {}
    for step in trace:
        node_name = step.get("node", "").replace("_node", "").capitalize()
        status_name = step.get("status", "executed")
        trace_map[node_name] = status_name

    trace_order = ["Profiling", "Anomaly", "Schema", "Root_cause", "Correction", "Approval", "Validation", "Final_report"]
    trace_lines = []
    for node_key in trace_order:
        disp_name = node_key.replace("_", " ")
        st = trace_map.get(node_key) or trace_map.get(disp_name)
        if st:
            trace_lines.append(f"- {disp_name}: {st}")
        else:
            trace_lines.append(f"- {disp_name}: skipped")

    final_rec = report.get("final_recommendation", "")

    lines = [
        "-----------------------------------",
        "AUTONOMOUS DATA QUALITY INVESTIGATION",
        "-----------------------------------",
        "",
        f"Dataset: {ds_name} ({ds_path})",
        f"Workflow Status: {wf_status}",
        "",
        "DATA QUALITY FINDINGS",
        f"- Missing values: {missing_val}",
        f"- Duplicate records: {dup_recs}",
        f"- Schema issues: {sch_issues}",
        f"- Anomalies: {anomalies}",
        "",
        "ROOT CAUSE",
        f"- Primary cause: {primary_cause}",
        f"- Confidence: {confidence}",
        f"- Evidence: {evidence}",
        "",
        "CORRECTION RECOMMENDATION",
        f"- Action: {action_desc}",
        f"- Priority: {priority}",
        "- Approval requirement: Human approval required before data remediation",
        "",
        "APPROVAL",
        f"- Status: {appr_status}",
    ]

    if corr_ds and corr_ds.get("corrected_filename"):
        lines.extend([
            "",
            "CORRECTED DATASET",
            f"- Original Dataset: {corr_ds.get('original_filename', '')}",
            f"- Corrected Dataset: {corr_ds.get('corrected_filename', '')}",
            f"- Corrections Applied: {corr_ds.get('corrections_applied', 0)}",
            f"- Unresolved Issues: {corr_ds.get('issues_remaining', 0)}",
            f"- Validation: {corr_ds.get('validation_verdict', 'NOT_VALIDATED')}",
            f"- Download Available: {'Yes' if corr_ds.get('download_available') else 'No'}",
        ])

    lines.extend([
        "",
        "VALIDATION",
        f"- Status: {val_status}",
        f"- Passed checks: {passed_checks}",
        f"- Failed checks: {failed_checks}",
        f"- Unresolved issues: {len(unresolved)} ({', '.join(unresolved) if unresolved else 'None'})",
        "",
        "EXECUTION TRACE",
    ])
    lines.extend(trace_lines)
    lines.extend([
        "",
        "FINAL RECOMMENDATION",
        final_rec,
        "-----------------------------------",
    ])

    return "\n".join(lines)

