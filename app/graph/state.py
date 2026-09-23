"""
app/graph/state.py
------------------
Phase 9 — LangGraph Shared Workflow State.

Defines InvestigationState TypedDict used across LangGraph nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TypedDict

from app.agents.profiling_agent import AgentProfilingResponse
from app.agents.anomaly_agent import AgentAnomalyResponse
from app.agents.schema_agent import AgentSchemaResponse
from app.agents.root_cause_agent import AgentRootCauseResponse
from app.agents.correction_agent import AgentCorrectionResponse
from app.agents.validation_agent import AgentValidationResponse


class InvestigationState(TypedDict, total=False):
    """
    Shared LangGraph workflow state for Autonomous Data Quality Investigation Agent.
    """
    # Inputs
    dataset_path: str
    dataset_name: Optional[str]
    schema_name: Union[str, Dict[str, Any]]
    corrected_dataset_path: Optional[str]
    approval_status: str  # "APPROVED", "REJECTED", "PENDING", "NOT_REQUIRED"
    use_llm: bool

    # Diagnostics Agent Outputs
    profiling_result: Optional[AgentProfilingResponse]
    anomaly_result: Optional[AgentAnomalyResponse]
    schema_result: Optional[AgentSchemaResponse]
    root_cause_result: Optional[AgentRootCauseResponse]
    correction_result: Optional[AgentCorrectionResponse]
    correction_execution: Optional[Dict[str, Any]]
    validation_result: Optional[AgentValidationResponse]

    # Final Report & Audit Outputs
    final_report: Optional[Dict[str, Any]]
    approval_record: Optional[Dict[str, Any]]
    data_change_summary: Optional[Dict[str, Any]]

    # Metadata & Control
    workflow_status: str  # "RUNNING", "COMPLETED", "NO_CORRECTION_REQUIRED", "REJECTED", "PENDING_APPROVAL", "FAILED"
    errors: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]


def create_initial_state(
    dataset_path: str,
    dataset_name: Optional[str] = None,
    schema_name: Union[str, Dict[str, Any]] = "sales",
    approval_status: str = "PENDING",
    corrected_dataset_path: Optional[str] = None,
    use_llm: bool = True,
) -> InvestigationState:
    """
    Create a clean, initialized InvestigationState dictionary.
    """
    return {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "schema_name": schema_name,
        "approval_status": approval_status,
        "corrected_dataset_path": corrected_dataset_path,
        "use_llm": use_llm,
        "profiling_result": None,
        "anomaly_result": None,
        "schema_result": None,
        "root_cause_result": None,
        "correction_result": None,
        "correction_execution": None,
        "validation_result": None,
        "final_report": None,
        "approval_record": None,
        "data_change_summary": None,
        "workflow_status": "RUNNING",
        "errors": [],
        "execution_trace": [],
    }
