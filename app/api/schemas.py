"""
app/api/schemas.py
-------------------
Phase 10 — Pydantic Request & Response Schemas for FastAPI Endpoints.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ApprovalStatusEnum(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Current system operational status.")
    service: str = Field(default="Autonomous Data Quality Investigation Agent", description="Service title.")
    environment: str = Field(..., description="Active runtime environment.")
    phase: str = Field(default="Phase 10 — Final End-to-End Demo, API & Project Hardening", description="Current development phase.")
    agents_active: bool = Field(default=True, description="Whether data quality agents are online and active.")
    llm_configured: bool = Field(default=False, description="Whether LLM API keys are configured.")


class InvestigationRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to CSV dataset file for data quality investigation.")
    approval_status: Optional[str] = Field(
        default="PENDING",
        description="Human approval decision: 'PENDING', 'APPROVED', or 'REJECTED'."
    )
    corrected_dataset_path: Optional[str] = Field(
        default=None,
        description="Optional path to post-remediation CSV dataset for Phase 8 validation."
    )
    schema_name: Optional[str] = Field(
        default="sales",
        description="Schema definition preset or name (default 'sales')."
    )
    use_llm: Optional[bool] = Field(
        default=False,
        description="Enable Gemini LLM interpretation and synthesis (default False)."
    )

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("dataset_path must not be empty.")
        path = Path(v.strip())
        if not path.is_file():
            raise ValueError(f"Dataset file does not exist: '{v}'")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Unsupported file type '{path.suffix}'. Only .csv files are supported.")
        return str(path)

    @field_validator("approval_status")
    @classmethod
    def validate_approval_status(cls, v: Optional[str]) -> str:
        if not v:
            return "PENDING"
        v_upper = v.strip().upper()
        if v_upper not in {"APPROVED", "REJECTED", "PENDING"}:
            raise ValueError(f"Invalid approval_status: '{v}'. Must be one of 'APPROVED', 'REJECTED', or 'PENDING'.")
        return v_upper

    @field_validator("corrected_dataset_path")
    @classmethod
    def validate_corrected_dataset_path(cls, v: Optional[str]) -> Optional[str]:
        if not v or not v.strip():
            return None
        path = Path(v.strip())
        if not path.is_file():
            raise ValueError(f"Corrected dataset file does not exist: '{v}'")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Unsupported corrected file type '{path.suffix}'. Only .csv files are supported.")
        return str(path)


class InvestigationResponse(BaseModel):
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
    unresolved_issues: List[str] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    final_recommendation: str = ""


class UploadResponse(BaseModel):
    status: str = Field(default="success", description="Upload operation status.")
    filename: str = Field(..., description="Sanitized saved filename on server.")
    dataset_path: str = Field(..., description="Server-side relative path to uploaded dataset for investigation.")
    size_bytes: int = Field(..., description="Total size in bytes of the uploaded file.")
    message: str = Field(default="File uploaded successfully.", description="Status message.")
