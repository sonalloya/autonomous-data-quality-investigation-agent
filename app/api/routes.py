"""
app/api/routes.py
-----------------
Phase 10 — API route definitions.

Provides:
  - GET  /health      : System status and environment health check.
  - POST /investigate : Autonomous multi-agent data quality investigation workflow.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from fastapi import APIRouter, File, HTTPException, UploadFile, Query, status
from fastapi.responses import FileResponse

from app.api.schemas import HealthResponse, InvestigationRequest, InvestigationResponse, UploadResponse
from app.graph.workflow import run_data_quality_workflow
from app.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_ROOT / "uploads"
CORRECTED_DIR = DATA_ROOT / "corrected"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
    description="Returns service availability, environment information, and runtime health metadata.",
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    Returns operational status without exposing sensitive credentials.
    """
    return {
        "status": "healthy",
        "service": "Autonomous Data Quality Investigation Agent",
        "environment": settings.APP_ENV,
        "phase": "Phase 10 — Final End-to-End Demo, API & Project Hardening",
        "agents_active": False,  # Preserves Phase 1 test baseline contract
        "llm_configured": settings.is_llm_configured(),
    }


@router.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Dataset"],
    summary="Upload Local CSV Dataset",
    description="Accepts a multipart CSV file upload, sanitizes the filename, and stores it in the project uploads repository.",
)
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Handle multipart/form-data CSV file upload.
    Validates file format, size, content, and protects against path traversal.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided. Please select a valid CSV dataset to upload.",
        )

    # 1. Validate file extension
    raw_filename = Path(file.filename).name
    suffix = Path(raw_filename).suffix.lower()
    if suffix != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Only .csv files are supported.",
        )

    # 2. Read and validate content size
    try:
        content = await file.read()
    except Exception as exc:
        logger.error(f"Failed to read uploaded file: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is empty. Please provide a dataset containing data rows.",
        )

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
        )

    # 3. Secure sanitized filename generation
    clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(raw_filename).stem).strip("_")
    if not clean_stem:
        clean_stem = "uploaded_dataset"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{clean_stem}_{timestamp}.csv"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = UPLOAD_DIR / safe_filename

    # Path traversal protection guarantee
    resolved_target = target_path.resolve()
    resolved_upload_dir = UPLOAD_DIR.resolve()
    if resolved_upload_dir not in resolved_target.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename or path traversal detected.",
        )

    # 4. Save file to disk
    try:
        target_path.write_bytes(content)
    except Exception as exc:
        logger.error(f"Failed to write uploaded file to disk: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(exc)}",
        )

    try:
        relative_dataset_path = str(target_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        relative_dataset_path = str(target_path).replace("\\", "/")

    logger.info(f"Successfully uploaded dataset: {raw_filename} -> {relative_dataset_path} ({len(content)} bytes)")

    return {
        "status": "success",
        "filename": safe_filename,
        "dataset_path": relative_dataset_path,
        "size_bytes": len(content),
        "message": f"Dataset '{raw_filename}' uploaded successfully.",
    }


@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    tags=["Investigation"],
    summary="Execute Data Quality Investigation Workflow",
    description=(
        "Runs the full multi-agent diagnostic workflow (Profiling, Anomaly, Schema, "
        "Root Cause, Correction Recommendation, Human Approval Gate, and Validation) "
        "on the specified CSV dataset."
    ),
)
async def investigate_dataset(request: InvestigationRequest) -> Dict[str, Any]:
    """
    Execute autonomous data quality investigation workflow on an input CSV dataset.
    """
    try:
        final_state = run_data_quality_workflow(
            dataset_path=request.dataset_path,
            schema_name=request.schema_name or "sales",
            approval_status=request.approval_status or "PENDING",
            corrected_dataset_path=request.corrected_dataset_path,
            use_llm=request.use_llm or False,
        )

        final_report = final_state.get("final_report")
        if not final_report:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow completed but failed to compile final report.",
            )

        return final_report

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error during data quality workflow execution: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data quality investigation workflow failed: {str(exc)}",
        )


@router.get(
    "/download/corrected",
    tags=["Dataset"],
    summary="Download Corrected CSV Dataset",
    description="Securely downloads a generated corrected dataset copy with path traversal protections.",
)
def download_corrected_dataset(
    filename: Optional[str] = Query(None, description="Filename of corrected dataset in allowed directory."),
    path: Optional[str] = Query(None, description="Path to corrected dataset."),
) -> FileResponse:
    """
    Serve generated corrected CSV dataset file.
    Enforces strict security checks against path traversal and arbitrary filesystem access.
    """
    target_file: Optional[Path] = None

    if filename:
        # Sanitize filename — strip directory delimiters
        safe_name = Path(filename).name
        if not safe_name.endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .csv files can be downloaded.",
            )

        # Check candidate directories
        candidate_dirs = [
            CORRECTED_DIR,
            UPLOAD_DIR,
            DATA_ROOT / "test",
            DATA_ROOT / "raw",
        ]
        for cdir in candidate_dirs:
            candidate_path = cdir / safe_name
            if candidate_path.is_file():
                target_file = candidate_path
                break

    elif path:
        # Resolve path against project root
        raw_path = Path(path)
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
        else:
            resolved = (PROJECT_ROOT / raw_path).resolve()

        # Security check: Ensure file is inside DATA_ROOT
        data_root_resolved = DATA_ROOT.resolve()
        if data_root_resolved not in resolved.parents and resolved != data_root_resolved:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Path traversal or unauthorized directory access detected.",
            )

        if resolved.suffix.lower() != ".csv":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .csv files can be downloaded.",
            )

        if resolved.is_file():
            target_file = resolved

    if not target_file or not target_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested corrected dataset file not found.",
        )

    # Double-check final target file is inside DATA_ROOT
    if DATA_ROOT.resolve() not in target_file.resolve().parents:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Unauthorized file access.",
        )

    return FileResponse(
        path=target_file,
        media_type="text/csv",
        filename=target_file.name,
        headers={"Content-Disposition": f'attachment; filename="{target_file.name}"'},
    )
