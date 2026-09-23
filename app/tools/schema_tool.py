"""
app/tools/schema_tool.py
------------------------
Phase 5 — Schema Analysis Tool.

A deterministic tool for validating a Pandas DataFrame against an expected dataset schema.

Checks performed:
  1. Missing columns
  2. Unexpected / extra columns
  3. Logical data-type mismatches
  4. Date/time format inconsistencies & parse failures
  5. Nullability violations (nulls in non-nullable fields)
  6. Basic schema constraints (min/max range, allowed categorical values)

Outputs a structured dataclass / dictionary (SchemaAnalysisResult).
Does NOT make LLM calls, invent statistics, or attempt root-cause analysis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.config.schema_config import ColumnSchema, SchemaDefinition, load_schema, get_sales_schema

logger = logging.getLogger(__name__)


@dataclass
class SchemaFinding:
    """Represents an individual schema violation or issue finding."""
    issue_type: str  # 'missing_column', 'unexpected_column', 'type_mismatch', 'format_issue', 'nullability_violation', 'constraint_violation'
    column: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "column": self.column,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass
class SchemaAnalysisResult:
    """Structured container for full schema analysis findings."""
    dataset_name: str
    total_rows: int
    total_columns: int
    schema_valid: bool
    expected_columns: List[str]
    actual_columns: List[str]
    missing_columns: List[Dict[str, Any]] = field(default_factory=list)
    unexpected_columns: List[Dict[str, Any]] = field(default_factory=list)
    type_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    format_issues: List[Dict[str, Any]] = field(default_factory=list)
    nullability_violations: List[Dict[str, Any]] = field(default_factory=list)
    constraint_violations: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "schema_valid": self.schema_valid,
            "expected_columns": self.expected_columns,
            "actual_columns": self.actual_columns,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "type_mismatches": self.type_mismatches,
            "format_issues": self.format_issues,
            "nullability_violations": self.nullability_violations,
            "constraint_violations": self.constraint_violations,
            "summary": self.summary,
        }


def _infer_actual_logical_type(series: pd.Series) -> str:
    """Infer logical type of a Pandas Series."""
    dtype_str = str(series.dtype).lower()

    if "int" in dtype_str:
        return "integer"
    elif "float" in dtype_str:
        # Check if float series contains integer-like values
        return "float"
    elif "bool" in dtype_str:
        return "boolean"
    elif "datetime" in dtype_str:
        return "datetime"
    else:
        # Object / string dtype — examine non-null values
        non_nulls = series.dropna()
        if len(non_nulls) == 0:
            return "string"

        # Check if all non-null values can be numeric or datetime
        sample = non_nulls.astype(str).head(100)

        # Check datetime
        try:
            pd.to_datetime(sample, errors="raise")
            return "datetime"
        except (ValueError, TypeError, OverflowError):
            pass

        return "string"


def analyze_schema(
    df: pd.DataFrame,
    schema: Union[SchemaDefinition, Dict[str, Any], str, Path] = "sales",
    dataset_name: str = "dataset",
) -> SchemaAnalysisResult:
    """
    Perform deterministic schema analysis on a DataFrame against an expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    schema : SchemaDefinition, Dict, str, or Path
        The expected schema definition or key/file path.
    dataset_name : str
        Name or identifier of the dataset.

    Returns
    -------
    SchemaAnalysisResult
        Structured results detailing missing columns, extra columns, type mismatches,
        date format issues, nullability violations, and constraint violations.
    """
    schema_def = load_schema(schema)
    expected_cols = [c.name for c in schema_def.columns]
    actual_cols = list(df.columns)
    total_rows = len(df)
    total_cols = len(df.columns)

    missing_cols_list: List[Dict[str, Any]] = []
    unexpected_cols_list: List[Dict[str, Any]] = []
    type_mismatches_list: List[Dict[str, Any]] = []
    format_issues_list: List[Dict[str, Any]] = []
    nullability_violations_list: List[Dict[str, Any]] = []
    constraint_violations_list: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Missing Column Detection
    # -------------------------------------------------------------------------
    for col_schema in schema_def.columns:
        if col_schema.name not in actual_cols:
            severity = "CRITICAL" if col_schema.required and col_schema.name == "transaction_id" else "HIGH"
            finding = SchemaFinding(
                issue_type="missing_column",
                column=col_schema.name,
                severity=severity,
                description=f"Required column '{col_schema.name}' is missing from the dataset.",
                evidence={
                    "expected_logical_type": col_schema.logical_type,
                    "required": col_schema.required,
                },
            )
            missing_cols_list.append(finding.to_dict())

    # -------------------------------------------------------------------------
    # 2. Unexpected / Extra Column Detection
    # -------------------------------------------------------------------------
    for col in actual_cols:
        if col not in expected_cols:
            finding = SchemaFinding(
                issue_type="unexpected_column",
                column=col,
                severity="LOW",
                description=f"Column '{col}' exists in actual data but is not defined in the expected schema.",
                evidence={
                    "actual_pandas_dtype": str(df[col].dtype),
                },
            )
            unexpected_cols_list.append(finding.to_dict())

    # -------------------------------------------------------------------------
    # 3. Data Type, Format, Nullability, and Constraint Checks (for present columns)
    # -------------------------------------------------------------------------
    for col_schema in schema_def.columns:
        col_name = col_schema.name
        if col_name not in df.columns:
            continue

        series = df[col_name]
        dtype_str = str(series.dtype).lower()
        expected_type = col_schema.logical_type
        actual_type = _infer_actual_logical_type(series)

        # ---------------------------------------------------------------------
        # 3A. Type Mismatch Analysis
        # ---------------------------------------------------------------------
        is_type_mismatch = False
        mismatch_evidence: Dict[str, Any] = {}

        if expected_type == "integer":
            if "int" in dtype_str:
                # Compatible integer
                pass
            elif "float" in dtype_str:
                # Check if floats contain non-integers
                non_null = series.dropna()
                non_integers = non_null[non_null != non_null.round()]
                if len(non_integers) > 0:
                    is_type_mismatch = True
                    mismatch_evidence = {
                        "expected_type": "integer",
                        "actual_pandas_dtype": dtype_str,
                        "non_integer_float_count": int(len(non_integers)),
                        "sample_non_integers": non_integers.head(5).tolist(),
                    }
            elif dtype_str in ("object", "string", "category"):
                non_null_str = series.dropna().astype(str)
                # Check for string suffixes like 'units', '$', etc.
                non_digit_pattern = re.compile(r"[^\d\-+]")
                dirty_strings = [s for s in non_null_str if non_digit_pattern.search(s.strip())]
                is_type_mismatch = True
                mismatch_evidence = {
                    "expected_type": "integer",
                    "actual_pandas_dtype": dtype_str,
                    "dirty_string_count": len(dirty_strings),
                    "sample_invalid_strings": dirty_strings[:5],
                }

        elif expected_type == "float":
            if "float" in dtype_str or "int" in dtype_str:
                # Compatible numeric
                pass
            elif dtype_str in ("object", "string", "category"):
                non_null_str = series.dropna().astype(str)
                unparseable: List[str] = []
                for val in non_null_str:
                    try:
                        float(val.replace("$", "").replace(",", "").strip())
                    except ValueError:
                        unparseable.append(val)
                is_type_mismatch = True
                mismatch_evidence = {
                    "expected_type": "float",
                    "actual_pandas_dtype": dtype_str,
                    "unparseable_count": len(unparseable),
                    "sample_unparseable": unparseable[:5],
                }

        elif expected_type == "datetime":
            # Datetime string vs datetime object check
            if "datetime" not in dtype_str and dtype_str not in ("object", "string", "category"):
                is_type_mismatch = True
                mismatch_evidence = {
                    "expected_type": "datetime",
                    "actual_pandas_dtype": dtype_str,
                }

        if is_type_mismatch:
            finding = SchemaFinding(
                issue_type="type_mismatch",
                column=col_name,
                severity="HIGH",
                description=f"Column '{col_name}' expects logical type '{expected_type}', but actual dtype is '{dtype_str}'.",
                evidence=mismatch_evidence,
            )
            type_mismatches_list.append(finding.to_dict())

        # ---------------------------------------------------------------------
        # 3B. Date / Time Format Analysis
        # ---------------------------------------------------------------------
        if expected_type == "datetime":
            non_null_dates = series.dropna()
            total_non_null = len(non_null_dates)

            if total_non_null > 0:
                invalid_dates: List[str] = []
                format_inconsistencies: List[str] = []
                expected_fmt = col_schema.expected_format or "%Y-%m-%d"

                for val in non_null_dates:
                    s_val = str(val).strip()
                    # Try expected format first (ISO YYYY-MM-DD)
                    try:
                        pd.to_datetime(s_val, format=expected_fmt, errors="raise")
                    except Exception:
                        # Failed expected format — try flexible parsing
                        try:
                            pd.to_datetime(s_val, errors="raise")
                            # Parsed with alternative format (e.g. MM/DD/YYYY)
                            format_inconsistencies.append(s_val)
                        except Exception:
                            # Completely unparseable date
                            invalid_dates.append(s_val)

                if len(invalid_dates) > 0 or len(format_inconsistencies) > 0:
                    invalid_pct = round(len(invalid_dates) / max(1, total_rows), 4)
                    inconsistent_pct = round(len(format_inconsistencies) / max(1, total_rows), 4)

                    finding = SchemaFinding(
                        issue_type="format_issue",
                        column=col_name,
                        severity="HIGH" if len(invalid_dates) > 0 else "MEDIUM",
                        description=(
                            f"Column '{col_name}' has {len(invalid_dates)} unparseable dates and "
                            f"{len(format_inconsistencies)} format inconsistencies."
                        ),
                        evidence={
                            "expected_format": expected_fmt,
                            "invalid_count": len(invalid_dates),
                            "invalid_percentage": invalid_pct,
                            "sample_invalid_values": invalid_dates[:5],
                            "format_inconsistency_count": len(format_inconsistencies),
                            "format_inconsistency_percentage": inconsistent_pct,
                            "sample_inconsistent_values": format_inconsistencies[:5],
                        },
                    )
                    format_issues_list.append(finding.to_dict())

        # ---------------------------------------------------------------------
        # 3C. Nullability Violations
        # ---------------------------------------------------------------------
        if not col_schema.nullable:
            null_mask = series.isna() | (series.astype(str).str.strip().isin(["", "nan", "null", "none", "n/a"]))
            null_count = int(null_mask.sum())

            if null_count > 0:
                null_pct = round(null_count / max(1, total_rows), 4)
                severity = "CRITICAL" if col_name == "transaction_id" else "HIGH"
                finding = SchemaFinding(
                    issue_type="nullability_violation",
                    column=col_name,
                    severity=severity,
                    description=f"Column '{col_name}' is non-nullable, but contains {null_count} null/missing values.",
                    evidence={
                        "null_count": null_count,
                        "null_percentage": null_pct,
                        "required": col_schema.required,
                        "nullable": col_schema.nullable,
                    },
                )
                nullability_violations_list.append(finding.to_dict())

        # ---------------------------------------------------------------------
        # 3D. Constraint Violations
        # ---------------------------------------------------------------------
        # Convert to numeric if expected numeric
        numeric_series: Optional[pd.Series] = None
        if expected_type in ("integer", "float"):
            # Try coercing to numeric cleanly
            if dtype_str in ("object", "string", "category"):
                # Clean strings like '3 units' or '$12' for constraint evaluation
                cleaned = series.astype(str).str.extract(r"([\-+]?\d*\.?\d+)")[0]
                numeric_series = pd.to_numeric(cleaned, errors="coerce")
            else:
                numeric_series = pd.to_numeric(series, errors="coerce")

        if numeric_series is not None:
            # Check min_value_exclusive (e.g. quantity > 0, unit_price > 0)
            if col_schema.min_value_exclusive is not None:
                viol_mask = numeric_series <= col_schema.min_value_exclusive
                viol_count = int(viol_mask.dropna().sum())
                if viol_count > 0:
                    sample_viols = numeric_series[viol_mask].head(5).tolist()
                    finding = SchemaFinding(
                        issue_type="constraint_violation",
                        column=col_name,
                        severity="HIGH",
                        description=f"Column '{col_name}' has {viol_count} values violating constraint (> {col_schema.min_value_exclusive}).",
                        evidence={
                            "constraint": f"> {col_schema.min_value_exclusive}",
                            "violation_count": viol_count,
                            "sample_violations": sample_viols,
                        },
                    )
                    constraint_violations_list.append(finding.to_dict())

            # Check min_value (e.g. total_amount >= 0)
            if col_schema.min_value is not None:
                viol_mask = numeric_series < col_schema.min_value
                viol_count = int(viol_mask.dropna().sum())
                if viol_count > 0:
                    sample_viols = numeric_series[viol_mask].head(5).tolist()
                    finding = SchemaFinding(
                        issue_type="constraint_violation",
                        column=col_name,
                        severity="HIGH",
                        description=f"Column '{col_name}' has {viol_count} values violating constraint (>= {col_schema.min_value}).",
                        evidence={
                            "constraint": f">= {col_schema.min_value}",
                            "violation_count": viol_count,
                            "sample_violations": sample_viols,
                        },
                    )
                    constraint_violations_list.append(finding.to_dict())

        # Categorical allowed values check
        if col_schema.allowed_values is not None:
            non_null_cats = series.dropna().astype(str).str.strip()
            invalid_cats = non_null_cats[~non_null_cats.isin(col_schema.allowed_values)]
            if len(invalid_cats) > 0:
                unknown_values = invalid_cats.unique().tolist()
                finding = SchemaFinding(
                    issue_type="constraint_violation",
                    column=col_name,
                    severity="MEDIUM",
                    description=f"Column '{col_name}' contains {len(invalid_cats)} values outside allowed domain.",
                    evidence={
                        "allowed_values": col_schema.allowed_values,
                        "unknown_values": unknown_values[:10],
                        "invalid_count": len(invalid_cats),
                    },
                )
                constraint_violations_list.append(finding.to_dict())

    # -------------------------------------------------------------------------
    # 4. Summary & Overall Validity Calculation
    # -------------------------------------------------------------------------
    all_issues = (
        missing_cols_list
        + unexpected_cols_list
        + type_mismatches_list
        + format_issues_list
        + nullability_violations_list
        + constraint_violations_list
    )

    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for issue in all_issues:
        sev = issue.get("severity", "LOW")
        counts[sev] = counts.get(sev, 0) + 1

    total_issues = len(all_issues)
    # Schema valid only if 0 CRITICAL and 0 HIGH issues exist
    schema_valid = (counts["CRITICAL"] == 0 and counts["HIGH"] == 0)

    summary_dict = {
        "total_issues": total_issues,
        "critical": counts["CRITICAL"],
        "high": counts["HIGH"],
        "medium": counts["MEDIUM"],
        "low": counts["LOW"],
    }

    return SchemaAnalysisResult(
        dataset_name=dataset_name,
        total_rows=total_rows,
        total_columns=total_cols,
        schema_valid=schema_valid,
        expected_columns=expected_cols,
        actual_columns=actual_cols,
        missing_columns=missing_cols_list,
        unexpected_columns=unexpected_cols_list,
        type_mismatches=type_mismatches_list,
        format_issues=format_issues_list,
        nullability_violations=nullability_violations_list,
        constraint_violations=constraint_violations_list,
        summary=summary_dict,
    )
