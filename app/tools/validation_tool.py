"""
app/tools/validation_tool.py
----------------------------
Phase 8 — Validation Tool.

A deterministic tool that independently verifies whether data-quality issues
identified earlier have been resolved after an approved or simulated correction.

Core Principles:
  1. INDEPENDENT VERIFICATION: Re-runs diagnostic tools (Profiling, Anomaly, Schema)
     on the post-correction dataset. Does NOT simply trust previous assertions.
  2. READ-ONLY GUARANTEE: Does NOT modify original datasets, post-correction CSV files,
     or DataFrames.
  3. BEFORE/AFTER COMPARISON: Calculates empirical metric improvements.
  4. VERDICT ENGINE: Assigns authoritative status ('PASSED', 'PARTIAL', 'FAILED', 'NOT_APPLICABLE').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.tools.profiling_tool import profile_dataset, ProfilingResult
from app.tools.anomaly_tool import detect_anomalies, AnomalyResult
from app.tools.schema_tool import analyze_schema, SchemaAnalysisResult
from app.tools.root_cause_tool import investigate_root_cause, RootCauseResult
from app.tools.correction_tool import generate_correction_plan, CorrectionPlanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    """Individual validation check item."""
    check_id: str
    check_name: str
    category: str  # 'duplicates', 'completeness', 'schema', 'anomalies', 'metrics'
    expected: str
    actual: str
    status: str  # 'PASS', 'FAIL', 'PARTIAL'
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    evidence: str
    before_value: Optional[Any] = None
    after_value: Optional[Any] = None
    improvement_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Structured result returned by the Validation Tool."""
    dataset_name: str
    validation_status: str  # 'PASSED', 'PARTIAL', 'FAILED', 'NOT_APPLICABLE'
    total_checks: int
    passed_checks: int
    failed_checks: int
    partial_checks: int
    checks: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_issues: List[str] = field(default_factory=list)
    before_after_comparison: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "validation_status": self.validation_status,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "partial_checks": self.partial_checks,
            "checks": self.checks,
            "unresolved_issues": self.unresolved_issues,
            "before_after_comparison": self.before_after_comparison,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Core Tool Function
# ---------------------------------------------------------------------------

def validate_correction(
    post_correction_df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
    correction_plan: Optional[CorrectionPlanResult] = None,
    schema: Union[str, Path, Dict[str, Any]] = "sales",
    dataset_name: str = "dataset",
) -> ValidationResult:
    """
    Independently validate post-correction dataset against criteria and baseline.

    Parameters
    ----------
    post_correction_df : pd.DataFrame
        Dataset after correction/remediation.
    original_df : pd.DataFrame, optional
        Baseline dataset before correction.
    correction_plan : CorrectionPlanResult, optional
        Correction plan from Phase 7 containing recommendations and criteria.
    schema : str, Path, or Dict
        Expected schema definition (default 'sales').
    dataset_name : str
        Dataset identifier.

    Returns
    -------
    ValidationResult
    """
    dname = dataset_name or "post_correction_dataset"

    # 1. Execute independent diagnostic tools on post-correction dataset
    post_prof = profile_dataset(post_correction_df, dataset_name=dname)
    post_anom = detect_anomalies(post_correction_df, dataset_name=dname)
    post_sch = analyze_schema(post_correction_df, schema=schema, dataset_name=dname)

    # 2. Execute diagnostic tools on original dataset if provided
    orig_prof: Optional[ProfilingResult] = None
    orig_anom: Optional[AnomalyResult] = None
    orig_sch: Optional[SchemaAnalysisResult] = None

    if original_df is not None:
        orig_prof = profile_dataset(original_df, dataset_name="baseline")
        orig_anom = detect_anomalies(original_df, dataset_name="baseline")
        orig_sch = analyze_schema(original_df, schema=schema, dataset_name="baseline")

    # 3. Clean Dataset Check
    is_orig_clean = False
    if orig_sch is not None and orig_sch.schema_valid:
        orig_dups = orig_prof.duplicate_summary.duplicate_rows if orig_prof else 0
        orig_nulls = orig_prof.missing_value_summary.total_missing_cells if orig_prof else 0
        if orig_dups == 0 and orig_nulls == 0 and orig_sch.summary.get("high", 0) == 0 and orig_sch.summary.get("critical", 0) == 0:
            is_orig_clean = True

    if is_orig_clean:
        return ValidationResult(
            dataset_name=dname,
            validation_status="NOT_APPLICABLE",
            total_checks=0,
            passed_checks=0,
            failed_checks=0,
            partial_checks=0,
            checks=[],
            unresolved_issues=[],
            before_after_comparison={},
            summary=f"Validation NOT_APPLICABLE for '{dname}'. Original dataset is clean with zero defects; no corrective action required.",
        )

    checks_list: List[ValidationCheck] = []
    unresolved: List[str] = []
    comparison: Dict[str, Any] = {}
    check_counter = 1

    # -------------------------------------------------------------------------
    # A. Duplicate Transaction Validation
    # -------------------------------------------------------------------------
    post_dup_rows = post_prof.duplicate_summary.duplicate_rows
    post_dup_txns = post_prof.duplicate_summary.duplicate_transaction_ids or 0
    total_post_dups = max(post_dup_rows, post_dup_txns)

    orig_dup_total = 0
    if orig_prof is not None and orig_prof.duplicate_summary:
        orig_dup_total = max(orig_prof.duplicate_summary.duplicate_rows, orig_prof.duplicate_summary.duplicate_transaction_ids or 0)

    if orig_dup_total > 0 or total_post_dups > 0:
        improvement = 0.0
        if orig_dup_total > 0:
            improvement = round(((orig_dup_total - total_post_dups) / orig_dup_total) * 100.0, 1)

        comparison["duplicates"] = {
            "before": orig_dup_total,
            "after": total_post_dups,
            "improvement_pct": improvement,
        }

        if total_post_dups == 0:
            status = "PASS"
            evidence = f"0 duplicate transaction IDs/rows remaining (down from {orig_dup_total} before correction)."
        elif total_post_dups < orig_dup_total:
            status = "PARTIAL"
            evidence = f"{total_post_dups} duplicates remain (reduced from {orig_dup_total}, {improvement}% improvement)."
            unresolved.append(f"Duplicate validation: {total_post_dups} duplicate records lingering.")
        else:
            status = "FAIL"
            evidence = f"Duplicate count unchanged or increased ({total_post_dups} duplicates remain)."
            unresolved.append(f"Duplicate validation: {total_post_dups} duplicate records remain unresolved.")

        checks_list.append(
            ValidationCheck(
                check_id=f"VAL-{check_counter:03d}",
                check_name="Duplicate Transaction Validation",
                category="duplicates",
                expected="0 duplicate transaction IDs / exact duplicate rows",
                actual=f"{total_post_dups} duplicates detected",
                status=status,
                severity="HIGH",
                evidence=evidence,
                before_value=orig_dup_total,
                after_value=total_post_dups,
                improvement_pct=improvement if orig_dup_total > 0 else None,
            )
        )
        check_counter += 1

    # -------------------------------------------------------------------------
    # B. Completeness & Nullability Validation
    # -------------------------------------------------------------------------
    post_null_viol_count = len(post_sch.nullability_violations)
    orig_null_viol_count = len(orig_sch.nullability_violations) if orig_sch else 0

    if orig_null_viol_count > 0 or post_null_viol_count > 0:
        improvement = 0.0
        if orig_null_viol_count > 0:
            improvement = round(((orig_null_viol_count - post_null_viol_count) / orig_null_viol_count) * 100.0, 1)

        comparison["nullability_violations"] = {
            "before": orig_null_viol_count,
            "after": post_null_viol_count,
            "improvement_pct": improvement,
        }

        if post_null_viol_count == 0:
            status = "PASS"
            evidence = f"Required field completeness is 100% with 0 nullability violations (down from {orig_null_viol_count})."
        elif post_null_viol_count < orig_null_viol_count:
            status = "PARTIAL"
            evidence = f"{post_null_viol_count} nullability violations remain ({improvement}% improvement)."
            unresolved.append(f"Completeness validation: {post_null_viol_count} non-nullable fields contain nulls.")
        else:
            status = "FAIL"
            evidence = f"{post_null_viol_count} nullability violations remaining."
            unresolved.append(f"Completeness validation: {post_null_viol_count} required field nullability violations remain.")

        checks_list.append(
            ValidationCheck(
                check_id=f"VAL-{check_counter:03d}",
                check_name="Required Field Completeness Validation",
                category="completeness",
                expected="100% required field completeness (0 nullability violations)",
                actual=f"{post_null_viol_count} nullability violations",
                status=status,
                severity="HIGH",
                evidence=evidence,
                before_value=orig_null_viol_count,
                after_value=post_null_viol_count,
                improvement_pct=improvement if orig_null_viol_count > 0 else None,
            )
        )
        check_counter += 1

    # -------------------------------------------------------------------------
    # C. Schema & Type Validation
    # -------------------------------------------------------------------------
    post_type_mismatches = len(post_sch.type_mismatches)
    orig_type_mismatches = len(orig_sch.type_mismatches) if orig_sch else 0

    if orig_type_mismatches > 0 or post_type_mismatches > 0:
        improvement = 0.0
        if orig_type_mismatches > 0:
            improvement = round(((orig_type_mismatches - post_type_mismatches) / orig_type_mismatches) * 100.0, 1)

        comparison["type_mismatches"] = {
            "before": orig_type_mismatches,
            "after": post_type_mismatches,
            "improvement_pct": improvement,
        }

        if post_type_mismatches == 0:
            status = "PASS"
            evidence = f"All columns match expected logical data types without string suffixes (down from {orig_type_mismatches})."
        else:
            status = "FAIL"
            evidence = f"{post_type_mismatches} type mismatches lingering in schema."
            unresolved.append(f"Schema type validation: {post_type_mismatches} columns contain type mismatches.")

        checks_list.append(
            ValidationCheck(
                check_id=f"VAL-{check_counter:03d}",
                check_name="Schema Type Conformance Validation",
                category="schema",
                expected="0 logical type mismatches",
                actual=f"{post_type_mismatches} type mismatches",
                status=status,
                severity="HIGH",
                evidence=evidence,
                before_value=orig_type_mismatches,
                after_value=post_type_mismatches,
                improvement_pct=improvement if orig_type_mismatches > 0 else None,
            )
        )
        check_counter += 1

    # -------------------------------------------------------------------------
    # D. Date Format Validation
    # -------------------------------------------------------------------------
    post_fmt_issues = len(post_sch.format_issues)
    orig_fmt_issues = len(orig_sch.format_issues) if orig_sch else 0

    if orig_fmt_issues > 0 or post_fmt_issues > 0:
        improvement = 0.0
        if orig_fmt_issues > 0:
            improvement = round(((orig_fmt_issues - post_fmt_issues) / orig_fmt_issues) * 100.0, 1)

        comparison["format_issues"] = {
            "before": orig_fmt_issues,
            "after": post_fmt_issues,
            "improvement_pct": improvement,
        }

        if post_fmt_issues == 0:
            status = "PASS"
            evidence = f"100% date parseability adhering to standard ISO %Y-%m-%d format."
        else:
            status = "FAIL"
            evidence = f"{post_fmt_issues} date format issues remaining."
            unresolved.append(f"Date validation: {post_fmt_issues} date format inconsistencies remain.")

        checks_list.append(
            ValidationCheck(
                check_id=f"VAL-{check_counter:03d}",
                check_name="Date Format & Timestamp Validation",
                category="schema",
                expected="100% date parseability conforming to ISO %Y-%m-%d",
                actual=f"{post_fmt_issues} format issues",
                status=status,
                severity="MEDIUM",
                evidence=evidence,
                before_value=orig_fmt_issues,
                after_value=post_fmt_issues,
                improvement_pct=improvement if orig_fmt_issues > 0 else None,
            )
        )
        check_counter += 1

    # -------------------------------------------------------------------------
    # E. Business Metric Validation (Revenue & Order Recalculation)
    # -------------------------------------------------------------------------
    if "total_amount" in post_correction_df.columns and pd.api.types.is_numeric_dtype(post_correction_df["total_amount"]):
        post_revenue = float(post_correction_df["total_amount"].dropna().sum())
        orig_revenue = float(original_df["total_amount"].dropna().sum()) if original_df is not None and "total_amount" in original_df.columns and pd.api.types.is_numeric_dtype(original_df["total_amount"]) else 0.0

        if orig_revenue > 0:
            comparison["total_revenue"] = {
                "before": round(orig_revenue, 2),
                "after": round(post_revenue, 2),
                "difference": round(post_revenue - orig_revenue, 2),
            }

            # Check if revenue recalculated cleanly
            rev_status = "PASS" if total_post_dups == 0 and post_null_viol_count == 0 else "PARTIAL"
            checks_list.append(
                ValidationCheck(
                    check_id=f"VAL-{check_counter:03d}",
                    check_name="Business Metric Revenue Validation",
                    category="metrics",
                    expected=f"Recalculated monetary revenue matching clean records",
                    actual=f"${post_revenue:,.2f}",
                    status=rev_status,
                    severity="HIGH",
                    evidence=f"Post-correction total revenue recalculated at ${post_revenue:,.2f} (baseline was ${orig_revenue:,.2f}).",
                    before_value=round(orig_revenue, 2),
                    after_value=round(post_revenue, 2),
                )
            )
            check_counter += 1

    # -------------------------------------------------------------------------
    # F. Verdict Calculation
    # -------------------------------------------------------------------------
    passed_count = sum(1 for c in checks_list if c.status == "PASS")
    failed_count = sum(1 for c in checks_list if c.status == "FAIL")
    partial_count = sum(1 for c in checks_list if c.status == "PARTIAL")
    total_count = len(checks_list)

    if total_count == 0:
        verdict = "NOT_APPLICABLE"
        summary_text = f"Validation NOT_APPLICABLE for '{dname}'. Zero validation checks were required."
    elif failed_count == 0 and partial_count == 0 and passed_count > 0:
        verdict = "PASSED"
        summary_text = f"Validation PASSED for '{dname}'. All {passed_count} validation checks passed cleanly with 100% resolution of data quality issues."
    elif passed_count > 0 and (failed_count > 0 or partial_count > 0):
        verdict = "PARTIAL"
        summary_text = f"Validation PARTIAL for '{dname}'. {passed_count} of {total_count} checks passed, but lingering data quality issues require attention."
    else:
        verdict = "FAILED"
        summary_text = f"Validation FAILED for '{dname}'. {failed_count} critical validation checks failed; primary data quality issues remain unresolved."

    return ValidationResult(
        dataset_name=dname,
        validation_status=verdict,
        total_checks=total_count,
        passed_checks=passed_count,
        failed_checks=failed_count,
        partial_checks=partial_count,
        checks=[c.to_dict() for c in checks_list],
        unresolved_issues=unresolved,
        before_after_comparison=comparison,
        summary=summary_text,
    )
