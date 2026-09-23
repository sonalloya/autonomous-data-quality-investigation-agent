"""
app/tools/correction_tool.py
----------------------------
Phase 7 — Correction Recommendation Tool.

A deterministic tool that converts root-cause investigation results into a safe,
prioritized, explainable correction plan.

Core Principles:
  1. RECOMMENDATION ONLY: Does NOT modify production databases or source datasets.
  2. SAFETY & APPROVAL: Data-altering or quarantining steps explicitly state
     `human_approval_required: True`.
  3. PRIORITIZATION: Prioritizes recommendations by severity, confidence, and metric risk.
  4. VALIDATION CRITERIA: Defines explicit, verifiable criteria consumed by Phase 8.
  5. CLEAN DATA & UNDETERMINED CAUSES:
     - Clean datasets return "No corrective action required."
     - Undetermined causes recommend safe investigation/escalation without inventing data fixes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from app.tools.root_cause_tool import RootCauseResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CorrectionRecommendation:
    """Individual corrective recommendation item."""
    recommendation_id: str
    priority: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    issue: str
    root_cause: str
    affected_columns: List[str]
    evidence: List[str]
    action_steps: List[str]
    expected_outcome: str
    validation_criteria: List[str]
    human_approval_required: bool
    risk_level: str  # 'HIGH', 'MEDIUM', 'LOW'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionPlanResult:
    """Structured correction plan container."""
    dataset_name: str
    status: str  # 'recommendations_generated', 'no_action_required', 'escalation_required'
    summary: str
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    total_recommendations: int = 0
    human_approval_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "status": self.status,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "total_recommendations": self.total_recommendations,
            "human_approval_summary": self.human_approval_summary,
        }


# ---------------------------------------------------------------------------
# Core Tool Function
# ---------------------------------------------------------------------------

def generate_correction_plan(
    root_cause_result: RootCauseResult,
    dataset_name: str = "dataset",
) -> CorrectionPlanResult:
    """
    Generate a deterministic, prioritized correction plan from root cause results.

    Parameters
    ----------
    root_cause_result : RootCauseResult
        Diagnostic result from Phase 6 root-cause analysis.
    dataset_name : str
        Dataset identifier.

    Returns
    -------
    CorrectionPlanResult
    """
    dname = dataset_name or root_cause_result.dataset_name
    p_cause_dict = root_cause_result.primary_root_cause
    p_cause_name = p_cause_dict.get("cause", "Undetermined")
    status = root_cause_result.investigation_status

    # 1. Clean Dataset Scenario
    if status == "no_issues_detected" or p_cause_name == "No Data Quality Issues Detected":
        return CorrectionPlanResult(
            dataset_name=dname,
            status="no_action_required",
            summary=f"No corrective action required for '{dname}'. The dataset exhibits clean data quality across all checks.",
            recommendations=[],
            total_recommendations=0,
            human_approval_summary={"required": 0, "safe_analysis_only": 0},
        )

    # 2. Undetermined Primary Root Cause Scenario
    if p_cause_name == "Undetermined" or status == "inconclusive":
        rec_undetermined = CorrectionRecommendation(
            recommendation_id="REC-000",
            priority="MEDIUM",
            issue="Unresolved data quality evidence requiring investigation",
            root_cause="Undetermined",
            affected_columns=[],
            evidence=[e.get("description", "") for e in root_cause_result.supporting_evidence[:5]],
            action_steps=[
                "Collect additional upstream pipeline extraction logs and API payload metadata.",
                "Inspect source payload transmission timestamps against database ingestion events.",
                "Review data lineage with data engineering and source system teams.",
                "Request manual data governance investigation before applying data modifications.",
            ],
            expected_outcome="Establish a definitive primary root cause with HIGH confidence prior to remediation.",
            validation_criteria=[
                "root_cause_confidence > LOW",
                "unresolved_questions == 0",
            ],
            human_approval_required=False,
            risk_level="LOW",
        )
        return CorrectionPlanResult(
            dataset_name=dname,
            status="escalation_required",
            summary=f"Primary root cause is Undetermined for '{dname}'. Generated escalation recommendations for human data governance review.",
            recommendations=[rec_undetermined.to_dict()],
            total_recommendations=1,
            human_approval_summary={"required": 0, "safe_analysis_only": 1},
        )

    # 3. Specific Root Cause Remediation Mapping
    recommendations_list: List[CorrectionRecommendation] = []
    rec_counter = 1

    # Combine primary cause + contributing causes
    all_causes = [p_cause_dict] + root_cause_result.contributing_causes

    for c_dict in all_causes:
        c_name = c_dict.get("cause", "")
        c_conf = c_dict.get("confidence", "MEDIUM")

        # A. Duplicate Ingestion
        if "Duplicate" in c_name:
            rec = CorrectionRecommendation(
                recommendation_id=f"REC-{rec_counter:03d}",
                priority="CRITICAL" if c_conf == "HIGH" else "HIGH",
                issue="Duplicate transaction records distorting revenue metrics",
                root_cause=c_name,
                affected_columns=["transaction_id", "total_amount"],
                evidence=[e.get("description", "") for e in c_dict.get("supporting_evidence", []) if "duplicate" in e.get("description", "").lower() or "anomaly" in e.get("description", "").lower()],
                action_steps=[
                    "Identify exact duplicate transaction IDs across ingestion batches.",
                    "Determine canonical records based on creation timestamp and payload completeness.",
                    "Quarantine duplicate transaction records into quarantine storage.",
                    "Recalculate affected aggregate revenue and order volume metrics.",
                    "Re-run duplicate and revenue quality validation checks.",
                ],
                expected_outcome="Duplicate transaction rate returns to 0% and monetary metrics reflect unique records.",
                validation_criteria=[
                    "duplicate_transaction_ids == 0",
                    "exact_duplicate_rows == 0",
                    "total_revenue == recalculated_unique_sum",
                ],
                human_approval_required=True,
                risk_level="HIGH",
            )
            recommendations_list.append(rec)
            rec_counter += 1

        # B. Upstream Data Loss / Missing Fields
        elif "Upstream" in c_name or "Missing" in c_name or "loss" in c_name.lower():
            affected_cols = list({e.get("column") for e in c_dict.get("supporting_evidence", []) if e.get("column")})
            if not affected_cols:
                affected_cols = ["customer_id", "region", "quantity", "payment_method"]

            rec = CorrectionRecommendation(
                recommendation_id=f"REC-{rec_counter:03d}",
                priority="HIGH",
                issue="Missing required business fields from payload extract",
                root_cause=c_name,
                affected_columns=affected_cols,
                evidence=[e.get("description", "") for e in c_dict.get("supporting_evidence", []) if "null" in e.get("description", "").lower() or "missing" in e.get("description", "").lower()],
                action_steps=[
                    "Identify all records containing missing values in required columns.",
                    "Inspect source extraction logs and payload transmission metadata for missing fields.",
                    "Recover missing field values where authoritative source lookup data exists.",
                    "Quarantine unrecoverable records missing mandatory identifiers.",
                    "Reprocess affected ingestion batch and re-verify field completeness.",
                ],
                expected_outcome="Required field completeness improves to 100% across mandatory schema columns.",
                validation_criteria=[
                    "nullability_violations == 0",
                    "required_field_completeness == 100%",
                ],
                human_approval_required=True,
                risk_level="HIGH",
            )
            recommendations_list.append(rec)
            rec_counter += 1

        # C. Data Transformation / Type-Conversion Issue
        elif "transformation" in c_name.lower() or "type-conversion" in c_name.lower():
            affected_cols = list({e.get("column") for e in c_dict.get("supporting_evidence", []) if e.get("column")})
            if not affected_cols:
                affected_cols = ["quantity", "unit_price", "total_amount"]

            rec = CorrectionRecommendation(
                recommendation_id=f"REC-{rec_counter:03d}",
                priority="HIGH",
                issue="Unparsed string values and numeric boundary violations",
                root_cause=c_name,
                affected_columns=affected_cols,
                evidence=[e.get("description", "") for e in c_dict.get("supporting_evidence", []) if "type" in e.get("description", "").lower() or "constraint" in e.get("description", "").lower()],
                action_steps=[
                    "Identify malformed string values (e.g. text suffixes like 'units' or currency symbols).",
                    "Safely parse convertible numeric strings into target integer and float types.",
                    "Quarantine unparseable string values for manual review.",
                    "Apply strict integer/float schema casting.",
                    "Re-run numerical constraint and schema type validation checks.",
                ],
                expected_outcome="All numerical columns conform strictly to integer/float types without string suffixes.",
                validation_criteria=[
                    "quantity contains only valid positive integers",
                    "unit_price contains only valid positive floats",
                    "schema_type_mismatches == 0",
                ],
                human_approval_required=True,
                risk_level="MEDIUM",
            )
            recommendations_list.append(rec)
            rec_counter += 1

        # D. Date Parsing / Timestamp Format Inconsistency
        elif "Date" in c_name or "timestamp" in c_name.lower():
            rec = CorrectionRecommendation(
                recommendation_id=f"REC-{rec_counter:03d}",
                priority="MEDIUM",
                issue="Mixed string date formats and unparseable timestamps",
                root_cause=c_name,
                affected_columns=["transaction_date"],
                evidence=[e.get("description", "") for e in c_dict.get("supporting_evidence", []) if "date" in e.get("description", "").lower() or "format" in e.get("description", "").lower()],
                action_steps=[
                    "Identify records with non-ISO date string formats (e.g. MM/DD/YYYY).",
                    "Normalize valid date strings to standard ISO 8601 format (%Y-%m-%d).",
                    "Quarantine unparseable date strings.",
                    "Re-run date format consistency and temporal anomaly checks.",
                ],
                expected_outcome="All transaction dates follow standard ISO %Y-%m-%d format and 100% pass date parsing.",
                validation_criteria=[
                    "transaction_date parseability == 100%",
                    "format_inconsistency_count == 0",
                ],
                human_approval_required=True,
                risk_level="MEDIUM",
            )
            recommendations_list.append(rec)
            rec_counter += 1

        # E. Invalid Categorical Domain Entry
        elif "Categorical" in c_name or "domain" in c_name.lower():
            affected_cols = list({e.get("column") for e in c_dict.get("supporting_evidence", []) if e.get("column")})
            if not affected_cols:
                affected_cols = ["region", "payment_method", "sales_channel"]

            rec = CorrectionRecommendation(
                recommendation_id=f"REC-{rec_counter:03d}",
                priority="MEDIUM",
                issue="Out-of-domain categorical values",
                root_cause=c_name,
                affected_columns=affected_cols,
                evidence=[e.get("description", "") for e in c_dict.get("supporting_evidence", []) if "domain" in e.get("description", "").lower() or "allowed" in e.get("description", "").lower()],
                action_steps=[
                    "Identify categorical values outside approved reference domain lists.",
                    "Compare unknown values against reference mapping tables.",
                    "Map typos or alias values only when an authoritative mapping rule exists.",
                    "Quarantine unmapped out-of-domain values for data governance escalation.",
                    "Re-run categorical domain validation checks.",
                ],
                expected_outcome="All categorical values strictly match approved reference domain values.",
                validation_criteria=[
                    "categorical_constraint_violations == 0",
                    "allowed_values_compliance == 100%",
                ],
                human_approval_required=True,
                risk_level="MEDIUM",
            )
            recommendations_list.append(rec)
            rec_counter += 1

    # Summarize human approval count
    req_count = sum(1 for r in recommendations_list if r.human_approval_required)
    safe_count = len(recommendations_list) - req_count

    summary_text = (
        f"Generated {len(recommendations_list)} prioritized correction recommendations for '{dname}'. "
        f"Primary cause addressed: '{p_cause_name}'. {req_count} actions require explicit human approval."
    )

    return CorrectionPlanResult(
        dataset_name=dname,
        status="recommendations_generated",
        summary=summary_text,
        recommendations=[r.to_dict() for r in recommendations_list],
        total_recommendations=len(recommendations_list),
        human_approval_summary={"required": req_count, "safe_analysis_only": safe_count},
    )
