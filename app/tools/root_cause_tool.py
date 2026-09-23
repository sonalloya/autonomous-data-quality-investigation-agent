"""
app/tools/root_cause_tool.py
----------------------------
Phase 6 — Root Cause Analysis Tool.

A deterministic tool that normalizes, correlates, evaluates, and ranks evidence
from Data Profiling, Anomaly Detection, and Schema Analysis tools to determine
the root cause(s) of data-quality issues.

Core Principles:
  1. Reason ONLY from empirical evidence provided by diagnostic tools.
  2. Normalize findings into structured EvidenceItem representation.
  3. Correlate related findings across independent analysis tools.
  4. Evaluate and rank candidate root causes with explainable scoring.
  5. Assign explicit confidence ratings ('HIGH', 'MEDIUM', 'LOW').
  6. Return 'Undetermined' primary cause when evidence is inconclusive or contradictory.
  7. Do NOT invent unverified hypotheses or claim unsupported certainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from app.tools.profiling_tool import ProfilingResult
from app.tools.anomaly_tool import AnomalyResult
from app.tools.schema_tool import SchemaAnalysisResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """Normalized finding representation across diagnostic tools."""
    source: str  # 'profiling', 'anomaly', 'schema'
    finding_type: str
    column: Optional[str]
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    description: str
    observed_value: Any = None
    expected_value: Any = None
    evidence_strength: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateCause:
    """Candidate root cause hypothesis under evaluation."""
    cause: str
    affected_data_area: str
    supporting_evidence: List[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = field(default_factory=list)
    evidence_count: int = 0
    score: float = 0.0
    confidence: str = "LOW"  # 'HIGH', 'MEDIUM', 'LOW'
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "affected_data_area": self.affected_data_area,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "contradicting_evidence": [e.to_dict() for e in self.contradicting_evidence],
            "evidence_count": self.evidence_count,
            "score": round(self.score, 2),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class RootCauseResult:
    """Structured result returned by the Root Cause Analysis Tool."""
    dataset_name: str
    investigation_status: str  # 'completed', 'inconclusive', 'no_issues_detected'
    primary_root_cause: Dict[str, Any]
    contributing_causes: List[Dict[str, Any]] = field(default_factory=list)
    unrelated_findings: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "investigation_status": self.investigation_status,
            "primary_root_cause": self.primary_root_cause,
            "contributing_causes": self.contributing_causes,
            "unrelated_findings": self.unrelated_findings,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "unresolved_questions": self.unresolved_questions,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# 1. Evidence Normalization
# ---------------------------------------------------------------------------

def normalize_evidence(
    profiling_res: Optional[ProfilingResult] = None,
    anomaly_res: Optional[AnomalyResult] = None,
    schema_res: Optional[SchemaAnalysisResult] = None,
) -> List[EvidenceItem]:
    """Extract and normalize evidence items from diagnostic results."""
    items: List[EvidenceItem] = []

    # A. Profiling Evidence
    if profiling_res is not None and profiling_res.duplicate_summary is not None:
        dup_rows = profiling_res.duplicate_summary.duplicate_rows
        dup_txns = profiling_res.duplicate_summary.duplicate_transaction_ids or 0
        total_dups = max(dup_rows, dup_txns)
        if total_dups > 0:
            items.append(
                EvidenceItem(
                    source="profiling",
                    finding_type="duplicate_records",
                    column=None,
                    severity="HIGH" if total_dups > 50 else "MEDIUM",
                    description=f"{total_dups} duplicate records/IDs detected in dataset.",
                    observed_value=total_dups,
                    expected_value=0,
                )
            )

        # Null value profiling
        for col_detail in profiling_res.column_details:
            if col_detail.null_count > 0:
                sev = "HIGH" if col_detail.null_percentage > 3.0 else "MEDIUM"
                items.append(
                    EvidenceItem(
                        source="profiling",
                        finding_type="missing_values",
                        column=col_detail.column_name,
                        severity=sev,
                        description=f"Column '{col_detail.column_name}' has {col_detail.null_count} nulls ({col_detail.null_percentage}%).",
                        observed_value=col_detail.null_count,
                        expected_value=0,
                    )
                )

    # B. Anomaly Evidence
    if anomaly_res is not None:
        # Business metrics anomalies (e.g. revenue drop/spike)
        for m_anom in anomaly_res.metric_anomalies:
            items.append(
                EvidenceItem(
                    source="anomaly",
                    finding_type="metric_anomaly",
                    column=m_anom.metric,
                    severity=m_anom.severity,
                    description=f"Business metric anomaly on '{m_anom.metric}': deviation {m_anom.pct_deviation:.1f}% ({m_anom.direction} baseline).",
                    observed_value=m_anom.observed_value,
                    expected_value=m_anom.baseline_value,
                )
            )

        # Categorical / distribution anomalies
        for d_anom in anomaly_res.distribution_anomalies:
            items.append(
                EvidenceItem(
                    source="anomaly",
                    finding_type="distribution_anomaly",
                    column=d_anom.column,
                    severity=d_anom.severity,
                    description=f"Distribution anomaly in '{d_anom.column}': dominant value '{d_anom.dominant_value}' ({d_anom.dominant_pct}%).",
                    observed_value=d_anom.dominant_pct,
                    expected_value="Uniform distribution",
                )
            )

        # Numerical outliers
        for num_out in anomaly_res.numerical_outliers:
            if num_out.severity in ("HIGH", "MEDIUM"):
                items.append(
                    EvidenceItem(
                        source="anomaly",
                        finding_type="numerical_outlier",
                        column=num_out.column,
                        severity=num_out.severity,
                        description=f"Numerical outliers in '{num_out.column}' via {num_out.method}: {num_out.outlier_count} outliers ({num_out.outlier_pct}%).",
                        observed_value=num_out.outlier_count,
                        expected_value=0,
                    )
                )

    # C. Schema Evidence
    if schema_res is not None:
        for m_col in schema_res.missing_columns:
            items.append(
                EvidenceItem(
                    source="schema",
                    finding_type="missing_column",
                    column=m_col.get("column"),
                    severity=m_col.get("severity", "HIGH"),
                    description=m_col.get("description", ""),
                )
            )

        for u_col in schema_res.unexpected_columns:
            items.append(
                EvidenceItem(
                    source="schema",
                    finding_type="unexpected_column",
                    column=u_col.get("column"),
                    severity=u_col.get("severity", "LOW"),
                    description=u_col.get("description", ""),
                )
            )

        for t_mis in schema_res.type_mismatches:
            items.append(
                EvidenceItem(
                    source="schema",
                    finding_type="type_mismatch",
                    column=t_mis.get("column"),
                    severity=t_mis.get("severity", "HIGH"),
                    description=t_mis.get("description", ""),
                    observed_value=t_mis.get("evidence", {}).get("actual_pandas_dtype"),
                    expected_value=t_mis.get("evidence", {}).get("expected_type"),
                )
            )

        for f_iss in schema_res.format_issues:
            items.append(
                EvidenceItem(
                    source="schema",
                    finding_type="format_issue",
                    column=f_iss.get("column"),
                    severity=f_iss.get("severity", "HIGH"),
                    description=f_iss.get("description", ""),
                )
            )

        for n_viol in schema_res.nullability_violations:
            items.append(
                EvidenceItem(
                    source="schema",
                    finding_type="nullability_violation",
                    column=n_viol.get("column"),
                    severity=n_viol.get("severity", "HIGH"),
                    description=n_viol.get("description", ""),
                )
            )

        for c_viol in schema_res.constraint_violations:
            items.append(
                EvidenceItem(
                    source="schema",
                    finding_type="constraint_violation",
                    column=c_viol.get("column"),
                    severity=c_viol.get("severity", "HIGH"),
                    description=c_viol.get("description", ""),
                )
            )

    return items


# ---------------------------------------------------------------------------
# 2. Candidate Root Cause Generation & Evaluation
# ---------------------------------------------------------------------------

def generate_candidate_causes(evidence_list: List[EvidenceItem]) -> List[CandidateCause]:
    """Generate candidate root cause hypotheses correlated with evidence."""
    candidates: List[CandidateCause] = []

    # Category A: Duplicate Transaction Ingestion
    dup_items = [e for e in evidence_list if e.finding_type == "duplicate_records"]
    revenue_anom_items = [e for e in evidence_list if e.finding_type == "metric_anomaly" and e.column in ("total_amount", "revenue", "daily_revenue")]
    if dup_items:
        c_dup = CandidateCause(
            cause="Duplicate transaction ingestion in data pipeline",
            affected_data_area="Transaction ingestion & storage",
            supporting_evidence=dup_items + revenue_anom_items,
            reasoning="Exact duplicate transaction records were detected, which inflates order counts and distorts monetary revenue metrics.",
        )
        candidates.append(c_dup)

    # Category B: Missing or Dropped Upstream Records / Ingestion Loss
    null_items = [e for e in evidence_list if e.finding_type in ("nullability_violation", "missing_values") and e.severity in ("CRITICAL", "HIGH")]
    missing_col_items = [e for e in evidence_list if e.finding_type == "missing_column"]
    metric_drop_items = [e for e in evidence_list if e.finding_type == "metric_anomaly" and "drop" in e.description.lower()]
    if null_items or missing_col_items:
        c_missing = CandidateCause(
            cause="Upstream data loss or missing required fields during ingestion",
            affected_data_area="Source ETL extract & payload transmission",
            supporting_evidence=null_items + missing_col_items + metric_drop_items,
            reasoning="Required business fields contain missing/null values or dropped columns, indicating upstream extract or transfer failures.",
        )
        candidates.append(c_missing)

    # Category C: Data Transformation / Type-Conversion Failure
    type_items = [e for e in evidence_list if e.finding_type == "type_mismatch"]
    constraint_items = [e for e in evidence_list if e.finding_type == "constraint_violation" and e.column in ("quantity", "unit_price", "total_amount")]
    if type_items or constraint_items:
        c_trans = CandidateCause(
            cause="Data transformation and type-conversion parsing failure",
            affected_data_area="ETL type casting and numerical normalization",
            supporting_evidence=type_items + constraint_items,
            reasoning="Numerical fields contain string text/suffixes or range constraint violations, indicating unparsed data conversion logic.",
        )
        candidates.append(c_trans)

    # Category D: Date Parsing and Temporal Format Inconsistency
    format_items = [e for e in evidence_list if e.finding_type == "format_issue"]
    if format_items:
        c_date = CandidateCause(
            cause="Date parsing and timestamp format inconsistency",
            affected_data_area="Date/Time formatting & parsing pipeline",
            supporting_evidence=format_items,
            reasoning="Date columns contain unparseable values or mixed string date formats (e.g., MM/DD/YYYY vs YYYY-MM-DD ISO standard).",
        )
        candidates.append(c_date)

    # Category E: Out-of-Domain Categorical Data Entry Error
    cat_constraint_items = [e for e in evidence_list if e.finding_type == "constraint_violation" and e.column in ("region", "payment_method", "sales_channel")]
    dist_items = [e for e in evidence_list if e.finding_type == "distribution_anomaly"]
    if cat_constraint_items:
        c_cat = CandidateCause(
            cause="Invalid categorical values or domain entry error",
            affected_data_area="Reference domain validation & lookup tables",
            supporting_evidence=cat_constraint_items + dist_items,
            reasoning="Categorical columns contain values outside approved business domain reference lists.",
        )
        candidates.append(c_cat)

    # Category F: Natural Statistical Price Range Variation (Clean Data)
    outlier_items = [e for e in evidence_list if e.finding_type == "numerical_outlier"]
    has_severe_defects = any(
        e.finding_type in ("duplicate_records", "missing_values", "nullability_violation", "type_mismatch", "format_issue", "missing_column")
        for e in evidence_list
    )
    if outlier_items and not has_severe_defects:
        c_clean = CandidateCause(
            cause="No Data Quality Issues Detected",
            affected_data_area="Statistical numerical distribution",
            supporting_evidence=outlier_items,
            reasoning="Statistical outliers are expected artifacts of a wide natural price range across products in a structurally valid dataset with zero duplicate, missing, or format errors.",
        )
        c_clean.score = 5.0
        c_clean.confidence = "HIGH"
        candidates.append(c_clean)

    return candidates


# ---------------------------------------------------------------------------
# 3. Scoring & Ranking Logic
# ---------------------------------------------------------------------------

def rank_candidates(candidates: List[CandidateCause]) -> List[CandidateCause]:
    """Score and rank candidate causes based on evidence strength and multi-tool agreement."""
    sev_weights = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.5}

    for c in candidates:
        score = 0.0
        sources_seen = set()

        for item in c.supporting_evidence:
            weight = sev_weights.get(item.severity, 1.0) * item.evidence_strength
            score += weight
            sources_seen.add(item.source)

        # Multi-tool consistency bonus: +1.0 for each additional diagnostic tool agreeing
        if len(sources_seen) >= 2:
            score += 1.5 * (len(sources_seen) - 1)

        # Subtract penalty for contradictory evidence
        for item in c.contradicting_evidence:
            weight = sev_weights.get(item.severity, 1.0)
            score -= weight

        c.score = max(0.0, score)
        c.evidence_count = len(c.supporting_evidence)

        # Assign confidence rating
        if c.cause == "No Data Quality Issues Detected":
            c.confidence = "HIGH"
        elif c.score >= 4.0 and len(sources_seen) >= 2 and c.evidence_count >= 2:
            c.confidence = "HIGH"
        elif c.score >= 2.0 and c.evidence_count >= 1:
            c.confidence = "MEDIUM"
        else:
            c.confidence = "LOW"

    # Sort descending by score
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Core Tool Function
# ---------------------------------------------------------------------------

def investigate_root_cause(
    profiling_result: Optional[ProfilingResult] = None,
    anomaly_result: Optional[AnomalyResult] = None,
    schema_result: Optional[SchemaAnalysisResult] = None,
    dataset_name: str = "dataset",
) -> RootCauseResult:
    """
    Perform evidence-based root cause investigation across diagnostic outputs.

    Parameters
    ----------
    profiling_result : ProfilingResult, optional
    anomaly_result : AnomalyResult, optional
    schema_result : SchemaAnalysisResult, optional
    dataset_name : str

    Returns
    -------
    RootCauseResult
    """
    # 1. Normalize evidence from all provided tools
    evidence_list = normalize_evidence(profiling_result, anomaly_result, schema_result)

    if not evidence_list:
        # No issues detected across any tool
        primary_cause = {
            "cause": "No Data Quality Issues Detected",
            "confidence": "HIGH",
            "supporting_evidence_count": 0,
            "reasoning": "All diagnostic tools (Profiling, Anomaly Detection, Schema Analysis) reported 100% clean data with zero structural or metric anomalies.",
        }
        return RootCauseResult(
            dataset_name=dataset_name,
            investigation_status="no_issues_detected",
            primary_root_cause=primary_cause,
            contributing_causes=[],
            unrelated_findings=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            unresolved_questions=[],
            summary=f"Dataset '{dataset_name}' exhibits clean data quality with zero detected anomalies, schema violations, or missing fields.",
        )

    # 2. Generate candidate causes
    candidates = generate_candidate_causes(evidence_list)

    # 3. Score and rank candidates
    ranked_candidates = rank_candidates(candidates)

    unresolved_questions: List[str] = []
    contributing_causes: List[Dict[str, Any]] = []

    # 4. Primary cause determination
    if not ranked_candidates or ranked_candidates[0].score < 1.5:
        # Inconclusive / Undetermined
        primary_cause = {
            "cause": "Undetermined",
            "confidence": "LOW",
            "supporting_evidence_count": len(evidence_list),
            "reasoning": "Detected findings are minor or lack sufficient cross-tool correlation to establish a definitive single root cause.",
        }
        status = "inconclusive"
        unresolved_questions.append("Are there additional upstream pipeline execution logs available?")
        unresolved_questions.append("Can transaction source systems verify raw payload integrity?")
    else:
        top_cand = ranked_candidates[0]
        if top_cand.cause == "No Data Quality Issues Detected":
            primary_cause = top_cand.to_dict()
            status = "no_issues_detected"
            contributing_causes = []
        else:
            # Check if secondary candidates exist with close scores
            close_competitors = [c for c in ranked_candidates[1:] if c.score >= 0.7 * top_cand.score and c.score >= 2.0]

            if len(close_competitors) > 0:
                # Multiple strong root causes
                primary_cause = top_cand.to_dict()
                for comp in close_competitors:
                    contributing_causes.append(comp.to_dict())
                status = "completed"
            else:
                primary_cause = top_cand.to_dict()
                contributing_causes = [c.to_dict() for c in ranked_candidates[1:] if c.score >= 1.5]
                status = "completed"

    # Identify findings not accounted for by primary cause
    primary_ev_descs = {e.get("description") for e in primary_cause.get("supporting_evidence", [])}
    unrelated = [e.to_dict() for e in evidence_list if e.description not in primary_ev_descs]

    # Build investigation summary narrative
    if primary_cause["cause"] == "Undetermined":
        summary_text = (
            f"Investigation for '{dataset_name}' evaluated {len(evidence_list)} evidence findings. "
            f"No single primary root cause could be established with sufficient confidence from the available evidence."
        )
    else:
        summary_text = (
            f"Primary root cause identified for '{dataset_name}': '{primary_cause['cause']}' "
            f"with {primary_cause['confidence']} confidence based on {primary_cause.get('evidence_count', len(primary_cause.get('supporting_evidence', [])))} supporting evidence findings."
        )

    return RootCauseResult(
        dataset_name=dataset_name,
        investigation_status=status,
        primary_root_cause=primary_cause,
        contributing_causes=contributing_causes,
        unrelated_findings=unrelated,
        supporting_evidence=[e.to_dict() for e in evidence_list],
        contradicting_evidence=[],
        unresolved_questions=unresolved_questions,
        summary=summary_text,
    )
