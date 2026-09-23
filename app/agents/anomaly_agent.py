"""
app/agents/anomaly_agent.py
---------------------------
Phase 4 — Anomaly Detection Agent.

Responsibilities:
  1. Accept a dataset path, DataFrame, or pre-computed AnomalyResult.
  2. Run the deterministic anomaly detection tool (anomaly_tool.py).
  3. Interpret findings — via Gemini LLM if configured, otherwise via
     deterministic rule-based interpretation.
  4. Return a structured AgentAnomalyResponse.

Architecture principle:
  All NUMBERS come from Pandas/NumPy (anomaly_tool.py).
  The LLM or rule-based interpreter only EXPLAINS them.
  The agent must NOT invent statistics, claim root causes, or recommend fixes.

Can be run as CLI:
    python -m app.agents.anomaly_agent data/raw/sales_problematic.csv
    python -m app.agents.anomaly_agent data/test/anomaly.csv --json

Or imported programmatically:
    from app.agents.anomaly_agent import run_anomaly_agent
    response = run_anomaly_agent("data/raw/sales_problematic.csv")
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from app.tools.anomaly_tool import detect_anomalies, AnomalyResult, Severity
from app.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Data Quality Anomaly Analyst.

Your task is to review structured anomaly-detection results that have been
produced by a Python/Pandas statistical analysis tool, and write a concise,
factual interpretation.

STRICT RULES:
1. Every number in your response must come directly from the anomaly data
   provided. Do NOT invent or estimate any value.
2. Do NOT perform calculations yourself — all statistics are pre-computed.
3. Do NOT identify or claim a root cause. Root-cause analysis is a later step.
4. Do NOT recommend corrections or fixes at this stage.
5. Clearly label what is a statistical fact vs. an observation.
6. Be concise: 3–6 bullet-point observations plus a brief paragraph summary.
   Total output: under 350 words.
7. Rank your observations by severity (HIGH first).
8. If no anomalies were found, say so clearly and briefly.

Structure your response exactly as follows:
---
SUMMARY:
<one short paragraph>

KEY FINDINGS:
• <finding 1 — most severe first>
• <finding 2>
• <finding 3>
(add up to 3 more if genuinely warranted)

NOTABLE CONCERNS:
• <concern, or "No significant anomaly concerns detected." if none>
---
"""

# ---------------------------------------------------------------------------
# Agent output schema
# ---------------------------------------------------------------------------


@dataclass
class AgentAnomalyResponse:
    """
    Structured output returned by the Anomaly Detection Agent.

    'summary' and 'key_findings' may come from the LLM (if available)
    or from deterministic rule-based generation.
    'anomaly_metrics' is always from the detection tool.
    """
    dataset_name:      str
    source_path:       str
    anomalies_detected:bool
    total_anomalies:   int
    severity_summary:  dict[str, int]

    summary:           str
    key_findings:      list[str]
    notable_concerns:  list[str]

    # Pass-through details from the tool
    anomaly_metrics:   dict[str, Any]
    outlier_details:   list[dict[str, Any]]
    distribution_details: list[dict[str, Any]]
    metric_details:    list[dict[str, Any]]

    # Meta
    analysis_completed:bool
    llm_used:          bool
    llm_model:         Optional[str]
    anomaly_error:     Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Deterministic (no-LLM) interpretation
# ---------------------------------------------------------------------------


def _deterministic_interpretation(
    result: AnomalyResult,
) -> tuple[str, list[str], list[str]]:
    """
    Generate factual summary and findings using only rule-based logic.
    Returns (summary, key_findings, notable_concerns).
    """
    ss = result.severity_summary
    no = result.numerical_outliers
    da = result.distribution_anomalies
    ma = result.metric_anomalies

    # --- Summary paragraph ---
    if not result.anomalies_detected:
        summary = (
            f"No statistical anomalies were detected in dataset "
            f"'{result.dataset_name}' ({result.row_count:,} rows). "
            "All numerical columns appear within expected IQR fences and "
            "no unusual categorical distributions were identified."
        )
        return summary, ["No anomalies detected."], ["No significant concerns found."]

    high_count   = ss.get(Severity.HIGH, 0)
    medium_count = ss.get(Severity.MEDIUM, 0)
    low_count    = ss.get(Severity.LOW, 0)

    summary = (
        f"Anomaly detection on '{result.dataset_name}' ({result.row_count:,} rows) "
        f"identified {result.total_anomalies} anomalous finding(s): "
        f"{high_count} HIGH, {medium_count} MEDIUM, {low_count} LOW severity. "
    )

    # Highlight top numerical outlier
    iqr_hits = [o for o in no if o.method == "IQR"]
    if iqr_hits:
        worst = max(iqr_hits, key=lambda o: o.outlier_pct or 0)
        summary += (
            f"The most significant numerical outlier is column '{worst.column}' "
            f"with {worst.outlier_count:,} IQR outliers ({worst.outlier_pct:.2f}%). "
        )

    # Highlight metric anomalies
    high_metrics = [m for m in ma if m.severity == Severity.HIGH]
    if high_metrics:
        summary += (
            f"{len(high_metrics)} time-series metric period(s) showed "
            f"HIGH-severity deviation from the daily baseline."
        )

    # --- Key findings ---
    findings: list[str] = []

    # Numerical outliers (IQR only for conciseness)
    for o in sorted(iqr_hits, key=lambda x: x.outlier_pct or 0, reverse=True)[:4]:
        findings.append(
            f"[{o.severity}] Column '{o.column}': {o.outlier_count:,} IQR outliers "
            f"({o.outlier_pct:.2f}%) outside fence "
            f"[{o.lower_fence}, {o.upper_fence}]. "
            f"Max outlier value: {o.max_outlier_value}."
        )

    # Distribution anomalies
    for d in sorted(da, key=lambda x: x.severity, reverse=True)[:3]:
        findings.append(f"[{d.severity}] {d.description}")

    # Metric anomalies (top 3 by severity then deviation)
    sorted_metrics = sorted(
        [m for m in ma if m.anomaly_flag],
        key=lambda m: (m.severity == Severity.HIGH, abs(m.pct_deviation or 0)),
        reverse=True,
    )
    for m in sorted_metrics[:3]:
        findings.append(
            f"[{m.severity}] {m.metric} on {m.period}: "
            f"observed={m.observed_value:,.2f}, baseline={m.baseline_value:,.2f}, "
            f"deviation={m.pct_deviation:+.1f}%"
        )

    if not findings:
        findings.append("No specific anomaly findings to report.")

    # --- Notable concerns ---
    concerns: list[str] = []
    if high_count > 0:
        concerns.append(
            f"{high_count} HIGH-severity anomaly/anomalies detected — "
            "warrants further investigation."
        )
    if any(o.outlier_pct and o.outlier_pct >= 5 for o in iqr_hits):
        worst = max(iqr_hits, key=lambda o: o.outlier_pct or 0)
        concerns.append(
            f"Column '{worst.column}' has a high outlier rate "
            f"({worst.outlier_pct:.2f}%) — this may significantly affect aggregate metrics."
        )
    if any(d.anomaly_type == "unexpected_category" for d in da):
        cols = [d.column for d in da if d.anomaly_type == "unexpected_category"]
        concerns.append(
            f"Unexpected/invalid category values detected in: {', '.join(set(cols))}."
        )
    if any(m.severity == Severity.HIGH for m in ma):
        concerns.append(
            "One or more time-series metric periods show extreme deviation from the "
            "daily baseline — possible data entry errors or collection gaps."
        )
    if not concerns:
        concerns.append(
            "No high-severity anomaly concerns. Statistical outliers exist but are "
            "within a low-to-medium range."
        )

    return summary, findings, concerns


# ---------------------------------------------------------------------------
# LLM interpretation
# ---------------------------------------------------------------------------


def _llm_interpretation(
    result: AnomalyResult,
) -> tuple[Optional[str], Optional[list[str]], Optional[list[str]]]:
    """
    Ask Gemini to interpret the compact anomaly result.
    Returns (summary, findings, concerns) or (None, None, None) on failure.
    """
    if not llm_client.is_available():
        return None, None, None

    compact  = result.to_compact_dict()
    user_msg = (
        "Here are the pre-computed anomaly detection results for this dataset:\n\n"
        + json.dumps(compact, indent=2, default=str)
        + "\n\nPlease provide your analysis following the format in your instructions."
    )

    raw = llm_client.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_message=user_msg,
        max_output_tokens=700,
    )

    if not raw:
        logger.warning("[AnomalyAgent] LLM returned empty response.")
        return None, None, None

    summary:  str       = ""
    findings: list[str] = []
    concerns: list[str] = []
    section:  Optional[str] = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        upper = stripped.upper()
        if upper.startswith("SUMMARY"):
            section = "summary"; continue
        if upper.startswith("KEY FINDING"):
            section = "findings"; continue
        if upper.startswith("NOTABLE CONCERN"):
            section = "concerns"; continue

        if section == "summary":
            summary += (" " if summary else "") + stripped
        elif section == "findings":
            cleaned = stripped.lstrip("•-* ").strip()
            if cleaned:
                findings.append(cleaned)
        elif section == "concerns":
            cleaned = stripped.lstrip("•-* ").strip()
            if cleaned:
                concerns.append(cleaned)

    if not summary and not findings:
        logger.warning("[AnomalyAgent] Could not parse LLM sections.")
        return None, None, None

    return summary or None, findings or None, concerns or None


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------


def run_anomaly_agent(
    source:       Union[str, Path, pd.DataFrame, AnomalyResult],
    dataset_name: Optional[str]  = None,
    date_col:     str            = "transaction_date",
    amount_col:   str            = "total_amount",
    id_col:       str            = "transaction_id",
    run_zscore:   bool           = True,
) -> AgentAnomalyResponse:
    """
    Run the full Anomaly Detection Agent pipeline.

    Parameters
    ----------
    source : str | Path | DataFrame | AnomalyResult
        Dataset to analyze.  If a pre-computed AnomalyResult is passed the
        detection step is skipped and only interpretation runs.
    dataset_name : str, optional
        Override the dataset name label.
    date_col, amount_col, id_col : str
        Column names for time-series metric analysis.
    run_zscore : bool
        Include Z-score secondary detection.

    Returns
    -------
    AgentAnomalyResponse
        Structured agent output.
    """
    # -----------------------------------------------------------------------
    # Step 1: Detect (or reuse existing result)
    # -----------------------------------------------------------------------
    if isinstance(source, AnomalyResult):
        result = source
    else:
        logger.info("[AnomalyAgent] Running anomaly detection on '%s' ...", source)
        result = detect_anomalies(
            source,
            dataset_name=dataset_name,
            date_col=date_col,
            amount_col=amount_col,
            id_col=id_col,
            run_zscore=run_zscore,
        )

    # Handle detection errors
    if result.anomaly_error:
        return AgentAnomalyResponse(
            dataset_name=result.dataset_name,
            source_path=result.source_path,
            anomalies_detected=False,
            total_anomalies=0,
            severity_summary={},
            summary=f"Anomaly detection failed: {result.anomaly_error}",
            key_findings=[],
            notable_concerns=[],
            anomaly_metrics={},
            outlier_details=[],
            distribution_details=[],
            metric_details=[],
            analysis_completed=False,
            llm_used=False,
            llm_model=None,
            anomaly_error=result.anomaly_error,
        )

    # -----------------------------------------------------------------------
    # Step 2: Interpret
    # -----------------------------------------------------------------------
    llm_used  = False
    llm_model = None
    summary   = ""
    findings: list[str] = []
    concerns: list[str] = []

    if llm_client.is_available():
        logger.info("[AnomalyAgent] Requesting Gemini interpretation (%s) ...",
                    llm_client.model_name())
        llm_sum, llm_find, llm_con = _llm_interpretation(result)
        if llm_sum is not None:
            summary   = llm_sum
            findings  = llm_find or []
            concerns  = llm_con  or []
            llm_used  = True
            llm_model = llm_client.model_name()
            logger.info("[AnomalyAgent] LLM interpretation successful.")
        else:
            logger.warning("[AnomalyAgent] LLM failed; using deterministic fallback.")

    if not llm_used:
        logger.info("[AnomalyAgent] Using deterministic interpretation.")
        summary, findings, concerns = _deterministic_interpretation(result)

    # -----------------------------------------------------------------------
    # Step 3: Assemble response
    # -----------------------------------------------------------------------
    meta = result.analysis_metadata
    anomaly_metrics = {
        "total_revenue":           meta.get("total_revenue"),
        "transaction_count":       meta.get("transaction_count"),
        "avg_transaction_value":   meta.get("avg_transaction_value"),
        "numeric_columns_analyzed":meta.get("numeric_columns_analyzed", []),
        "date_range_days":         meta.get("date_range_days"),
        "severity_HIGH":           result.severity_summary.get(Severity.HIGH, 0),
        "severity_MEDIUM":         result.severity_summary.get(Severity.MEDIUM, 0),
        "severity_LOW":            result.severity_summary.get(Severity.LOW, 0),
    }

    return AgentAnomalyResponse(
        dataset_name=result.dataset_name,
        source_path=result.source_path,
        anomalies_detected=result.anomalies_detected,
        total_anomalies=result.total_anomalies,
        severity_summary=result.severity_summary,
        summary=summary,
        key_findings=findings,
        notable_concerns=concerns,
        anomaly_metrics=anomaly_metrics,
        outlier_details=[o.to_dict() for o in result.numerical_outliers],
        distribution_details=[d.to_dict() for d in result.distribution_anomalies],
        metric_details=[m.to_dict() for m in result.metric_anomalies],
        analysis_completed=True,
        llm_used=llm_used,
        llm_model=llm_model,
        anomaly_error=None,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def _pretty_print(response: AgentAnomalyResponse) -> None:
    sep = "=" * 64
    print(sep)
    print(f"  ANOMALY DETECTION REPORT: {response.dataset_name}")
    print(sep)

    status  = "OK" if response.analysis_completed else "FAILED"
    llm_tag = f" | LLM: {response.llm_model}" if response.llm_used else " | LLM: deterministic"
    print(f"  Status: {status}{llm_tag}")

    if response.anomaly_error:
        print(f"\n  ERROR: {response.anomaly_error}\n")
        return

    ann_tag = "YES" if response.anomalies_detected else "NONE"
    ss = response.severity_summary
    print(f"  Anomalies: {ann_tag}  |  Total: {response.total_anomalies}  "
          f"|  HIGH={ss.get('HIGH',0)} MEDIUM={ss.get('MEDIUM',0)} LOW={ss.get('LOW',0)}")

    print(f"\nSUMMARY\n{'-' * 64}")
    print(response.summary)

    print(f"\nKEY FINDINGS\n{'-' * 64}")
    for f in response.key_findings:
        print(f"  * {f}")

    print(f"\nNOTABLE CONCERNS\n{'-' * 64}")
    for c in response.notable_concerns:
        print(f"  ! {c}")

    print(f"\nQUALITY METRICS\n{'-' * 64}")
    for k, v in response.anomaly_metrics.items():
        if v is not None:
            print(f"  {k:<38} {v}")

    if response.outlier_details:
        print(f"\nNUMERICAL OUTLIERS ({len(response.outlier_details)})\n{'-' * 64}")
        for o in response.outlier_details:
            print(f"  [{o['severity']:<6}] [{o['method']:<7}] {o['column']:<20} "
                  f"outliers={o['outlier_count']:>5} ({o['outlier_pct']:.2f}%) "
                  f"max={o['max_outlier_value']}")

    if response.distribution_details:
        print(f"\nDISTRIBUTION ANOMALIES ({len(response.distribution_details)})\n{'-' * 64}")
        for d in response.distribution_details:
            print(f"  [{d['severity']:<6}] {d['column']:<20} {d['anomaly_type']}")
            print(f"           {d['description'][:90]}")

    if response.metric_details:
        high_low = [m for m in response.metric_details if m["anomaly_flag"]]
        print(f"\nMETRIC ANOMALIES ({len(high_low)} flagged)\n{'-' * 64}")
        for m in sorted(high_low, key=lambda x: abs(x["pct_deviation"] or 0), reverse=True)[:8]:
            print(f"  [{m['severity']:<6}] {m['metric']:<35} {m['period']}  "
                  f"obs={m['observed_value']:>10.2f}  "
                  f"dev={m['pct_deviation']:>+7.1f}%")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args = sys.argv[1:]
    if not args:
        print("Usage: python -m app.agents.anomaly_agent <path/to/dataset.csv> [--json]")
        print("\nExample datasets:")
        print("  data/raw/sales_clean.csv")
        print("  data/raw/sales_problematic.csv")
        print("  data/test/anomaly.csv")
        print("  data/test/combined_issues.csv")
        sys.exit(0)

    csv_path = args[0]
    print(f"\nRunning anomaly detection on: {csv_path}\n")

    response = run_anomaly_agent(csv_path)
    _pretty_print(response)

    if "--json" in args:
        print("\nJSON OUTPUT:")
        print(response.to_json())
