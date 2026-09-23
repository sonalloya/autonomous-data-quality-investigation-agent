"""
app/agents/profiling_agent.py
-----------------------------
Phase 3 — Data Profiling Agent.

Responsibilities:
  1. Accept a dataset path (or ProfilingResult).
  2. Run the deterministic Pandas profiling tool.
  3. Interpret findings — via LLM if available, otherwise via
     deterministic rule-based interpretation.
  4. Return a structured AgentProfilingResponse.

Architecture principle:
  All NUMBERS come from Pandas (profiling_tool.py).
  The LLM or rule-based interpreter only EXPLAINS them.

Can be run directly:
    python -m app.agents.profiling_agent data/raw/sales_problematic.csv

Or imported and called programmatically:
    from app.agents.profiling_agent import run_profiling_agent
    response = run_profiling_agent("data/raw/sales_clean.csv")
"""

from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from app.tools.profiling_tool import profile_dataset, ProfilingResult
from app.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# System prompt for the Gemini profiling agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Data Quality Profiling Analyst.

Your task is to review structured data-profiling statistics that have been
calculated by a Python Pandas tool, and produce a concise, factual summary.

STRICT RULES:
1. Do NOT invent or guess numerical values. Every number in your response
   must come directly from the profiling data provided to you.
2. Do NOT perform any calculations yourself. All statistics are pre-computed.
3. Do NOT suggest root causes or fixes at this stage. Only observe and describe.
4. Do NOT claim to have performed the calculations. Python/Pandas computed them.
5. Distinguish clearly between facts (from the data) and observations.
6. Keep the summary concise — 3 to 5 bullet-point observations and a short
   paragraph summary. Total output: under 300 words.
7. Highlight the most important data-quality concerns clearly.
8. Use plain English. Avoid jargon.

Structure your response as follows:
---
SUMMARY:
<one short paragraph summarising what the dataset looks like overall>

KEY OBSERVATIONS:
• <observation 1>
• <observation 2>
• <observation 3>
(add up to 2 more if genuinely important)

QUALITY CONCERNS:
• <concern 1 — only if real problems were found>
---
"""

# ---------------------------------------------------------------------------
# Agent output schema
# ---------------------------------------------------------------------------


@dataclass
class AgentProfilingResponse:
    """
    Structured output returned by the Data Profiling Agent.

    'summary' and 'key_observations' may come from the LLM (if available)
    or from deterministic rule-based generation.
    'quality_metrics' and 'column_findings' are always Pandas-computed.
    """
    dataset_name:       str
    source_path:        str
    summary:            str
    key_observations:   list[str]
    quality_concerns:   list[str]

    # Direct pass-through of Pandas-computed metrics
    quality_metrics:   dict[str, Any]
    column_findings:   list[dict[str, Any]]

    # Meta
    profiling_completed: bool
    llm_used:            bool
    llm_model:           Optional[str]
    profiling_error:     Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Deterministic (no-LLM) interpretation
# ---------------------------------------------------------------------------


def _deterministic_interpretation(result: ProfilingResult) -> tuple[str, list[str], list[str]]:
    """
    Generate a factual summary and observations using only rule-based logic.
    Used when the LLM is not available.

    Returns (summary, key_observations, quality_concerns).
    """
    qi = result.quality_indicators
    ms = result.missing_value_summary
    ds = result.duplicate_summary

    # --- Summary paragraph ---
    summary = (
        f"Dataset '{result.dataset_name}' contains {result.row_count:,} rows and "
        f"{result.column_count} columns. "
        f"Overall completeness is {qi.completeness_pct:.1f}% "
        f"({ms.total_missing_cells:,} missing cells out of {ms.total_cells:,} total). "
    )
    if ds.duplicate_rows > 0:
        summary += (
            f"There are {ds.duplicate_rows:,} fully duplicate rows "
            f"({ds.duplicate_row_percentage:.2f}% of all rows). "
        )
    else:
        summary += "No fully duplicate rows were detected. "

    if ds.duplicate_transaction_ids is not None:
        if ds.duplicate_transaction_ids > 0:
            summary += (
                f"Additionally, {ds.duplicate_transaction_ids:,} duplicate "
                f"transaction_id values were found. "
            )
        else:
            summary += "All transaction_id values are unique. "

    summary = summary.strip()

    # --- Key observations ---
    observations: list[str] = [
        f"Row count: {result.row_count:,} | Columns: {result.column_count}",
        f"Numeric columns ({len(result.numeric_columns)}): {', '.join(result.numeric_columns) or 'none'}",
        f"Categorical columns ({len(result.categorical_columns)}): {', '.join(result.categorical_columns) or 'none'}",
        f"Overall completeness: {qi.completeness_pct:.2f}% "
        f"({ms.columns_with_nulls} columns contain nulls)",
        f"Duplicate rows: {ds.duplicate_rows:,} ({qi.duplicate_pct:.2f}%)",
    ]
    if ds.duplicate_transaction_ids is not None:
        observations.append(
            f"Duplicate transaction_ids: {ds.duplicate_transaction_ids:,}"
        )

    # Columns with highest null rates
    if ms.columns_with_missing:
        worst = ms.columns_with_missing[0]
        observations.append(
            f"Highest missing-value rate: column '{worst['column']}' "
            f"({worst['count']:,} nulls, {worst['pct']:.2f}%)"
        )

    # --- Quality concerns ---
    concerns: list[str] = []
    if ms.overall_missing_pct > 0:
        concerns.append(
            f"Missing values detected across {ms.columns_with_nulls} column(s) "
            f"({ms.total_missing_cells:,} cells total, {ms.overall_missing_pct:.2f}% overall)"
        )
    if ds.duplicate_rows > 0:
        concerns.append(
            f"{ds.duplicate_rows:,} fully duplicate rows found "
            f"({ds.duplicate_row_percentage:.2f}% of dataset)"
        )
    if ds.duplicate_transaction_ids is not None and ds.duplicate_transaction_ids > 0:
        concerns.append(
            f"{ds.duplicate_transaction_ids:,} duplicate transaction_id values "
            f"may indicate duplicate transactions"
        )
    if qi.completeness_pct < 95:
        concerns.append(
            f"Dataset completeness below 95% — "
            f"downstream analyses may be affected by missing values"
        )

    if not concerns:
        concerns.append("No significant data-quality concerns detected in this profiling pass.")

    return summary, observations, concerns


# ---------------------------------------------------------------------------
# LLM interpretation
# ---------------------------------------------------------------------------


def _llm_interpretation(
    result: ProfilingResult,
) -> tuple[Optional[str], Optional[list[str]], Optional[list[str]]]:
    """
    Ask the LLM to interpret the compact profiling result.
    Returns (summary, observations, concerns) or (None, None, None) on failure.
    """
    if not llm_client.is_available():
        return None, None, None

    compact = result.to_compact_dict()
    user_msg = (
        "Here are the pre-computed profiling statistics for this dataset:\n\n"
        + json.dumps(compact, indent=2, default=str)
        + "\n\nPlease provide your analysis following the format in your instructions."
    )

    raw_response = llm_client.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_message=user_msg,
        max_output_tokens=600,
    )

    if not raw_response:
        logger.warning("[ProfilingAgent] LLM returned empty response.")
        return None, None, None

    # Parse the structured sections from the LLM response
    summary      = ""
    observations: list[str] = []
    concerns:     list[str] = []

    current_section = None
    for line in raw_response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("SUMMARY"):
            current_section = "summary"
            continue
        if upper.startswith("KEY OBSERVATION"):
            current_section = "observations"
            continue
        if upper.startswith("QUALITY CONCERN"):
            current_section = "concerns"
            continue
        if stripped.startswith("---"):
            continue

        if current_section == "summary":
            summary += (" " if summary else "") + stripped
        elif current_section == "observations":
            cleaned = stripped.lstrip("•-* ").strip()
            if cleaned:
                observations.append(cleaned)
        elif current_section == "concerns":
            cleaned = stripped.lstrip("•-* ").strip()
            if cleaned:
                concerns.append(cleaned)

    if not summary and not observations:
        logger.warning("[ProfilingAgent] Could not parse LLM response sections.")
        return None, None, None

    return summary or None, observations or None, concerns or None


# ---------------------------------------------------------------------------
# Column findings builder
# ---------------------------------------------------------------------------


def _build_column_findings(result: ProfilingResult) -> list[dict[str, Any]]:
    """
    Build a concise column-level findings list for the agent output.
    Stays close to the profiling data without re-doing analysis.
    """
    findings = []
    for cd in result.column_details:
        entry: dict[str, Any] = {
            "column":         cd.column_name,
            "dtype":          cd.dtype,
            "null_count":     cd.null_count,
            "null_pct":       cd.null_percentage,
            "unique_count":   cd.unique_count,
            "uniqueness_pct": cd.uniqueness_pct,
            "is_numeric":     cd.is_numeric,
            "is_datetime":    cd.is_datetime,
        }
        if cd.is_numeric:
            entry.update({
                "min": cd.min, "max": cd.max,
                "mean": cd.mean, "std": cd.std,
            })
        elif cd.top_values:
            entry["top_values"] = cd.top_values
        findings.append(entry)
    return findings


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------


def run_profiling_agent(
    source: Union[str, Path, pd.DataFrame, ProfilingResult],
    dataset_name: Optional[str] = None,
    id_column: str = "transaction_id",
) -> AgentProfilingResponse:
    """
    Run the full Data Profiling Agent pipeline.

    Parameters
    ----------
    source : str | Path | DataFrame | ProfilingResult
        Dataset to profile.  If a ProfilingResult is passed, the profiling
        step is skipped and only interpretation is run.
    dataset_name : str, optional
        Override the dataset name label.
    id_column : str
        Primary-key column for duplicate ID detection.

    Returns
    -------
    AgentProfilingResponse
        Structured agent output.
    """
    # -----------------------------------------------------------------------
    # Step 1: Profile (or reuse existing result)
    # -----------------------------------------------------------------------
    if isinstance(source, ProfilingResult):
        result = source
    else:
        logger.info("[ProfilingAgent] Running Pandas profiling on '%s' ...", source)
        result = profile_dataset(source, dataset_name=dataset_name, id_column=id_column)

    # Handle profiling errors
    if result.profiling_error:
        return AgentProfilingResponse(
            dataset_name=result.dataset_name,
            source_path=result.source_path,
            summary=f"Profiling failed: {result.profiling_error}",
            key_observations=[],
            quality_concerns=[],
            quality_metrics={},
            column_findings=[],
            profiling_completed=False,
            llm_used=False,
            llm_model=None,
            profiling_error=result.profiling_error,
        )

    # -----------------------------------------------------------------------
    # Step 2: Interpret results
    # -----------------------------------------------------------------------
    llm_used  = False
    llm_model = None
    summary   = ""
    observations: list[str] = []
    concerns:     list[str] = []

    if llm_client.is_available():
        logger.info("[ProfilingAgent] Requesting LLM interpretation (%s) ...",
                    llm_client.model_name())
        llm_summary, llm_obs, llm_concerns = _llm_interpretation(result)

        if llm_summary is not None:
            summary      = llm_summary
            observations = llm_obs or []
            concerns     = llm_concerns or []
            llm_used     = True
            llm_model    = llm_client.model_name()
            logger.info("[ProfilingAgent] LLM interpretation successful.")
        else:
            logger.warning("[ProfilingAgent] LLM interpretation failed; using deterministic fallback.")

    if not llm_used:
        logger.info("[ProfilingAgent] Using deterministic interpretation.")
        summary, observations, concerns = _deterministic_interpretation(result)

    # -----------------------------------------------------------------------
    # Step 3: Assemble response
    # -----------------------------------------------------------------------
    quality_metrics = {
        "completeness_pct":     result.quality_indicators.completeness_pct,
        "uniqueness_pct":       result.quality_indicators.uniqueness_pct,
        "duplicate_pct":        result.quality_indicators.duplicate_pct,
        "total_issues_detected": result.quality_indicators.total_issues_detected,
        "missing_cells_total":  result.missing_value_summary.total_missing_cells,
        "missing_pct_overall":  result.missing_value_summary.overall_missing_pct,
        "duplicate_rows":       result.duplicate_summary.duplicate_rows,
        "duplicate_transaction_ids": result.duplicate_summary.duplicate_transaction_ids,
    }

    return AgentProfilingResponse(
        dataset_name=result.dataset_name,
        source_path=result.source_path,
        summary=summary,
        key_observations=observations,
        quality_concerns=concerns,
        quality_metrics=quality_metrics,
        column_findings=_build_column_findings(result),
        profiling_completed=True,
        llm_used=llm_used,
        llm_model=llm_model,
        profiling_error=None,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper (for CLI and manual testing)
# ---------------------------------------------------------------------------


def _pretty_print(response: AgentProfilingResponse) -> None:
    """Print a human-readable profiling report to stdout."""
    sep = "=" * 62
    print(sep)
    print(f"  DATA PROFILING REPORT: {response.dataset_name}")
    print(sep)

    status = "OK" if response.profiling_completed else "FAILED"
    llm_tag = f" | LLM: {response.llm_model}" if response.llm_used else " | LLM: deterministic"
    print(f"  Status: {status}{llm_tag}")

    if response.profiling_error:
        print(f"\n  ERROR: {response.profiling_error}\n")
        return

    print(f"\nSUMMARY\n{'-' * 62}")
    print(response.summary)

    print(f"\nKEY OBSERVATIONS\n{'-' * 62}")
    for obs in response.key_observations:
        print(f"  * {obs}")

    print(f"\nQUALITY CONCERNS\n{'-' * 62}")
    for concern in response.quality_concerns:
        print(f"  ! {concern}")

    print(f"\nQUALITY METRICS\n{'-' * 62}")
    for key, val in response.quality_metrics.items():
        if val is not None:
            print(f"  {key:<36} {val}")

    print(f"\nCOLUMN FINDINGS ({len(response.column_findings)} columns)\n{'-' * 62}")
    for cf in response.column_findings:
        dtype_tag  = "[NUM]" if cf.get("is_numeric") else "[CAT]"
        null_info  = f"nulls={cf['null_count']} ({cf['null_pct']}%)"
        uniq_info  = f"unique={cf['unique_count']}"
        print(f"  {dtype_tag} {cf['column']:<26} {null_info:<22} {uniq_info}")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args = sys.argv[1:]
    if not args:
        print("Usage: python -m app.agents.profiling_agent <path/to/dataset.csv>")
        print("\nExample datasets:")
        print("  data/raw/sales_clean.csv")
        print("  data/raw/sales_problematic.csv")
        print("  data/test/missing_values.csv")
        sys.exit(0)

    csv_path = args[0]
    print(f"\nProfiling: {csv_path}\n")

    response = run_profiling_agent(csv_path)
    _pretty_print(response)

    if len(args) > 1 and args[1] == "--json":
        print("\nJSON OUTPUT:")
        print(response.to_json())
