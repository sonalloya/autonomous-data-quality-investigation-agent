"""
app/agents/validation_agent.py
------------------------------
Phase 8 — Validation Agent.

Responsibilities:
  1. Accept a post-correction dataset path or DataFrame (and optional original dataset).
  2. Run the deterministic independent validation tool (validation_tool.py).
  3. Synthesize human-readable validation report — via LLM (Gemini) if available,
     otherwise via deterministic rule-based interpretation.
  4. Return a structured AgentValidationResponse.

Authoritative Verdict Guarantee:
  All PASS / PARTIAL / FAILED / NOT_APPLICABLE statuses and numbers come from Python.
  The LLM or rule-based interpreter ONLY EXPLAINS them.
  Gemini CANNOT override or modify deterministic validation results.

Can be run directly via CLI:
    python -m app.agents.validation_agent <post_correction_csv_path> [original_csv_path]

Or imported programmatically:
    from app.agents.validation_agent import run_validation_agent
    response = run_validation_agent(post_df, orig_df)
"""

from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.tools.correction_tool import CorrectionPlanResult
from app.tools.validation_tool import validate_correction, ValidationResult
from app.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an Independent Data Quality Validation Analyst.

Your task is to review structured validation check results produced by a Python
independent validation engine, and synthesize a clear, objective validation report.

STRICT RULES:
1. The overall validation status (PASSED, PARTIAL, FAILED, or NOT_APPLICABLE)
   is AUTHORITATIVE and produced by Python. You MUST NOT change or override it.
2. Every statistic, check count, expected vs actual value, and improvement percentage
   MUST come directly from the validation data provided. Do NOT invent numbers.
3. Structure your response as follows:
   - OVERALL STATUS: Overall verdict (PASSED / PARTIAL / FAILED / NOT_APPLICABLE) and short summary.
   - VALIDATION CHECKS SUMMARY: List of checks performed with status [PASS/FAIL/PARTIAL].
   - BEFORE vs AFTER IMPACT: Summary of empirical metrics before and after correction.
   - UNRESOLVED ISSUES: Key lingering defects requiring further attention (if any).
4. Keep total output under 400 words.
"""


# ---------------------------------------------------------------------------
# Agent Output Schema
# ---------------------------------------------------------------------------

@dataclass
class AgentValidationResponse:
    """Structured output returned by the Validation Agent."""
    dataset_name: str
    source_path: str
    validation_status: str  # 'PASSED', 'PARTIAL', 'FAILED', 'NOT_APPLICABLE'
    total_checks: int
    passed_checks: int
    failed_checks: int
    partial_checks: int
    checks: List[Dict[str, Any]]
    unresolved_issues: List[str]
    before_after_comparison: Dict[str, Any]
    llm_interpretation: Optional[str] = None
    llm_used: bool = False
    raw_result: Optional[ValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "validation_status": self.validation_status,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "partial_checks": self.partial_checks,
            "checks": self.checks,
            "unresolved_issues": self.unresolved_issues,
            "before_after_comparison": self.before_after_comparison,
            "llm_interpretation": self.llm_interpretation,
            "llm_used": self.llm_used,
        }


# ---------------------------------------------------------------------------
# Fallback Rule-Based Interpretation
# ---------------------------------------------------------------------------

def _build_rule_based_interpretation(result: ValidationResult) -> str:
    """Generate deterministic, human-readable narrative when LLM is unavailable."""
    lines: List[str] = []

    lines.append(f"OVERALL STATUS: {result.validation_status}")
    lines.append(f"Summary: {result.summary}")

    if result.validation_status == "NOT_APPLICABLE":
        lines.append("\nVALIDATION CHECKS SUMMARY:\n• No validation checks required (dataset is clean).")
        lines.append("\nBEFORE vs AFTER IMPACT:\n• N/A")
        lines.append("\nUNRESOLVED ISSUES:\n• None")
        return "\n".join(lines)

    lines.append("\nVALIDATION CHECKS SUMMARY:")
    for c in result.checks:
        status_tag = f"[{c['status']}]"
        lines.append(f"• {status_tag} {c['check_name']} ({c['category']})")
        lines.append(f"  Expected: {c['expected']}")
        lines.append(f"  Actual: {c['actual']}")
        lines.append(f"  Evidence: {c['evidence']}")

    lines.append("\nBEFORE vs AFTER IMPACT:")
    if result.before_after_comparison:
        for k, v in result.before_after_comparison.items():
            if "improvement_pct" in v:
                lines.append(f"• {k.capitalize()}: Before = {v['before']}, After = {v['after']} ({v['improvement_pct']}% improvement)")
            else:
                lines.append(f"• {k.capitalize()}: Before = {v.get('before')}, After = {v.get('after')}")
    else:
        lines.append("• No baseline comparison data available.")

    lines.append("\nUNRESOLVED ISSUES:")
    if result.unresolved_issues:
        for u in result.unresolved_issues:
            lines.append(f"• {u}")
    else:
        lines.append("• Zero unresolved issues remaining.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Agent Function
# ---------------------------------------------------------------------------

def run_validation_agent(
    post_correction_input: Union[str, Path, pd.DataFrame],
    original_dataset_input: Optional[Union[str, Path, pd.DataFrame]] = None,
    correction_plan: Optional[CorrectionPlanResult] = None,
    schema: Union[str, Path, Dict[str, Any]] = "sales",
    use_llm: bool = True,
    dataset_name: Optional[str] = None,
) -> AgentValidationResponse:
    """
    Run the Independent Validation Agent.

    Parameters
    ----------
    post_correction_input : str, Path, or pd.DataFrame
        Dataset after correction/remediation.
    original_dataset_input : str, Path, or pd.DataFrame, optional
        Original baseline dataset prior to correction.
    correction_plan : CorrectionPlanResult, optional
        Phase 7 correction plan containing validation criteria.
    schema : str, Path, or Dict
        Expected schema definition.
    use_llm : bool
        If True, uses Gemini LLM for narrative synthesis.
    dataset_name : str, optional

    Returns
    -------
    AgentValidationResponse
    """
    source_path = ""
    if isinstance(post_correction_input, (str, Path)):
        source_path = str(post_correction_input)
        p = Path(post_correction_input)
        if not p.is_file():
            raise FileNotFoundError(f"Post-correction dataset file not found: {source_path}")
        dname = dataset_name or p.name
        try:
            post_df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to read post-correction CSV '{source_path}': {e}")
    elif isinstance(post_correction_input, pd.DataFrame):
        post_df = post_correction_input
        dname = dataset_name or "in_memory_post_correction_df"
    else:
        raise ValueError(f"Unsupported post-correction dataset type: {type(post_correction_input)}")

    orig_df: Optional[pd.DataFrame] = None
    if isinstance(original_dataset_input, (str, Path)):
        p_orig = Path(original_dataset_input)
        if p_orig.is_file():
            orig_df = pd.read_csv(p_orig, low_memory=False)
    elif isinstance(original_dataset_input, pd.DataFrame):
        orig_df = original_dataset_input

    # 1. Run deterministic validation tool
    val_result = validate_correction(
        post_correction_df=post_df,
        original_df=orig_df,
        correction_plan=correction_plan,
        schema=schema,
        dataset_name=dname,
    )

    # 2. Synthesize narrative report (LLM or fallback)
    interpretation: str = ""
    llm_used: bool = False

    if use_llm and llm_client.is_available():
        try:
            user_prompt = (
                f"Synthesize independent validation report for dataset '{dname}':\n\n"
                f"Validation Check Results:\n{json.dumps(val_result.to_dict(), indent=2)}"
            )
            response_text = llm_client.generate_text(
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.1,
            )
            if response_text and len(response_text.strip()) > 0:
                interpretation = response_text.strip()
                llm_used = True
        except Exception as e:
            logger.warning(f"Gemini LLM validation synthesis failed, falling back to rule-based: {e}")

    if not llm_used:
        interpretation = _build_rule_based_interpretation(val_result)

    return AgentValidationResponse(
        dataset_name=dname,
        source_path=source_path,
        validation_status=val_result.validation_status,
        total_checks=val_result.total_checks,
        passed_checks=val_result.passed_checks,
        failed_checks=val_result.failed_checks,
        partial_checks=val_result.partial_checks,
        checks=val_result.checks,
        unresolved_issues=val_result.unresolved_issues,
        before_after_comparison=val_result.before_after_comparison,
        llm_interpretation=interpretation,
        llm_used=llm_used,
        raw_result=val_result,
    )


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.agents.validation_agent <post_correction_csv_path> [original_csv_path] [schema_name]")
        sys.exit(1)

    post_csv = sys.argv[1]
    orig_csv = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith(".json") else None
    schema_arg = sys.argv[3] if len(sys.argv) > 3 else "sales"

    try:
        agent_response = run_validation_agent(
            post_correction_input=post_csv,
            original_dataset_input=orig_csv,
            schema=schema_arg,
            use_llm=True,
        )

        print("\n" + "=" * 70)
        print(f" INDEPENDENT VALIDATION REPORT: {agent_response.dataset_name}")
        print("=" * 70)
        print(f"Validation Status: {agent_response.validation_status}")
        print(f"Passed Checks: {agent_response.passed_checks} / {agent_response.total_checks}")
        print(f"Failed Checks: {agent_response.failed_checks} | Partial Checks: {agent_response.partial_checks}")
        print("-" * 70)

        print("\n" + "-" * 70)
        print(f"[REPORT NARRATIVE] (LLM used: {agent_response.llm_used})")
        print("-" * 70)
        print(agent_response.llm_interpretation)
        print("=" * 70 + "\n")

    except Exception as err:
        print(f"Error running Validation Agent: {err}", file=sys.stderr)
        sys.exit(2)
