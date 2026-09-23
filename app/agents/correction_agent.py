"""
app/agents/correction_agent.py
------------------------------
Phase 7 — Correction Recommendation Agent.

Responsibilities:
  1. Accept a dataset path, DataFrame, or RootCauseResult.
  2. Execute upstream diagnostic tools and Root Cause Agent as needed.
  3. Run deterministic correction strategy generator (correction_tool.py).
  4. Synthesize human-readable action plan — via LLM (Gemini) if available,
     otherwise via deterministic rule-based interpretation.
  5. Return a structured AgentCorrectionResponse.

Safety Guarantee:
  This agent produces RECOMMENDATIONS ONLY.
  It MUST NOT modify, update, or delete production data or raw datasets.
  All data-altering action steps explicitly flag `human_approval_required: True`.

Can be run directly via CLI:
    python -m app.agents.correction_agent data/raw/sales_problematic.csv

Or imported programmatically:
    from app.agents.correction_agent import run_correction_agent
    response = run_correction_agent("data/raw/sales_clean.csv")
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

from app.tools.profiling_tool import profile_dataset, ProfilingResult
from app.tools.anomaly_tool import detect_anomalies, AnomalyResult
from app.tools.schema_tool import analyze_schema, SchemaAnalysisResult
from app.tools.root_cause_tool import investigate_root_cause, RootCauseResult
from app.tools.correction_tool import generate_correction_plan, CorrectionPlanResult
from app.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Data Quality Correction Analyst.

Your task is to review structured correction recommendations produced by a Python
remediation planning engine, and write a concise, prioritized action plan.

STRICT SAFETY RULES:
1. Every recommendation, action step, priority, and validation criterion MUST come
   directly from the correction data provided. Do NOT invent new recommendations.
2. Explicitly state that these are RECOMMENDATIONS ONLY and that NO data has been
   modified, deleted, or updated yet.
3. Highlight all recommendations requiring EXPLICIT HUMAN APPROVAL (`human_approval_required: True`).
4. Structure your response as follows:
   - SUMMARY: Overview of correction plan status and total recommendations.
   - RECOMMENDED ACTIONS: Numbered list ordered by priority (CRITICAL/HIGH first).
     For each action, include: Priority, Issue, Root Cause, Action Steps, Validation Criteria, and Human Approval status.
   - RISK & APPROVAL SUMMARY: Summary of human approval requirements.
5. Keep total output under 450 words.
"""


# ---------------------------------------------------------------------------
# Agent Output Schema
# ---------------------------------------------------------------------------

@dataclass
class AgentCorrectionResponse:
    """Structured response object returned by the Correction Agent."""
    dataset_name: str
    source_path: str
    status: str
    summary: str
    recommendations: List[Dict[str, Any]]
    total_recommendations: int
    human_approval_summary: Dict[str, int]
    llm_interpretation: Optional[str] = None
    llm_used: bool = False
    raw_result: Optional[CorrectionPlanResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "status": self.status,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "total_recommendations": self.total_recommendations,
            "human_approval_summary": self.human_approval_summary,
            "llm_interpretation": self.llm_interpretation,
            "llm_used": self.llm_used,
        }


# ---------------------------------------------------------------------------
# Fallback Rule-Based Interpretation
# ---------------------------------------------------------------------------

def _build_rule_based_interpretation(result: CorrectionPlanResult) -> str:
    """Generate deterministic, human-readable narrative when LLM is unavailable."""
    lines: List[str] = []

    if result.status == "no_action_required":
        lines.append(f"SUMMARY:\nNo corrective action required for dataset '{result.dataset_name}'. Data quality is clean.")
        lines.append("\nRECOMMENDED ACTIONS:\n• None required.")
        lines.append("\nRISK & APPROVAL SUMMARY:\n• Safe analysis complete. 0 human approvals required.")
        return "\n".join(lines)

    lines.append(f"SUMMARY:\n{result.summary}")

    lines.append("\nRECOMMENDED ACTIONS:")
    for idx, rec in enumerate(result.recommendations, 1):
        prio = rec.get("priority", "MEDIUM")
        issue = rec.get("issue", "")
        cause = rec.get("root_cause", "")
        appr = "Required" if rec.get("human_approval_required") else "Safe Analysis Only"
        
        lines.append(f"\n{idx}. [{prio}] {issue}")
        lines.append(f"   Root Cause: {cause}")
        lines.append(f"   Human Approval: {appr} (Risk: {rec.get('risk_level', 'MEDIUM')})")
        lines.append("   Action Steps:")
        for step in rec.get("action_steps", []):
            lines.append(f"     - {step}")
        lines.append(f"   Expected Outcome: {rec.get('expected_outcome', '')}")
        lines.append("   Validation Criteria:")
        for vc in rec.get("validation_criteria", []):
            lines.append(f"     * {vc}")

    req = result.human_approval_summary.get("required", 0)
    lines.append(f"\nRISK & APPROVAL SUMMARY:\n• Total recommendations: {result.total_recommendations}")
    lines.append(f"• Human approval required: {req} action(s) before dataset remediation.")
    lines.append("• Safety guarantee: System has NOT modified any underlying data.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Agent Function
# ---------------------------------------------------------------------------

def run_correction_agent(
    dataset_input: Union[str, Path, pd.DataFrame],
    schema: Union[str, Path, Dict[str, Any]] = "sales",
    use_llm: bool = True,
    dataset_name: Optional[str] = None,
    root_cause_result: Optional[RootCauseResult] = None,
) -> AgentCorrectionResponse:
    """
    Run the Correction Recommendation Agent.

    Executes upstream diagnostic tools and Root Cause Agent if results are not
    pre-supplied, then generates deterministic, prioritized correction plan.

    Parameters
    ----------
    dataset_input : str, Path, or pd.DataFrame
    schema : str, Path, or Dict
    use_llm : bool
    dataset_name : str, optional
    root_cause_result : RootCauseResult, optional

    Returns
    -------
    AgentCorrectionResponse
    """
    source_path = ""
    if isinstance(dataset_input, (str, Path)):
        source_path = str(dataset_input)
        p = Path(dataset_input)
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {source_path}")
        dname = dataset_name or p.name
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to read CSV dataset '{source_path}': {e}")
    elif isinstance(dataset_input, pd.DataFrame):
        df = dataset_input
        dname = dataset_name or "in_memory_dataframe"
    else:
        raise ValueError(f"Unsupported dataset input type: {type(dataset_input)}")

    # 1. Run Root Cause Agent if result not pre-supplied
    if root_cause_result is None:
        prof_res = profile_dataset(df, dataset_name=dname)
        anom_res = detect_anomalies(df, dataset_name=dname)
        sch_res = analyze_schema(df, schema=schema, dataset_name=dname)
        rc_result = investigate_root_cause(prof_res, anom_res, sch_res, dataset_name=dname)
    else:
        rc_result = root_cause_result

    # 2. Run deterministic correction tool
    plan_result = generate_correction_plan(rc_result, dataset_name=dname)

    # 3. Generate narrative interpretation (LLM or fallback)
    interpretation: str = ""
    llm_used: bool = False

    if use_llm and llm_client.is_available():
        try:
            user_prompt = (
                f"Synthesize correction action plan for dataset '{dname}':\n\n"
                f"Correction Plan Findings:\n{json.dumps(plan_result.to_dict(), indent=2)}"
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
            logger.warning(f"Gemini LLM correction synthesis failed, falling back to rule-based: {e}")

    if not llm_used:
        interpretation = _build_rule_based_interpretation(plan_result)

    return AgentCorrectionResponse(
        dataset_name=dname,
        source_path=source_path,
        status=plan_result.status,
        summary=plan_result.summary,
        recommendations=plan_result.recommendations,
        total_recommendations=plan_result.total_recommendations,
        human_approval_summary=plan_result.human_approval_summary,
        llm_interpretation=interpretation,
        llm_used=llm_used,
        raw_result=plan_result,
    )


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.agents.correction_agent <dataset_csv_path> [schema_name_or_file]")
        sys.exit(1)

    csv_path = sys.argv[1]
    schema_arg = sys.argv[2] if len(sys.argv) > 2 else "sales"

    try:
        agent_response = run_correction_agent(csv_path, schema=schema_arg, use_llm=True)

        print("\n" + "=" * 70)
        print(f" CORRECTION RECOMMENDATION REPORT: {agent_response.dataset_name}")
        print("=" * 70)
        print(f"Correction Status: {agent_response.status}")
        print(f"Total Recommendations: {agent_response.total_recommendations}")
        print(f"Human Approval Summary: {agent_response.human_approval_summary}")
        print("-" * 70)

        print("\n" + "-" * 70)
        print(f"[ACTION PLAN] (LLM used: {agent_response.llm_used})")
        print("-" * 70)
        print(agent_response.llm_interpretation)
        print("=" * 70 + "\n")

    except Exception as err:
        print(f"Error running Correction Agent: {err}", file=sys.stderr)
        sys.exit(2)
