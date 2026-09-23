"""
app/agents/root_cause_agent.py
------------------------------
Phase 6 — Root Cause Investigation Agent.

Responsibilities:
  1. Accept a dataset path, DataFrame, or diagnostic results.
  2. Execute Profiling, Anomaly Detection, and Schema Analysis tools as needed.
  3. Run deterministic root-cause investigation (root_cause_tool.py).
  4. Interpret investigation findings — via LLM (Gemini) if available,
     otherwise via deterministic rule-based interpretation.
  5. Return a structured AgentRootCauseResponse.

Architecture principle:
  All EVIDENCE, candidates, confidence ratings, and numbers come from Python
  (root_cause_tool.py, profiling_tool.py, anomaly_tool.py, schema_tool.py).
  The LLM or rule-based interpreter ONLY EXPLAINS them.
  It does NOT invent evidence, claim unverified certainty, or suggest fixes.

Can be run directly via CLI:
    python -m app.agents.root_cause_agent data/raw/sales_problematic.csv

Or imported programmatically:
    from app.agents.root_cause_agent import run_root_cause_agent
    response = run_root_cause_agent("data/raw/sales_clean.csv")
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
from app.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Data Quality Root Cause Analyst.

Your task is to review structured root-cause investigation findings produced by
a Python evidence correlation engine, and write a concise, factual summary.

STRICT RULES:
1. Every statistic, count, candidate cause, and confidence rating MUST come
   directly from the investigation data provided to you. Do NOT invent values.
2. Do NOT claim certainty when evidence is inconclusive or confidence is LOW.
3. Do NOT claim access to unavailable database logs or external production systems.
4. Do NOT recommend final data fixes or corrective SQL scripts at this stage.
5. Clearly distinguish between facts (tool findings) and evidence interpretations.
6. Structure your narrative with:
   - SUMMARY: Overview of investigation status and primary root cause.
   - PRIMARY ROOT CAUSE: Cause name, confidence level, and supporting evidence.
   - CONTRIBUTING CAUSES: Other identified contributing factors (if any).
   - UNRESOLVED QUESTIONS: Key evidence gaps or questions requiring follow-up.
7. Keep total output under 400 words.
"""


# ---------------------------------------------------------------------------
# Agent Output Schema
# ---------------------------------------------------------------------------

@dataclass
class AgentRootCauseResponse:
    """Structured output returned by the Root Cause Investigation Agent."""
    dataset_name: str
    source_path: str
    investigation_status: str
    primary_root_cause: Dict[str, Any]
    contributing_causes: List[Dict[str, Any]]
    unrelated_findings: List[Dict[str, Any]]
    supporting_evidence: List[Dict[str, Any]]
    unresolved_questions: List[str]
    llm_interpretation: Optional[str] = None
    llm_used: bool = False
    raw_result: Optional[RootCauseResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "investigation_status": self.investigation_status,
            "primary_root_cause": self.primary_root_cause,
            "contributing_causes": self.contributing_causes,
            "unrelated_findings": self.unrelated_findings,
            "supporting_evidence": self.supporting_evidence,
            "unresolved_questions": self.unresolved_questions,
            "llm_interpretation": self.llm_interpretation,
            "llm_used": self.llm_used,
        }


# ---------------------------------------------------------------------------
# Fallback Rule-Based Interpretation
# ---------------------------------------------------------------------------

def _build_rule_based_interpretation(result: RootCauseResult) -> str:
    """Generate deterministic, human-readable narrative when LLM is unavailable."""
    lines: List[str] = []

    p_cause = result.primary_root_cause
    status = result.investigation_status

    if status == "no_issues_detected":
        lines.append(f"SUMMARY:\nThe dataset '{result.dataset_name}' exhibits high data quality across all diagnostic checks. No root causes were generated because zero anomalies or schema violations were detected.")
        lines.append("\nPRIMARY ROOT CAUSE:\n• Cause: No Data Quality Issues Detected (Confidence: HIGH)")
        lines.append("\nCONTRIBUTING CAUSES:\n• None")
        lines.append("\nUNRESOLVED QUESTIONS:\n• None")
        return "\n".join(lines)

    lines.append(f"SUMMARY:\n{result.summary}")

    lines.append("\nPRIMARY ROOT CAUSE:")
    cause_name = p_cause.get("cause", "Undetermined")
    conf = p_cause.get("confidence", "LOW")
    lines.append(f"• Cause: {cause_name} (Confidence: {conf})")
    if p_cause.get("reasoning"):
        lines.append(f"  Reasoning: {p_cause['reasoning']}")

    lines.append("\nCONTRIBUTING CAUSES:")
    if result.contributing_causes:
        for cc in result.contributing_causes:
            lines.append(f"• {cc.get('cause')} (Confidence: {cc.get('confidence')})")
    else:
        lines.append("• No secondary contributing causes identified.")

    lines.append("\nUNRESOLVED QUESTIONS:")
    if result.unresolved_questions:
        for q in result.unresolved_questions:
            lines.append(f"• {q}")
    else:
        lines.append("• No unresolved evidence questions remaining.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Agent Function
# ---------------------------------------------------------------------------

def run_root_cause_agent(
    dataset_input: Union[str, Path, pd.DataFrame],
    schema: Union[str, Path, Dict[str, Any]] = "sales",
    use_llm: bool = True,
    dataset_name: Optional[str] = None,
    profiling_result: Optional[ProfilingResult] = None,
    anomaly_result: Optional[AnomalyResult] = None,
    schema_result: Optional[SchemaAnalysisResult] = None,
) -> AgentRootCauseResponse:
    """
    Run the Root Cause Investigation Agent.

    Executes Profiling, Anomaly Detection, and Schema Analysis tools if results
    are not directly provided, then performs evidence correlation and ranking.

    Parameters
    ----------
    dataset_input : str, Path, or pd.DataFrame
    schema : str, Path, or Dict
    use_llm : bool
    dataset_name : str, optional
    profiling_result : ProfilingResult, optional
    anomaly_result : AnomalyResult, optional
    schema_result : SchemaAnalysisResult, optional

    Returns
    -------
    AgentRootCauseResponse
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

    # 1. Run upstream diagnostic tools if results were not pre-supplied
    prof_res = profiling_result or profile_dataset(df, dataset_name=dname)
    anom_res = anomaly_result or detect_anomalies(df, dataset_name=dname)
    sch_res = schema_result or analyze_schema(df, schema=schema, dataset_name=dname)

    # 2. Run deterministic root cause tool
    rc_result = investigate_root_cause(
        profiling_result=prof_res,
        anomaly_result=anom_res,
        schema_result=sch_res,
        dataset_name=dname,
    )

    # 3. Generate interpretation narrative (LLM or fallback)
    interpretation: str = ""
    llm_used: bool = False

    if use_llm and llm_client.is_available():
        try:
            user_prompt = (
                f"Perform root-cause analysis narrative synthesis for dataset '{dname}':\n\n"
                f"Root Cause Analysis Findings:\n{json.dumps(rc_result.to_dict(), indent=2)}"
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
            logger.warning(f"Gemini LLM root-cause synthesis failed, falling back to rule-based: {e}")

    if not llm_used:
        interpretation = _build_rule_based_interpretation(rc_result)

    return AgentRootCauseResponse(
        dataset_name=dname,
        source_path=source_path,
        investigation_status=rc_result.investigation_status,
        primary_root_cause=rc_result.primary_root_cause,
        contributing_causes=rc_result.contributing_causes,
        unrelated_findings=rc_result.unrelated_findings,
        supporting_evidence=rc_result.supporting_evidence,
        unresolved_questions=rc_result.unresolved_questions,
        llm_interpretation=interpretation,
        llm_used=llm_used,
        raw_result=rc_result,
    )


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.agents.root_cause_agent <dataset_csv_path> [schema_name_or_file]")
        sys.exit(1)

    csv_path = sys.argv[1]
    schema_arg = sys.argv[2] if len(sys.argv) > 2 else "sales"

    try:
        agent_response = run_root_cause_agent(csv_path, schema=schema_arg, use_llm=True)

        print("\n" + "=" * 70)
        print(f" ROOT CAUSE INVESTIGATION REPORT: {agent_response.dataset_name}")
        print("=" * 70)
        print(f"Investigation Status: {agent_response.investigation_status}")
        p_cause = agent_response.primary_root_cause
        print(f"Primary Root Cause: {p_cause.get('cause', 'Undetermined')} (Confidence: {p_cause.get('confidence', 'LOW')})")
        print(f"Supporting Evidence Count: {len(agent_response.supporting_evidence)}")
        print("-" * 70)

        if agent_response.contributing_causes:
            print("\n[CONTRIBUTING CAUSES]")
            for cc in agent_response.contributing_causes:
                print(f"  - {cc.get('cause')} (Confidence: {cc.get('confidence')})")

        if agent_response.unresolved_questions:
            print("\n[UNRESOLVED QUESTIONS]")
            for q in agent_response.unresolved_questions:
                print(f"  - {q}")

        print("\n" + "-" * 70)
        print(f"[INVESTIGATION NARRATIVE] (LLM used: {agent_response.llm_used})")
        print("-" * 70)
        print(agent_response.llm_interpretation)
        print("=" * 70 + "\n")

    except Exception as err:
        print(f"Error running Root Cause Agent: {err}", file=sys.stderr)
        sys.exit(2)
