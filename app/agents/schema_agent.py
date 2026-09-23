"""
app/agents/schema_agent.py
--------------------------
Phase 5 — Schema Analysis Agent.

Responsibilities:
  1. Accept a dataset path or DataFrame (and expected schema).
  2. Run the deterministic schema analysis tool (schema_tool.py).
  3. Interpret findings — via LLM (Gemini) if available, otherwise via
     deterministic rule-based interpretation.
  4. Return a structured AgentSchemaResponse.

Architecture principle:
  All FACTS, counts, and boolean flags come from Python (schema_tool.py).
  The LLM or rule-based interpreter ONLY EXPLAINS them.
  It does NOT claim root cause or recommend corrective action.

Can be run directly via CLI:
    python -m app.agents.schema_agent data/raw/sales_problematic.csv

Or imported programmatically:
    from app.agents.schema_agent import run_schema_agent
    response = run_schema_agent("data/raw/sales_clean.csv")
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

from app.config.schema_config import SchemaDefinition, load_schema
from app.tools.schema_tool import analyze_schema, SchemaAnalysisResult
from app.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Data Quality Schema Analyst.

Your task is to review structured schema-analysis results produced by a Python
validation tool, and write a concise, factual interpretation.

STRICT RULES:
1. Every number, column name, and statistic in your response MUST come directly
   from the schema analysis data provided. Do NOT invent or estimate any value.
2. Do NOT perform calculations yourself — all results are pre-computed.
3. Do NOT identify or claim an overall root cause. Root cause reasoning belongs to a later step.
4. Do NOT recommend final corrections or data fixes.
5. Clearly label what is a factual schema finding vs. an observation.
6. Be concise: 3–6 bullet-point observations plus a brief paragraph summary.
   Total output: under 350 words.
7. Rank observations by severity (CRITICAL/HIGH first).
8. If the schema is valid and no issues were found, state that clearly and briefly.

Structure your response exactly as follows:
---
SUMMARY:
<one short paragraph summarising schema validity and overall structural health>

KEY FINDINGS:
• <finding 1 — most severe first>
• <finding 2>
• <finding 3>
(add up to 3 more if warranted)

SCHEMA CONCERNS:
• <concern, or "No significant schema violations detected." if none>
---
"""


# ---------------------------------------------------------------------------
# Agent output schema
# ---------------------------------------------------------------------------

@dataclass
class AgentSchemaResponse:
    """Structured response object for Schema Agent."""
    dataset_name: str
    source_path: str
    total_rows: int
    total_columns: int
    schema_valid: bool
    summary: Dict[str, int]
    missing_columns: List[Dict[str, Any]]
    unexpected_columns: List[Dict[str, Any]]
    type_mismatches: List[Dict[str, Any]]
    format_issues: List[Dict[str, Any]]
    nullability_violations: List[Dict[str, Any]]
    constraint_violations: List[Dict[str, Any]]
    llm_interpretation: Optional[str] = None
    llm_used: bool = False
    raw_result: Optional[SchemaAnalysisResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "schema_valid": self.schema_valid,
            "summary": self.summary,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "type_mismatches": self.type_mismatches,
            "format_issues": self.format_issues,
            "nullability_violations": self.nullability_violations,
            "constraint_violations": self.constraint_violations,
            "llm_interpretation": self.llm_interpretation,
            "llm_used": self.llm_used,
        }


# ---------------------------------------------------------------------------
# Fallback Rule-Based Interpretation
# ---------------------------------------------------------------------------

def _build_rule_based_interpretation(result: SchemaAnalysisResult) -> str:
    """Generate deterministic, human-readable narrative when LLM is unavailable."""
    lines: List[str] = []

    if result.schema_valid:
        lines.append(f"SUMMARY:\nThe dataset '{result.dataset_name}' conforms strictly to its expected schema across all {result.total_columns} columns and {result.total_rows} records. No critical structural or format violations were detected.")
        lines.append("\nKEY FINDINGS:")
        lines.append(f"• Expected columns matched: {len(result.expected_columns)} of {len(result.expected_columns)} required columns present.")
        lines.append("• Logical data types: All columns match expected types without type mismatches.")
        lines.append("• Constraints & Nullability: Zero required-field null violations or range constraint failures.")
        lines.append("\nSCHEMA CONCERNS:\n• No significant schema violations detected.")
        return "\n".join(lines)

    s = result.summary
    lines.append(
        f"SUMMARY:\nThe dataset '{result.dataset_name}' contains schema violations causing schema validation to fail. "
        f"A total of {s['total_issues']} schema issues were detected ({s['critical']} Critical, {s['high']} High, {s['medium']} Medium, {s['low']} Low) across {result.total_rows} rows."
    )

    lines.append("\nKEY FINDINGS:")
    # Missing columns
    for item in result.missing_columns:
        lines.append(f"• [CRITICAL/HIGH] Missing Column: Column '{item['column']}' is required by schema but absent from dataset.")

    # Type mismatches
    for item in result.type_mismatches:
        lines.append(f"• [HIGH] Type Mismatch: Column '{item['column']}' — {item['description']}")

    # Format issues
    for item in result.format_issues:
        ev = item.get("evidence", {})
        lines.append(f"• [HIGH/MEDIUM] Format Issue: Column '{item['column']}' — {ev.get('invalid_count', 0)} unparseable dates, {ev.get('format_inconsistency_count', 0)} inconsistent formats.")

    # Nullability violations
    for item in result.nullability_violations:
        ev = item.get("evidence", {})
        lines.append(f"• [HIGH] Nullability Violation: Column '{item['column']}' contains {ev.get('null_count', 0)} null/missing values in a non-nullable field.")

    # Constraint violations
    for item in result.constraint_violations:
        ev = item.get("evidence", {})
        lines.append(f"• [HIGH/MEDIUM] Constraint Violation: Column '{item['column']}' — {item['description']}")

    # Unexpected columns
    for item in result.unexpected_columns:
        lines.append(f"• [LOW] Unexpected Column: Column '{item['column']}' is present in data but not defined in expected schema.")

    lines.append("\nSCHEMA CONCERNS:")
    if s['critical'] > 0 or s['high'] > 0:
        lines.append(f"• Severe schema structural failures ({s['critical']} critical, {s['high']} high) require resolution prior to downstream processing.")
    else:
        lines.append("• Minor schema deviations detected.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Agent Function
# ---------------------------------------------------------------------------

def run_schema_agent(
    dataset_input: Union[str, Path, pd.DataFrame],
    schema: Union[SchemaDefinition, Dict[str, Any], str, Path] = "sales",
    use_llm: bool = True,
    dataset_name: Optional[str] = None,
) -> AgentSchemaResponse:
    """
    Run the Schema Analysis Agent on a dataset.

    Parameters
    ----------
    dataset_input : str, Path, or pd.DataFrame
        Path to CSV file or existing DataFrame.
    schema : SchemaDefinition, Dict, str, or Path
        Expected schema definition or preset name (default "sales").
    use_llm : bool
        If True, attempts Gemini interpretation when configured.
    dataset_name : str, optional
        Custom dataset name.

    Returns
    -------
    AgentSchemaResponse
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

    # 1. Run deterministic schema analysis tool
    result = analyze_schema(df, schema=schema, dataset_name=dname)

    # 2. Generate interpretation (LLM or fallback)
    interpretation: str = ""
    llm_used: bool = False

    if use_llm and llm_client.is_available():
        try:
            user_prompt = (
                f"Analyze the following schema validation results for dataset '{dname}':\n\n"
                f"Schema Analysis Output:\n{json.dumps(result.to_dict(), indent=2)}"
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
            logger.warning(f"Gemini LLM interpretation failed, falling back to rule-based: {e}")

    if not llm_used:
        interpretation = _build_rule_based_interpretation(result)

    return AgentSchemaResponse(
        dataset_name=dname,
        source_path=source_path,
        total_rows=result.total_rows,
        total_columns=result.total_columns,
        schema_valid=result.schema_valid,
        summary=result.summary,
        missing_columns=result.missing_columns,
        unexpected_columns=result.unexpected_columns,
        type_mismatches=result.type_mismatches,
        format_issues=result.format_issues,
        nullability_violations=result.nullability_violations,
        constraint_violations=result.constraint_violations,
        llm_interpretation=interpretation,
        llm_used=llm_used,
        raw_result=result,
    )


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.agents.schema_agent <dataset_csv_path> [schema_name_or_file]")
        sys.exit(1)

    csv_path = sys.argv[1]
    schema_arg = sys.argv[2] if len(sys.argv) > 2 else "sales"

    try:
        agent_response = run_schema_agent(csv_path, schema=schema_arg, use_llm=True)
        
        print("\n" + "=" * 70)
        print(f" SCHEMA ANALYSIS AGENT REPORT: {agent_response.dataset_name}")
        print("=" * 70)
        print(f"Total Rows: {agent_response.total_rows} | Total Columns: {agent_response.total_columns}")
        print(f"Schema Valid: {agent_response.schema_valid}")
        print(f"Issues Summary: {agent_response.summary}")
        print("-" * 70)

        print("\n[DETAILED FINDINGS]")
        if agent_response.missing_columns:
            print(f"  Missing Columns ({len(agent_response.missing_columns)}):")
            for item in agent_response.missing_columns:
                print(f"    - {item['column']}: {item['description']}")
        if agent_response.unexpected_columns:
            print(f"  Unexpected Columns ({len(agent_response.unexpected_columns)}):")
            for item in agent_response.unexpected_columns:
                print(f"    - {item['column']}: {item['description']}")
        if agent_response.type_mismatches:
            print(f"  Type Mismatches ({len(agent_response.type_mismatches)}):")
            for item in agent_response.type_mismatches:
                print(f"    - {item['column']}: {item['description']}")
        if agent_response.format_issues:
            print(f"  Format Issues ({len(agent_response.format_issues)}):")
            for item in agent_response.format_issues:
                print(f"    - {item['column']}: {item['description']}")
        if agent_response.nullability_violations:
            print(f"  Nullability Violations ({len(agent_response.nullability_violations)}):")
            for item in agent_response.nullability_violations:
                print(f"    - {item['column']}: {item['description']}")
        if agent_response.constraint_violations:
            print(f"  Constraint Violations ({len(agent_response.constraint_violations)}):")
            for item in agent_response.constraint_violations:
                print(f"    - {item['column']}: {item['description']}")

        print("\n" + "-" * 70)
        print(f"[INTERPRETATION] (LLM used: {agent_response.llm_used})")
        print("-" * 70)
        print(agent_response.llm_interpretation)
        print("=" * 70 + "\n")

    except Exception as err:
        print(f"Error running Schema Agent: {err}", file=sys.stderr)
        sys.exit(2)
