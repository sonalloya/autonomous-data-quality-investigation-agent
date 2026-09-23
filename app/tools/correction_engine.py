"""
app/tools/correction_engine.py
------------------------------
Phase 12 — Generic Deterministic Correction Engine.

Executes deterministic, Python/Pandas-based dataset remediation on ANY CSV dataset.

Core Principles & Guarantees:
  1. IMMUTABILITY GUARANTEE: The original dataset is NEVER modified or overwritten.
  2. DETERMINISTIC EXECUTION: Modifies data exclusively using rule-based Python/Pandas logic.
     Does NOT use an LLM to rewrite CSV files.
  3. NO SILENT GUESSING: Ambiguous or unrecoverable issues are left unresolved and reported.
  4. GENERIC & EXTENSIBLE: Works with configured schemas (e.g., sales) or generic unknown CSVs.
  5. COMPREHENSIVE CORRECTION SUITE:
     - Duplicate removal (retaining canonical records)
     - Date format normalization (to standard ISO %Y-%m-%d)
     - Safe numeric type conversion & suffix stripping ("$12.50", "10 units")
     - Safe numeric constraint & sign inversion fixes
     - Categorical domain normalization & alias mapping
     - Computable field recovery (e.g., total_amount = quantity * unit_price)
     - Identifier preservation (never invents missing IDs)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.config.schema_config import ColumnSchema, SchemaDefinition, load_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CorrectionActionRecord:
    """Record of an individual correction applied to the dataset."""
    correction_type: str  # 'duplicate_removal', 'date_normalization', 'type_conversion', 'constraint_fix', 'categorical_mapping', 'computed_field'
    column: Optional[str]
    row_index: Optional[int]
    original_value: Any
    corrected_value: Any
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionExecutionResult:
    """Structured result returned by the Correction Engine."""
    dataset_name: str
    original_dataset_path: str
    corrected_dataset_path: Optional[str]
    status: str  # 'CORRECTED', 'PARTIALLY_CORRECTED', 'CLEAN_NO_ACTION_REQUIRED', 'FAILED'
    corrections_applied: int
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    issues_resolved: int = 0
    issues_remaining: int = 0
    unresolved_issues: List[str] = field(default_factory=list)
    is_clean: bool = False
    summary: str = ""
    data_change_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "original_dataset_path": self.original_dataset_path,
            "corrected_dataset_path": self.corrected_dataset_path,
            "status": self.status,
            "corrections_applied": self.corrections_applied,
            "corrections": self.corrections,
            "issues_resolved": self.issues_resolved,
            "issues_remaining": self.issues_remaining,
            "unresolved_issues": self.unresolved_issues,
            "is_clean": self.is_clean,
            "summary": self.summary,
            "data_change_summary": self.data_change_summary,
        }


# ---------------------------------------------------------------------------
# Helper Normalizers
# ---------------------------------------------------------------------------

def _is_null_or_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "null", "none", "nan", "na", "<na>", "n/a"):
        return True
    return False


def _clean_numeric_string(val: Any) -> Tuple[Optional[float], bool]:
    """
    Attempt unambiguous conversion of a string into a float/int.
    Strips currency symbols ($ € £ ¥), commas, whitespace, and clean unit suffixes.
    Returns (cleaned_value, is_converted).
    """
    if _is_null_or_empty(val):
        return None, False

    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val), False

    s = str(val).strip()

    # Pattern for numbers with optional currency prefix/suffix and units
    pattern = r"^\s*[\$€£¥]?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?:units|items|usd|eur|gbp|lbs|kg|m|ea)?\s*$"
    match = re.match(pattern, s, re.IGNORECASE)
    if match:
        raw_num = match.group(1).replace(",", "")
        try:
            num = float(raw_num)
            return num, True
        except ValueError:
            pass

    return None, False


def _parse_and_normalize_date(val: Any, target_format: str = "%Y-%m-%d") -> Tuple[Optional[str], bool, bool]:
    """
    Safely parse date string into standard ISO %Y-%m-%d.
    Returns (normalized_str, was_converted, is_unparseable).
    """
    if _is_null_or_empty(val):
        return None, False, False

    s = str(val).strip()

    # Check if already strictly matching target_format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            pd.to_datetime(s, format="%Y-%m-%d", errors="raise")
            return s, False, False
        except Exception:
            return s, False, True

    # Common recognizable formats to try explicitly
    known_formats = [
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y.%m.%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]

    for fmt in known_formats:
        try:
            dt = pd.to_datetime(s, format=fmt, errors="raise")
            normalized = dt.strftime(target_format)
            return normalized, True, False
        except Exception:
            continue

    # Fallback to general parsing
    try:
        dt = pd.to_datetime(s, errors="raise")
        normalized = dt.strftime(target_format)
        return normalized, True, False
    except Exception:
        # Invalid / unparseable date — do NOT guess
        return s, False, True


# ---------------------------------------------------------------------------
# Core Correction Engine Function
# ---------------------------------------------------------------------------

def apply_corrections(
    dataset_input: Union[str, Path, pd.DataFrame],
    schema: Union[str, Path, Dict[str, Any], SchemaDefinition, None] = "sales",
    dataset_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, CorrectionExecutionResult]:
    """
    Execute deterministic data quality corrections on input CSV / DataFrame.

    Parameters
    ----------
    dataset_input : str, Path, or pd.DataFrame
        Source dataset. If a file path, the source file is NEVER modified.
    schema : str, Path, Dict, SchemaDefinition, or None
        Expected schema definition. If None or unrecognized, generic rules apply.
    dataset_name : str, optional
        Label for the dataset.
    output_path : str or Path, optional
        Target destination for the corrected CSV copy.

    Returns
    -------
    Tuple[pd.DataFrame, CorrectionExecutionResult]
        The corrected DataFrame copy and the execution summary result.
    """
    original_path_str = ""
    if isinstance(dataset_input, (str, Path)):
        p = Path(dataset_input)
        original_path_str = str(p).replace("\\", "/")
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {original_path_str}")
        dname = dataset_name or p.name
        try:
            df_orig = pd.read_csv(p, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to read CSV dataset '{original_path_str}': {e}")
    elif isinstance(dataset_input, pd.DataFrame):
        df_orig = dataset_input
        dname = dataset_name or "in_memory_dataframe"
        original_path_str = "in_memory_dataframe.csv"
    else:
        raise ValueError(f"Unsupported dataset input type: {type(dataset_input)}")

    # 1. IMMUTABILITY GUARANTEE: Deep copy
    df = df_orig.copy(deep=True)

    # 2. Resolve Schema Definition if available
    schema_def: Optional[SchemaDefinition] = None
    if schema:
        try:
            schema_def = load_schema(schema)
        except Exception as e:
            logger.info(f"No specific schema loaded for '{schema}'; using generic correction rules: {e}")
            schema_def = None

    actions: List[CorrectionActionRecord] = []
    unresolved: List[str] = []

    # -------------------------------------------------------------------------
    # STEP 1: DUPLICATE RECORD REMOVAL
    # -------------------------------------------------------------------------
    initial_row_count = len(df)
    
    # Check if primary ID column exists in schema or detected
    id_col = None
    if schema_def:
        for col_name in ["transaction_id", "id", "order_id", "record_id"]:
            if col_name in df.columns:
                id_col = col_name
                break
    elif "transaction_id" in df.columns:
        id_col = "transaction_id"
    elif "id" in df.columns:
        id_col = "id"

    # Identify duplicate rows
    if id_col and id_col in df.columns:
        dup_mask = df[id_col].notna() & df.duplicated(subset=[id_col], keep="first")
        dup_indices = df[dup_mask].index.tolist()
        if dup_indices:
            df = df.drop(index=dup_indices).reset_index(drop=True)
            actions.append(
                CorrectionActionRecord(
                    correction_type="duplicate_removal",
                    column=id_col,
                    row_index=None,
                    original_value=f"{len(dup_indices)} duplicate records",
                    corrected_value="Canonical first occurrences retained",
                    description=f"Removed {len(dup_indices)} duplicate records based on primary identifier '{id_col}'.",
                )
            )
    else:
        exact_dup_mask = df.duplicated(keep="first")
        exact_dup_count = int(exact_dup_mask.sum())
        if exact_dup_count > 0:
            df = df.drop_duplicates(keep="first").reset_index(drop=True)
            actions.append(
                CorrectionActionRecord(
                    correction_type="duplicate_removal",
                    column="<all_columns>",
                    row_index=None,
                    original_value=f"{exact_dup_count} duplicate rows",
                    corrected_value="Unique rows retained",
                    description=f"Removed {exact_dup_count} exact duplicate rows from dataset.",
                )
            )

    # -------------------------------------------------------------------------
    # STEP 2: WHITESPACE & STRING NORMALIZATION
    # -------------------------------------------------------------------------
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).lower() in ("string", "category"):
            trimmed_series = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
            diffs = (df[col] != trimmed_series) & df[col].notna()
            diff_count = int(diffs.sum())
            if diff_count > 0:
                df[col] = trimmed_series
                actions.append(
                    CorrectionActionRecord(
                        correction_type="whitespace_normalization",
                        column=col,
                        row_index=None,
                        original_value=f"{diff_count} unstripped strings",
                        corrected_value="Whitespace trimmed",
                        description=f"Normalized leading/trailing whitespace in {diff_count} entries for column '{col}'.",
                    )
                )

    # -------------------------------------------------------------------------
    # STEP 3: SCHEMA-DRIVEN OR INFERRED COLUMN CORRECTIONS
    # -------------------------------------------------------------------------
    for col in df.columns:
        col_rule: Optional[ColumnSchema] = schema_def.get_column(col) if schema_def else None

        # A. DATE / DATETIME NORMALIZATION
        is_date_col = False
        target_fmt = "%Y-%m-%d"
        if col_rule and col_rule.logical_type == "datetime":
            is_date_col = True
            target_fmt = col_rule.expected_format or "%Y-%m-%d"
        elif "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
            is_date_col = True

        if is_date_col:
            conv_count = 0
            unparseable_count = 0
            new_col_vals = []

            for idx, val in enumerate(df[col]):
                if _is_null_or_empty(val):
                    new_col_vals.append(val)
                    continue

                norm_val, was_conv, is_bad = _parse_and_normalize_date(val, target_format=target_fmt)
                if is_bad:
                    unparseable_count += 1
                    new_col_vals.append(val)  # DO NOT GUESS — keep original
                elif was_conv:
                    conv_count += 1
                    new_col_vals.append(norm_val)
                else:
                    new_col_vals.append(norm_val)

            if conv_count > 0:
                df[col] = new_col_vals
                actions.append(
                    CorrectionActionRecord(
                        correction_type="date_normalization",
                        column=col,
                        row_index=None,
                        original_value=f"{conv_count} non-standard date strings",
                        corrected_value=f"ISO format ({target_fmt})",
                        description=f"Standardized {conv_count} date values in '{col}' to ISO format '{target_fmt}'.",
                    )
                )
            if unparseable_count > 0:
                unresolved.append(
                    f"Column '{col}' contains {unparseable_count} unparseable date values that were preserved without guessing."
                )

        # B. NUMERIC TYPE CLEANING & SUFFIX REMOVAL
        is_num_col = False
        is_int_target = False
        if col_rule:
            if col_rule.logical_type in ("integer", "int"):
                is_num_col = True
                is_int_target = True
            elif col_rule.logical_type in ("float", "decimal", "numeric"):
                is_num_col = True
        elif col.lower() in ("quantity", "qty", "count", "age", "items"):
            is_num_col = True
            is_int_target = True
        elif col.lower() in ("price", "unit_price", "amount", "total_amount", "revenue", "cost", "salary", "rate"):
            is_num_col = True

        if is_num_col:
            type_conv_count = 0
            unresolved_num_count = 0
            new_num_vals = []

            for idx, val in enumerate(df[col]):
                if _is_null_or_empty(val):
                    new_num_vals.append(np.nan)
                    continue

                if isinstance(val, (int, float, np.integer, np.floating)):
                    cleaned_val = float(val)
                    was_conv = False
                else:
                    cleaned_val, was_conv = _clean_numeric_string(val)

                if cleaned_val is not None:
                    if is_int_target:
                        int_candidate = int(round(cleaned_val))
                        new_num_vals.append(int_candidate)
                    else:
                        new_num_vals.append(round(cleaned_val, 2))

                    if was_conv or (isinstance(val, str) and not _is_null_or_empty(val)):
                        type_conv_count += 1
                else:
                    # Unparseable numeric string (e.g. "N/A", "unknown", "text")
                    new_num_vals.append(val)  # DO NOT GUESS
                    unresolved_num_count += 1

            if type_conv_count > 0:
                try:
                    df[col] = pd.to_numeric(new_num_vals, errors="coerce")
                except Exception:
                    df[col] = new_num_vals

                actions.append(
                    CorrectionActionRecord(
                        correction_type="type_conversion",
                        column=col,
                        row_index=None,
                        original_value=f"{type_conv_count} malformed numeric strings",
                        corrected_value="Pure numeric values",
                        description=f"Cleaned and parsed {type_conv_count} string formatted numeric entries in '{col}'.",
                    )
                )

            if unresolved_num_count > 0:
                unresolved.append(
                    f"Column '{col}' has {unresolved_num_count} unparseable non-numeric values preserved as unresolved."
                )

        # C. NUMERIC CONSTRAINT & SIGN INVERSION FIXES
        if col_rule and col_rule.logical_type in ("integer", "float"):
            min_excl = col_rule.min_value_exclusive
            min_incl = col_rule.min_value

            if min_excl is not None or (min_incl is not None and min_incl >= 0):
                if pd.api.types.is_numeric_dtype(df[col]):
                    neg_mask = df[col] < 0
                    neg_count = int(neg_mask.sum())
                    if neg_count > 0:
                        df.loc[neg_mask, col] = df.loc[neg_mask, col].abs()
                        actions.append(
                            CorrectionActionRecord(
                                correction_type="constraint_fix",
                                column=col,
                                row_index=None,
                                original_value=f"{neg_count} negative entries",
                                corrected_value="Absolute positive values",
                                description=f"Corrected {neg_count} inverted sign entries in column '{col}' to satisfy positive constraint.",
                            )
                        )

        # D. CATEGORICAL DOMAIN NORMALIZATION
        if col_rule and col_rule.allowed_values:
            allowed_list = col_rule.allowed_values
            allowed_lookup = {str(v).strip().lower(): v for v in allowed_list}
            for v in allowed_list:
                str_v = str(v).strip().lower()
                allowed_lookup[str_v.replace("-", " ")] = v
                allowed_lookup[str_v.replace(" ", "-")] = v
                allowed_lookup[str_v.replace("_", " ")] = v

            cat_conv_count = 0
            unresolved_cat_count = 0
            new_cat_vals = []

            for idx, val in enumerate(df[col]):
                if _is_null_or_empty(val):
                    new_cat_vals.append(val)
                    continue

                val_clean = str(val).strip()
                val_key = val_clean.lower()

                if val_clean in allowed_list:
                    new_cat_vals.append(val_clean)
                elif val_key in allowed_lookup:
                    mapped_val = allowed_lookup[val_key]
                    new_cat_vals.append(mapped_val)
                    cat_conv_count += 1
                else:
                    new_cat_vals.append(val)
                    unresolved_cat_count += 1

            if cat_conv_count > 0:
                df[col] = new_cat_vals
                actions.append(
                    CorrectionActionRecord(
                        correction_type="categorical_mapping",
                        column=col,
                        row_index=None,
                        original_value=f"{cat_conv_count} case/format mismatched category entries",
                        corrected_value="Conforming domain values",
                        description=f"Mapped {cat_conv_count} category values in '{col}' to canonical allowed domain values.",
                    )
                )

            if unresolved_cat_count > 0:
                unresolved.append(
                    f"Column '{col}' has {unresolved_cat_count} out-of-domain values with no safe mapping rule, preserved as unresolved."
                )

    # -------------------------------------------------------------------------
    # STEP 4: COMPUTED FIELD RECOVERY & RECALCULATION
    # -------------------------------------------------------------------------
    if "total_amount" in df.columns and "quantity" in df.columns and "unit_price" in df.columns:
        if pd.api.types.is_numeric_dtype(df["quantity"]) and pd.api.types.is_numeric_dtype(df["unit_price"]):
            computed_total = (df["quantity"] * df["unit_price"]).round(2)
            missing_total_mask = df["total_amount"].isna() & df["quantity"].notna() & df["unit_price"].notna()
            missing_filled_count = int(missing_total_mask.sum())
            if missing_filled_count > 0:
                df.loc[missing_total_mask, "total_amount"] = computed_total[missing_total_mask]
                actions.append(
                    CorrectionActionRecord(
                        correction_type="computed_field",
                        column="total_amount",
                        row_index=None,
                        original_value=f"{missing_filled_count} missing total_amount values",
                        corrected_value="quantity * unit_price",
                        description=f"Calculated {missing_filled_count} missing 'total_amount' values using deterministic formula quantity * unit_price.",
                    )
                )

    # -------------------------------------------------------------------------
    # STEP 5: REQUIRED IDENTIFIERS INTEGRITY
    # -------------------------------------------------------------------------
    for id_field in ["customer_id", "product_id", "transaction_id", "user_id", "account_id"]:
        if id_field in df.columns:
            missing_id_count = int(df[id_field].isna().sum())
            if missing_id_count > 0:
                unresolved.append(
                    f"Required identifier column '{id_field}' contains {missing_id_count} missing values. Values were NOT invented and are reported as unresolved."
                )

    # -------------------------------------------------------------------------
    # STEP 6: OUTPUT FILE CREATION & PATH RESOLUTION
    # -------------------------------------------------------------------------
    target_out_path = None
    if output_path:
        target_out_path = Path(output_path)
    elif original_path_str and original_path_str != "in_memory_dataframe.csv":
        orig_p = Path(original_path_str)
        stem = orig_p.stem
        if not stem.endswith("_corrected"):
            new_name = f"{stem}_corrected.csv"
        else:
            new_name = f"{stem}.csv"
        target_out_path = orig_p.parent / new_name
    else:
        target_out_path = Path("data/corrected") / f"{dname}_corrected.csv"

    if target_out_path:
        target_out_path.parent.mkdir(parents=True, exist_ok=True)
        if original_path_str and Path(original_path_str).resolve() == target_out_path.resolve():
            target_out_path = target_out_path.parent / f"{target_out_path.stem}_safe_copy.csv"

        df.to_csv(target_out_path, index=False)
        corrected_path_str = str(target_out_path).replace("\\", "/")
        logger.info(f"Successfully generated corrected dataset copy: {corrected_path_str}")
    else:
        corrected_path_str = None

    # -------------------------------------------------------------------------
    # STEP 7: SUMMARY COMPILATION
    # -------------------------------------------------------------------------
    total_actions_count = len(actions)
    unresolved_count = len(unresolved)

    if total_actions_count == 0 and unresolved_count == 0:
        exec_status = "CLEAN_NO_ACTION_REQUIRED"
        is_clean = True
        summary = f"Dataset '{dname}' is completely clean. Zero corrective actions were required."
    elif unresolved_count == 0 and total_actions_count > 0:
        exec_status = "CORRECTED"
        is_clean = False
        summary = (
            f"Successfully applied {total_actions_count} deterministic correction strategies. "
            f"0 issues remain unresolved."
        )
    elif total_actions_count > 0 and unresolved_count > 0:
        exec_status = "PARTIALLY_CORRECTED"
        is_clean = False
        summary = (
            f"Applied {total_actions_count} deterministic corrections. "
            f"{unresolved_count} issues remain unresolved and require governance review."
        )
    else:
        exec_status = "FAILED"
        is_clean = False
        summary = f"Unable to correct dataset '{dname}'. {unresolved_count} unresolved issues remain."

    data_change_summary = _analyze_data_changes(df_orig, df, id_col, unresolved)

    result = CorrectionExecutionResult(
        dataset_name=dname,
        original_dataset_path=original_path_str,
        corrected_dataset_path=corrected_path_str,
        status=exec_status,
        corrections_applied=total_actions_count,
        corrections=[a.to_dict() for a in actions],
        issues_resolved=total_actions_count,
        issues_remaining=unresolved_count,
        unresolved_issues=unresolved,
        is_clean=is_clean,
        summary=summary,
        data_change_summary=data_change_summary,
    )

    return df, result


def _analyze_data_changes(df_orig: pd.DataFrame, df_corr: pd.DataFrame, id_col: Optional[str], unresolved: List[str]) -> Dict[str, Any]:
    """Compare original and corrected dataframes to produce a data change summary."""
    orig_rows = len(df_orig)
    corr_rows = len(df_corr)
    removed_rows = max(0, orig_rows - corr_rows)
    
    changed_cells = 0
    new_missing = 0
    unresolved_values = len(unresolved)
    
    common_cols = [c for c in df_orig.columns if c in df_corr.columns]
    
    if id_col and id_col in df_orig.columns and id_col in df_corr.columns:
        # Align by ID
        orig_aligned = df_orig.set_index(id_col).sort_index()
        corr_aligned = df_corr.set_index(id_col).sort_index()
        common_idx = orig_aligned.index.intersection(corr_aligned.index)
        
        # id_col is now the index, so it is no longer in the columns of orig_aligned
        common_cols_no_id = [c for c in common_cols if c != id_col]
        
        orig_common = orig_aligned.loc[common_idx, common_cols_no_id]
        corr_common = corr_aligned.loc[common_idx, common_cols_no_id]
        
        try:
            # Compare
            diff_mask = (orig_common != corr_common) & ~(orig_common.isna() & corr_common.isna())
            changed_cells = int(diff_mask.sum().sum())
            
            # New missing
            new_missing_mask = orig_common.notna() & corr_common.isna()
            new_missing = int(new_missing_mask.sum().sum())
        except Exception:
            pass
    else:
        # Fallback to positional comparison up to min length
        min_len = min(orig_rows, corr_rows)
        orig_slice = df_orig[common_cols].iloc[:min_len]
        corr_slice = df_corr[common_cols].iloc[:min_len]
        
        try:
            diff_mask = (orig_slice != corr_slice) & ~(orig_slice.isna() & corr_slice.isna())
            changed_cells = int(diff_mask.sum().sum())
            
            new_missing_mask = orig_slice.notna() & corr_slice.isna()
            new_missing = int(new_missing_mask.sum().sum())
        except Exception:
            pass
            
    # Classify summary
    category = "CORRECTED"
    if removed_rows > 0:
        category = "RECORD_REMOVED"
    if unresolved_values > 0:
        category = "PARTIALLY_CORRECTED" if changed_cells > 0 else "UNRESOLVED"
        
    return {
        "original_rows": orig_rows,
        "corrected_rows": corr_rows,
        "removed_duplicate_rows": removed_rows,
        "changed_cell_values": changed_cells,
        "newly_introduced_missing_values": new_missing,
        "unresolved_values": unresolved_values,
        "change_category": category
    }
