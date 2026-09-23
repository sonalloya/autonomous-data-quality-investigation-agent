"""
app/tools/profiling_tool.py
---------------------------
Phase 3 — Deterministic Data Profiling Tool.

Accepts a CSV file path or a Pandas DataFrame and returns a fully
structured ProfilingResult containing:

  - Basic dataset shape (rows, columns)
  - Per-column details (dtype, nulls, uniques, stats)
  - Missing-value summary
  - Duplicate analysis (full rows + optional transaction_id duplicates)
  - Numeric statistical summary
  - Categorical summary
  - High-level quality indicators

All calculations are performed by Pandas/NumPy.
No LLM is involved in this module.

Usage:
    from app.tools.profiling_tool import profile_dataset

    result = profile_dataset("data/raw/sales_clean.csv")
    print(result.to_dict())
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_N_CATEGORIES   = 5    # number of top values shown for categoricals
MAX_SAMPLE_UNIQUES = 20   # if unique count ≤ this, list all values; else truncate

# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------


@dataclass
class ColumnDetail:
    """Profiling detail for a single column."""
    column_name:        str
    dtype:              str
    null_count:         int
    null_percentage:    float   # 0.0 – 100.0
    unique_count:       int
    uniqueness_pct:     float   # 0.0 – 100.0

    # Numeric-only fields (None for categoricals)
    min:    Optional[float] = None
    max:    Optional[float] = None
    mean:   Optional[float] = None
    median: Optional[float] = None
    std:    Optional[float] = None
    q25:    Optional[float] = None
    q75:    Optional[float] = None

    # Categorical-only fields (None for numerics)
    top_values:        Optional[list[dict[str, Any]]] = None  # [{value, count, pct}]
    sample_values:     Optional[list[Any]]             = None  # up to 5 distinct sample vals

    is_numeric:   bool = False
    is_datetime:  bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MissingValueSummary:
    """Aggregated missing-value statistics."""
    total_missing_cells:   int
    total_cells:           int
    overall_missing_pct:   float
    columns_with_missing:  list[dict[str, Any]]   # [{column, count, pct}]
    columns_complete:      int
    columns_with_nulls:    int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateSummary:
    """Duplicate analysis results."""
    duplicate_rows:            int
    duplicate_row_percentage:  float
    duplicate_transaction_ids: Optional[int]   # None if no transaction_id column

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityIndicators:
    """
    High-level, explainable quality metrics.

    Formulas (all in 0–100 range):
      completeness_pct  = 100 * non-null cells / total cells
      uniqueness_pct    = 100 * unique rows / total rows
      duplicate_pct     = 100 * duplicate rows / total rows
    """
    completeness_pct: float
    uniqueness_pct:   float
    duplicate_pct:    float
    total_issues_detected: int   # simple count of notable problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProfilingResult:
    """
    Complete, structured output of the profiling tool.

    All numerical values are produced by Pandas; the LLM only receives
    this object for interpretation.
    """
    dataset_name:      str
    source_path:       str
    row_count:         int
    column_count:      int
    columns:           list[str]

    column_details:         list[ColumnDetail]
    missing_value_summary:  MissingValueSummary
    duplicate_summary:      DuplicateSummary
    quality_indicators:     QualityIndicators

    # Convenience sub-summaries (derived from column_details)
    numeric_columns:      list[str]
    categorical_columns:  list[str]
    datetime_columns:     list[str]

    profiling_error: Optional[str] = None   # set if loading/profiling failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name":     self.dataset_name,
            "source_path":      self.source_path,
            "row_count":        self.row_count,
            "column_count":     self.column_count,
            "columns":          self.columns,
            "column_details":   [c.to_dict() for c in self.column_details],
            "missing_value_summary": self.missing_value_summary.to_dict(),
            "duplicate_summary":     self.duplicate_summary.to_dict(),
            "quality_indicators":    self.quality_indicators.to_dict(),
            "numeric_columns":       self.numeric_columns,
            "categorical_columns":   self.categorical_columns,
            "datetime_columns":      self.datetime_columns,
            "profiling_error":       self.profiling_error,
        }

    def to_compact_dict(self) -> dict[str, Any]:
        """
        Returns a compact version suitable for LLM context windows.
        Omits raw per-column top_values lists; keeps key stats only.
        """
        col_summaries = []
        for cd in self.column_details:
            summary: dict[str, Any] = {
                "name":         cd.column_name,
                "dtype":        cd.dtype,
                "null_pct":     cd.null_percentage,
                "unique_count": cd.unique_count,
            }
            if cd.is_numeric:
                summary.update({
                    "min": cd.min, "max": cd.max,
                    "mean": cd.mean, "std": cd.std,
                })
            elif cd.top_values:
                summary["top_values"] = [
                    f"{tv['value']} ({tv['count']})" for tv in cd.top_values[:3]
                ]
            col_summaries.append(summary)

        return {
            "dataset_name":    self.dataset_name,
            "row_count":       self.row_count,
            "column_count":    self.column_count,
            "missing_summary": self.missing_value_summary.to_dict(),
            "duplicate_summary": self.duplicate_summary.to_dict(),
            "quality_indicators": self.quality_indicators.to_dict(),
            "column_summaries": col_summaries,
            "profiling_error": self.profiling_error,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_round(value: Any, decimals: int = 2) -> Optional[float]:
    """Round to `decimals` places; return None if value is NaN/None."""
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def _profile_numeric_column(series: pd.Series) -> dict[str, Optional[float]]:
    """Compute descriptive statistics for a numeric column."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {k: None for k in ("min", "max", "mean", "median", "std", "q25", "q75")}
    desc = clean.describe(percentiles=[0.25, 0.75])
    return {
        "min":    _safe_round(desc.get("min")),
        "max":    _safe_round(desc.get("max")),
        "mean":   _safe_round(desc.get("mean")),
        "median": _safe_round(clean.median()),
        "std":    _safe_round(desc.get("std")),
        "q25":    _safe_round(desc.get("25%")),
        "q75":    _safe_round(desc.get("75%")),
    }


def _profile_categorical_column(
    series: pd.Series,
    top_n: int = TOP_N_CATEGORIES,
) -> dict[str, Any]:
    """Compute value-frequency info for a categorical column."""
    non_null = series.dropna()
    total    = len(non_null)
    vc       = non_null.value_counts(dropna=True).head(top_n)

    top_values = [
        {
            "value": str(val),
            "count": int(cnt),
            "pct":   _safe_round(100 * cnt / total) if total else 0.0,
        }
        for val, cnt in vc.items()
    ]

    sample_count = min(MAX_SAMPLE_UNIQUES, series.nunique())
    sample_vals  = [str(v) for v in non_null.unique()[:sample_count]]

    return {"top_values": top_values, "sample_values": sample_vals}


def _is_datetime_column(series: pd.Series) -> bool:
    """Heuristic: try parsing a sample of non-null values as dates."""
    sample = series.dropna().head(20).astype(str)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        success_rate = parsed.notna().mean()
        return bool(success_rate >= 0.8)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-column profiler
# ---------------------------------------------------------------------------


def _profile_column(col_name: str, series: pd.Series, row_count: int) -> ColumnDetail:
    """Build a ColumnDetail for a single column."""
    null_count  = int(series.isna().sum())
    null_pct    = _safe_round(100 * null_count / row_count) if row_count else 0.0
    unique_cnt  = int(series.nunique(dropna=True))
    unique_pct  = _safe_round(100 * unique_cnt / row_count) if row_count else 0.0
    dtype_str   = str(series.dtype)

    # Determine column kind
    numeric_series = pd.to_numeric(series, errors="coerce")
    non_null_count = series.notna().sum()
    numeric_success_rate = (
        numeric_series.notna().sum() / non_null_count
        if non_null_count > 0 else 0
    )

    is_numeric  = numeric_success_rate >= 0.80
    is_datetime = (not is_numeric) and _is_datetime_column(series)

    cd = ColumnDetail(
        column_name=col_name,
        dtype=dtype_str,
        null_count=null_count,
        null_percentage=null_pct,
        unique_count=unique_cnt,
        uniqueness_pct=unique_pct,
        is_numeric=is_numeric,
        is_datetime=is_datetime,
    )

    if is_numeric:
        stats = _profile_numeric_column(numeric_series)
        cd.min    = stats["min"]
        cd.max    = stats["max"]
        cd.mean   = stats["mean"]
        cd.median = stats["median"]
        cd.std    = stats["std"]
        cd.q25    = stats["q25"]
        cd.q75    = stats["q75"]
    else:
        cat_info = _profile_categorical_column(series)
        cd.top_values    = cat_info["top_values"]
        cd.sample_values = cat_info["sample_values"]

    return cd


# ---------------------------------------------------------------------------
# Public profiling function
# ---------------------------------------------------------------------------


def profile_dataset(
    source: Union[str, Path, pd.DataFrame],
    dataset_name: Optional[str] = None,
    id_column: str = "transaction_id",
) -> ProfilingResult:
    """
    Profile a CSV file or existing DataFrame.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        Path to a CSV file, or an already-loaded DataFrame.
    dataset_name : str, optional
        Human-readable name for the dataset.  Defaults to the filename stem.
    id_column : str
        Name of the primary-key/ID column to check for duplicates.
        Default is 'transaction_id'.  Skipped if the column does not exist.

    Returns
    -------
    ProfilingResult
        Fully structured profiling output.  On loading errors the result
        will have ``profiling_error`` set and minimal zero-value stats.
    """
    source_path = ""

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    if isinstance(source, pd.DataFrame):
        df          = source.copy()
        source_path = "<DataFrame>"
        if dataset_name is None:
            dataset_name = "dataframe"
    else:
        source_path  = str(source)
        path_obj     = Path(source_path)
        if dataset_name is None:
            dataset_name = path_obj.stem

        if not path_obj.exists():
            err = f"File not found: {source_path}"
            logger.error(err)
            return _empty_error_result(dataset_name, source_path, err)

        try:
            # Load all columns as strings initially to avoid silent coercion
            df = pd.read_csv(path_obj, dtype=str)
            logger.info("Loaded %d rows x %d cols from '%s'",
                        len(df), len(df.columns), path_obj.name)
        except Exception as exc:
            err = f"CSV load error: {exc}"
            logger.error(err)
            return _empty_error_result(dataset_name, source_path, err)

    if df.empty:
        err = "Dataset is empty (0 rows)."
        logger.warning(err)
        return _empty_error_result(dataset_name, source_path, err)

    # -----------------------------------------------------------------------
    # 2. Basic shape
    # -----------------------------------------------------------------------
    row_count    = len(df)
    column_count = len(df.columns)
    columns      = list(df.columns)

    # -----------------------------------------------------------------------
    # 3. Per-column profiling
    # -----------------------------------------------------------------------
    column_details: list[ColumnDetail] = []
    numeric_cols:   list[str] = []
    categorical_cols: list[str] = []
    datetime_cols:  list[str] = []

    for col in columns:
        cd = _profile_column(col, df[col], row_count)
        column_details.append(cd)
        if cd.is_numeric:
            numeric_cols.append(col)
        elif cd.is_datetime:
            datetime_cols.append(col)
        else:
            categorical_cols.append(col)

    # -----------------------------------------------------------------------
    # 4. Missing value summary
    # -----------------------------------------------------------------------
    total_cells         = row_count * column_count
    null_per_col        = df.isnull().sum()
    total_missing_cells = int(null_per_col.sum())
    overall_missing_pct = _safe_round(100 * total_missing_cells / total_cells) if total_cells else 0.0

    cols_with_missing = [
        {
            "column":  col,
            "count":   int(cnt),
            "pct":     _safe_round(100 * cnt / row_count),
        }
        for col, cnt in null_per_col.items() if cnt > 0
    ]
    cols_with_missing.sort(key=lambda x: x["count"], reverse=True)

    missing_summary = MissingValueSummary(
        total_missing_cells=total_missing_cells,
        total_cells=total_cells,
        overall_missing_pct=overall_missing_pct,
        columns_with_missing=cols_with_missing,
        columns_complete=int((null_per_col == 0).sum()),
        columns_with_nulls=int((null_per_col > 0).sum()),
    )

    # -----------------------------------------------------------------------
    # 5. Duplicate analysis
    # -----------------------------------------------------------------------
    duplicate_rows     = int(df.duplicated().sum())
    dup_row_pct        = _safe_round(100 * duplicate_rows / row_count)
    dup_txn_ids: Optional[int] = None

    if id_column in df.columns:
        dup_txn_ids = int(df[id_column].duplicated(keep="first").sum())

    duplicate_summary = DuplicateSummary(
        duplicate_rows=duplicate_rows,
        duplicate_row_percentage=dup_row_pct,
        duplicate_transaction_ids=dup_txn_ids,
    )

    # -----------------------------------------------------------------------
    # 6. Quality indicators
    # -----------------------------------------------------------------------
    non_null_cells     = total_cells - total_missing_cells
    completeness_pct   = _safe_round(100 * non_null_cells / total_cells) if total_cells else 0.0
    unique_rows        = row_count - duplicate_rows
    uniqueness_pct     = _safe_round(100 * unique_rows / row_count) if row_count else 0.0
    duplicate_pct      = _safe_round(100 * duplicate_rows / row_count) if row_count else 0.0

    # Count 'notable issues' for the summary counter
    issues = 0
    if total_missing_cells > 0:
        issues += 1
    if duplicate_rows > 0:
        issues += 1
    if dup_txn_ids is not None and dup_txn_ids > 0:
        issues += 1

    quality_indicators = QualityIndicators(
        completeness_pct=completeness_pct,
        uniqueness_pct=uniqueness_pct,
        duplicate_pct=duplicate_pct,
        total_issues_detected=issues,
    )

    return ProfilingResult(
        dataset_name=dataset_name,
        source_path=source_path,
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        column_details=column_details,
        missing_value_summary=missing_summary,
        duplicate_summary=duplicate_summary,
        quality_indicators=quality_indicators,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
    )


# ---------------------------------------------------------------------------
# Error-result factory
# ---------------------------------------------------------------------------


def _empty_error_result(
    dataset_name: str,
    source_path: str,
    error_message: str,
) -> ProfilingResult:
    """Return a ProfilingResult that signals a loading/profiling failure."""
    return ProfilingResult(
        dataset_name=dataset_name,
        source_path=source_path,
        row_count=0,
        column_count=0,
        columns=[],
        column_details=[],
        missing_value_summary=MissingValueSummary(
            total_missing_cells=0,
            total_cells=0,
            overall_missing_pct=0.0,
            columns_with_missing=[],
            columns_complete=0,
            columns_with_nulls=0,
        ),
        duplicate_summary=DuplicateSummary(
            duplicate_rows=0,
            duplicate_row_percentage=0.0,
            duplicate_transaction_ids=None,
        ),
        quality_indicators=QualityIndicators(
            completeness_pct=0.0,
            uniqueness_pct=0.0,
            duplicate_pct=0.0,
            total_issues_detected=0,
        ),
        numeric_columns=[],
        categorical_columns=[],
        datetime_columns=[],
        profiling_error=error_message,
    )
