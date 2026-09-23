"""
app/tools/anomaly_tool.py
-------------------------
Phase 4 — Deterministic Anomaly Detection Tool.

Accepts a CSV file path or a Pandas DataFrame and returns a fully
structured AnomalyResult containing:

  - Numerical outliers (IQR method, primary)
  - Numerical outliers (Z-score method, secondary / optional)
  - Distribution anomalies for categorical columns
  - Business metric anomalies (revenue, transaction count, avg value)
  - Per-anomaly severity (LOW / MEDIUM / HIGH)
  - Anomaly summary and analysis metadata

All calculations are performed by Pandas/NumPy.
No LLM is involved in this module.

IQR Method (primary):
  Lower fence = Q1 - 1.5 × IQR
  Upper fence = Q3 + 1.5 × IQR
  Values outside the fences are reported as outliers.
  IQR = Q3 - Q1.

Z-Score Method (secondary):
  z = (x - mean) / std
  Applied only when n >= MIN_ZSCORE_ROWS (default 30) and std > 0.
  Default threshold: |z| > 3.0.
  Not used on small datasets to avoid false positives.

Severity rules:
  HIGH   — outlier_pct >= 5% OR metric deviation >= 30%
  MEDIUM — outlier_pct >= 1% OR metric deviation >= 15%
  LOW    — any detected outlier below MEDIUM threshold

Usage:
    from app.tools.anomaly_tool import detect_anomalies

    result = detect_anomalies("data/raw/sales_problematic.csv")
    print(result.to_dict())
"""

from __future__ import annotations

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

IQR_MULTIPLIER       = 1.5       # standard IQR fence factor
ZSCORE_THRESHOLD     = 3.0       # |z| threshold for Z-score method
MIN_ZSCORE_ROWS      = 30        # minimum rows before Z-score is reliable
MAX_EXAMPLE_VALUES   = 10        # max representative outlier values to report
TOP_N_CATEGORIES     = 10        # max categories to include in distribution info

# Severity thresholds
HIGH_OUTLIER_PCT     = 5.0       # >= 5% → HIGH severity
MEDIUM_OUTLIER_PCT   = 1.0       # >= 1% → MEDIUM severity
HIGH_METRIC_DEV_PCT  = 30.0      # >= 30% deviation → HIGH
MEDIUM_METRIC_DEV_PCT= 15.0      # >= 15% deviation → MEDIUM

# Business metric anomaly detection
METRIC_STDDEV_FACTOR = 2.0       # flag if |day - mean| > N * std
MIN_DAYS_FOR_METRIC  = 7         # minimum distinct days for time-series analysis

# Distribution anomaly detection
# Flag a category when its relative frequency deviates significantly from
# a uniform distribution baseline.
DIST_DOMINANCE_THRESHOLD = 0.70  # single category > 70% of values → flag

# ---------------------------------------------------------------------------
# Enums / string constants
# ---------------------------------------------------------------------------

class Severity:
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    NONE   = "NONE"


# ---------------------------------------------------------------------------
# Data classes — structured output
# ---------------------------------------------------------------------------


@dataclass
class NumericalOutlier:
    """Outlier report for one numeric column using one detection method."""
    column:           str
    method:           str           # "IQR" or "Z-score"
    observation_count:int
    lower_fence:      Optional[float]   # None for Z-score method
    upper_fence:      Optional[float]   # None for Z-score method
    threshold:        Optional[float]   # Z-score threshold if Z-score method
    outlier_count:    int
    outlier_pct:      float
    max_outlier_value:Optional[float]
    min_outlier_value:Optional[float]
    example_values:   list[float]       # up to MAX_EXAMPLE_VALUES
    severity:         str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistributionAnomaly:
    """Unexpected concentration or distribution in a categorical column."""
    column:           str
    anomaly_type:     str     # e.g. "dominant_category", "unexpected_category"
    description:      str
    dominant_value:   Optional[str]
    dominant_pct:     Optional[float]
    category_counts:  dict[str, int]   # top categories and their counts
    severity:         str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricAnomaly:
    """Time-series or aggregate business-metric anomaly."""
    metric:           str     # e.g. "daily_revenue"
    period:           str     # e.g. "2024-03-15" or "overall"
    observed_value:   float
    baseline_value:   float   # mean of other periods, or expected value
    pct_deviation:    float   # (observed - baseline) / baseline * 100
    direction:        str     # "above" or "below"
    anomaly_flag:     bool
    severity:         str
    note:             str     # brief human-readable note

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyResult:
    """
    Complete, structured output of the anomaly detection tool.
    All numerical values are produced by Pandas/NumPy.
    """
    dataset_name:          str
    source_path:           str
    row_count:             int
    anomalies_detected:    bool
    total_anomalies:       int

    numerical_outliers:    list[NumericalOutlier]
    distribution_anomalies:list[DistributionAnomaly]
    metric_anomalies:      list[MetricAnomaly]

    severity_summary:      dict[str, int]   # {HIGH: n, MEDIUM: n, LOW: n}
    analysis_metadata:     dict[str, Any]
    anomaly_error:         Optional[str]    # set if loading/analysis failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name":            self.dataset_name,
            "source_path":             self.source_path,
            "row_count":               self.row_count,
            "anomalies_detected":      self.anomalies_detected,
            "total_anomalies":         self.total_anomalies,
            "numerical_outliers":      [o.to_dict() for o in self.numerical_outliers],
            "distribution_anomalies":  [d.to_dict() for d in self.distribution_anomalies],
            "metric_anomalies":        [m.to_dict() for m in self.metric_anomalies],
            "severity_summary":        self.severity_summary,
            "analysis_metadata":       self.analysis_metadata,
            "anomaly_error":           self.anomaly_error,
        }

    def to_compact_dict(self) -> dict[str, Any]:
        """Compact version suitable for LLM context windows."""
        def _compact_outlier(o: NumericalOutlier) -> dict:
            return {
                "column": o.column, "method": o.method,
                "outlier_count": o.outlier_count,
                "outlier_pct": o.outlier_pct,
                "max_value": o.max_outlier_value,
                "severity": o.severity,
            }
        def _compact_dist(d: DistributionAnomaly) -> dict:
            return {
                "column": d.column, "type": d.anomaly_type,
                "description": d.description, "severity": d.severity,
            }
        def _compact_metric(m: MetricAnomaly) -> dict:
            return {
                "metric": m.metric, "period": m.period,
                "observed": m.observed_value, "baseline": m.baseline_value,
                "pct_deviation": m.pct_deviation, "severity": m.severity,
            }
        return {
            "dataset_name":       self.dataset_name,
            "row_count":          self.row_count,
            "anomalies_detected": self.anomalies_detected,
            "total_anomalies":    self.total_anomalies,
            "severity_summary":   self.severity_summary,
            "numerical_outliers": [_compact_outlier(o) for o in self.numerical_outliers],
            "distribution_anomalies": [_compact_dist(d) for d in self.distribution_anomalies],
            "metric_anomalies":   [_compact_metric(m) for m in self.metric_anomalies],
            "analysis_metadata":  self.analysis_metadata,
            "anomaly_error":      self.anomaly_error,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _safe_round(value: Any, decimals: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        return None if math.isnan(v) else round(v, decimals)
    except (TypeError, ValueError):
        return None


def _compute_severity_outlier(outlier_pct: float) -> str:
    if outlier_pct >= HIGH_OUTLIER_PCT:
        return Severity.HIGH
    if outlier_pct >= MEDIUM_OUTLIER_PCT:
        return Severity.MEDIUM
    return Severity.LOW


def _compute_severity_metric(abs_pct: float) -> str:
    if abs_pct >= HIGH_METRIC_DEV_PCT:
        return Severity.HIGH
    if abs_pct >= MEDIUM_METRIC_DEV_PCT:
        return Severity.MEDIUM
    return Severity.LOW


def _count_by_severity(items: list) -> dict[str, int]:
    counts: dict[str, int] = {Severity.HIGH: 0, Severity.MEDIUM: 0, Severity.LOW: 0}
    for item in items:
        sev = getattr(item, "severity", Severity.LOW)
        if sev in counts:
            counts[sev] += 1
    return counts


# ---------------------------------------------------------------------------
# 1. Numerical outlier detection — IQR (primary)
# ---------------------------------------------------------------------------


def _detect_iqr_outliers(
    series: pd.Series,
    col_name: str,
) -> Optional[NumericalOutlier]:
    """
    Apply the IQR fence method to a numeric series.

    Lower fence = Q1 - 1.5 × IQR
    Upper fence = Q3 + 1.5 × IQR

    Returns None when the column has fewer than 4 non-null values
    (insufficient data for meaningful quartile computation).
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n < 4:
        logger.debug("IQR skipped for '%s': only %d valid values.", col_name, n)
        return None

    q1  = float(clean.quantile(0.25))
    q3  = float(clean.quantile(0.75))
    iqr = q3 - q1

    # Degenerate case: zero IQR (all values identical or near-identical)
    if iqr == 0:
        logger.debug("IQR=0 for '%s' — skipping fence detection.", col_name)
        return None

    lo = q1 - IQR_MULTIPLIER * iqr
    hi = q3 + IQR_MULTIPLIER * iqr

    outliers = clean[(clean < lo) | (clean > hi)]
    out_count = len(outliers)
    out_pct   = _safe_round(100 * out_count / n, 4)

    example_vals = sorted(outliers.tolist(), key=abs, reverse=True)[:MAX_EXAMPLE_VALUES]

    return NumericalOutlier(
        column=col_name,
        method="IQR",
        observation_count=n,
        lower_fence=_safe_round(lo, 4),
        upper_fence=_safe_round(hi, 4),
        threshold=None,
        outlier_count=out_count,
        outlier_pct=out_pct,
        max_outlier_value=_safe_round(float(outliers.max())) if out_count else None,
        min_outlier_value=_safe_round(float(outliers.min())) if out_count else None,
        example_values=[_safe_round(v, 4) for v in example_vals],
        severity=_compute_severity_outlier(out_pct),
    )


# ---------------------------------------------------------------------------
# 2. Numerical outlier detection — Z-score (secondary)
# ---------------------------------------------------------------------------


def _detect_zscore_outliers(
    series: pd.Series,
    col_name: str,
    threshold: float = ZSCORE_THRESHOLD,
) -> Optional[NumericalOutlier]:
    """
    Apply Z-score outlier detection.

    Only applied when:
      - n >= MIN_ZSCORE_ROWS  (avoids distorted statistics on tiny datasets)
      - std > 0               (avoids divide-by-zero on constant columns)

    Threshold: default |z| > 3.0 (covers ~99.7% of a normal distribution).
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)

    if n < MIN_ZSCORE_ROWS:
        logger.debug("Z-score skipped for '%s': n=%d < %d.", col_name, n, MIN_ZSCORE_ROWS)
        return None

    mu  = float(clean.mean())
    std = float(clean.std(ddof=1))

    if std == 0:
        logger.debug("Z-score skipped for '%s': std=0.", col_name)
        return None

    z_scores = (clean - mu) / std
    outliers  = clean[z_scores.abs() > threshold]
    out_count = len(outliers)
    out_pct   = _safe_round(100 * out_count / n, 4)

    example_vals = sorted(outliers.tolist(), key=abs, reverse=True)[:MAX_EXAMPLE_VALUES]

    return NumericalOutlier(
        column=col_name,
        method="Z-score",
        observation_count=n,
        lower_fence=None,
        upper_fence=None,
        threshold=threshold,
        outlier_count=out_count,
        outlier_pct=out_pct,
        max_outlier_value=_safe_round(float(outliers.max())) if out_count else None,
        min_outlier_value=_safe_round(float(outliers.min())) if out_count else None,
        example_values=[_safe_round(v, 4) for v in example_vals],
        severity=_compute_severity_outlier(out_pct),
    )


# ---------------------------------------------------------------------------
# 3. Distribution anomalies
# ---------------------------------------------------------------------------


def _detect_distribution_anomalies(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> list[DistributionAnomaly]:
    """
    Check categorical columns for unusual value distributions.

    Flags:
    - Dominant category: one value accounts for >= DIST_DOMINANCE_THRESHOLD
      of all non-null values (e.g. 95% of all transactions are from one region).
    - Unexpected categories: values containing keywords like UNKNOWN, LEGACY,
      TEST, N/A, NULL, INVALID in categorical columns.
    """
    results: list[DistributionAnomaly] = []

    for col in categorical_cols:
        series   = df[col].dropna().astype(str)
        total    = len(series)
        if total == 0:
            continue

        vc = series.value_counts()
        category_counts = {str(k): int(v) for k, v in vc.head(TOP_N_CATEGORIES).items()}

        # Check for single dominant category
        top_val  = str(vc.index[0])
        top_cnt  = int(vc.iloc[0])
        top_pct  = _safe_round(100 * top_cnt / total, 2)

        if top_pct >= DIST_DOMINANCE_THRESHOLD * 100:
            severity = Severity.HIGH if top_pct >= 90 else Severity.MEDIUM
            results.append(DistributionAnomaly(
                column=col,
                anomaly_type="dominant_category",
                description=(
                    f"Category '{top_val}' accounts for {top_pct}% of all "
                    f"non-null values in column '{col}' — unusually concentrated."
                ),
                dominant_value=top_val,
                dominant_pct=top_pct,
                category_counts=category_counts,
                severity=severity,
            ))

        # Check for suspicious/unexpected category values
        suspicious_keywords = {
            "UNKNOWN", "LEGACY", "TEST", "INVALID", "N/A",
            "NULL", "NONE", "ERROR", "UNDEFINED", "MISC",
            "CRYPTOCURRENCY", "UNKNOWN_REGION",
        }
        found_suspicious = [
            v for v in vc.index
            if any(kw in str(v).upper() for kw in suspicious_keywords)
        ]
        if found_suspicious:
            sus_counts = {str(v): int(vc[v]) for v in found_suspicious}
            sus_total  = sum(sus_counts.values())
            sus_pct    = _safe_round(100 * sus_total / total, 2)
            severity   = Severity.MEDIUM if sus_pct >= 5 else Severity.LOW
            results.append(DistributionAnomaly(
                column=col,
                anomaly_type="unexpected_category",
                description=(
                    f"Column '{col}' contains {len(found_suspicious)} unexpected "
                    f"category value(s) ({sus_pct}% of non-null rows): "
                    f"{list(sus_counts.keys())[:5]}"
                ),
                dominant_value=None,
                dominant_pct=sus_pct,
                category_counts=sus_counts,
                severity=severity,
            ))

    return results


# ---------------------------------------------------------------------------
# 4. Business metric anomalies
# ---------------------------------------------------------------------------


def _detect_metric_anomalies(
    df: pd.DataFrame,
    date_col:   str = "transaction_date",
    amount_col: str = "total_amount",
    id_col:     str = "transaction_id",
) -> tuple[list[MetricAnomaly], dict[str, Any]]:
    """
    Analyse aggregate sales metrics and flag significant deviations.

    Returns (metric_anomalies, metadata_dict).

    Strategy:
      1. Calculate daily revenue, daily count, daily avg transaction value.
      2. Baseline = mean of the distribution of all daily values.
      3. Flag days where the value deviates by > METRIC_STDDEV_FACTOR standard
         deviations from the mean.
      4. Also report overall aggregate metric (total revenue, total count).
    """
    anomalies: list[MetricAnomaly] = []
    metadata:  dict[str, Any]      = {}

    # -----------------------------------------------------------------------
    # Guard: verify required columns exist and are usable
    # -----------------------------------------------------------------------
    if amount_col not in df.columns:
        metadata["metric_analysis"] = f"'{amount_col}' column not found — skipped"
        return anomalies, metadata

    amounts = pd.to_numeric(df[amount_col], errors="coerce")
    valid_amounts = amounts.dropna()
    if valid_amounts.empty:
        metadata["metric_analysis"] = "No valid numeric values in amount column"
        return anomalies, metadata

    total_revenue    = _safe_round(float(valid_amounts.sum()), 2)
    avg_txn_value    = _safe_round(float(valid_amounts.mean()), 4)
    txn_count        = int(valid_amounts.count())
    metadata["total_revenue"]     = total_revenue
    metadata["transaction_count"] = txn_count
    metadata["avg_transaction_value"] = avg_txn_value

    # -----------------------------------------------------------------------
    # Time-series analysis (requires a parseable date column)
    # -----------------------------------------------------------------------
    if date_col not in df.columns:
        metadata["time_series_analysis"] = f"'{date_col}' column not found — skipped"
        return anomalies, metadata

    try:
        df_ts = df[[date_col, amount_col]].copy()
        df_ts[date_col]   = pd.to_datetime(df_ts[date_col], errors="coerce")
        df_ts[amount_col] = pd.to_numeric(df_ts[amount_col], errors="coerce")
        df_ts = df_ts.dropna(subset=[date_col, amount_col])

        if df_ts.empty:
            metadata["time_series_analysis"] = "No valid date-amount pairs after cleaning"
            return anomalies, metadata

        # Aggregate to daily level
        daily = df_ts.groupby(df_ts[date_col].dt.date)[amount_col]
        daily_revenue = daily.sum()
        daily_count   = daily.count()
        daily_avg     = daily.mean()

        n_days = len(daily_revenue)
        metadata["date_range_days"]    = n_days
        metadata["date_min"]           = str(df_ts[date_col].min().date())
        metadata["date_max"]           = str(df_ts[date_col].max().date())

        if n_days < MIN_DAYS_FOR_METRIC:
            metadata["time_series_analysis"] = (
                f"Only {n_days} distinct days — minimum {MIN_DAYS_FOR_METRIC} "
                f"required for reliable time-series analysis"
            )
            return anomalies, metadata

        # Detect anomalous days for each metric
        for metric_name, series in [
            ("daily_revenue",                daily_revenue),
            ("daily_transaction_count",      daily_count),
            ("daily_avg_transaction_value",  daily_avg),
        ]:
            mu  = float(series.mean())
            std = float(series.std(ddof=1)) if len(series) > 1 else 0.0

            if std == 0 or mu == 0:
                continue

            threshold_val = METRIC_STDDEV_FACTOR * std
            flagged       = series[abs(series - mu) > threshold_val]

            for period, obs_val in flagged.items():
                obs_val_f  = float(obs_val)
                pct_dev    = _safe_round((obs_val_f - mu) / mu * 100, 2)
                direction  = "above" if obs_val_f > mu else "below"
                abs_dev    = abs(pct_dev) if pct_dev is not None else 0.0
                severity   = _compute_severity_metric(abs_dev)

                anomalies.append(MetricAnomaly(
                    metric=metric_name,
                    period=str(period),
                    observed_value=_safe_round(obs_val_f, 4),
                    baseline_value=_safe_round(mu, 4),
                    pct_deviation=pct_dev,
                    direction=direction,
                    anomaly_flag=True,
                    severity=severity,
                    note=(
                        f"{metric_name} on {period} was {direction} the daily "
                        f"mean by {abs_dev:.1f}% "
                        f"(>{METRIC_STDDEV_FACTOR}σ deviation)"
                    ),
                ))

        metadata["time_series_analysis"] = "completed"

    except Exception as exc:
        logger.warning("Time-series metric analysis failed: %s", exc)
        metadata["time_series_analysis"] = f"error: {exc}"

    return anomalies, metadata


# ---------------------------------------------------------------------------
# Helper: identify numeric columns robustly
# ---------------------------------------------------------------------------


def _find_numeric_columns(df: pd.DataFrame, threshold: float = 0.80) -> list[str]:
    """
    Return columns where >= threshold fraction of non-null values are numeric.
    Works regardless of how the CSV was loaded (all-string or typed).
    """
    numeric_cols = []
    for col in df.columns:
        series       = df[col].dropna()
        if series.empty:
            continue
        numeric_vals = pd.to_numeric(series, errors="coerce")
        success_rate = numeric_vals.notna().mean()
        if success_rate >= threshold:
            numeric_cols.append(col)
    return numeric_cols


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_anomalies(
    source:       Union[str, Path, pd.DataFrame],
    dataset_name: Optional[str]  = None,
    date_col:     str            = "transaction_date",
    amount_col:   str            = "total_amount",
    id_col:       str            = "transaction_id",
    run_zscore:   bool           = True,
    categorical_cols: Optional[list[str]] = None,
) -> AnomalyResult:
    """
    Run all anomaly detection algorithms on the given dataset.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        Path to a CSV file or an already-loaded DataFrame.
    dataset_name : str, optional
        Human-readable label. Defaults to the filename stem.
    date_col : str
        Column name for transaction dates (time-series metric analysis).
    amount_col : str
        Column name for the primary monetary amount.
    id_col : str
        Column name for unique transaction/row identifier.
    run_zscore : bool
        If True, run Z-score analysis in addition to IQR (on sufficiently
        large numeric columns). Default True.
    categorical_cols : list[str], optional
        Override which columns are treated as categorical for distribution
        analysis. If None, columns not classified as numeric are used.

    Returns
    -------
    AnomalyResult
        Fully structured anomaly output.  On loading errors the result has
        ``anomaly_error`` set and empty detection lists.
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
        source_path = str(source)
        path_obj    = Path(source_path)
        if dataset_name is None:
            dataset_name = path_obj.stem

        if not path_obj.exists():
            err = f"File not found: {source_path}"
            logger.error(err)
            return _empty_error_result(dataset_name, source_path, err)

        try:
            df = pd.read_csv(path_obj, dtype=str)
            logger.info("Loaded %d rows × %d cols from '%s'",
                        len(df), len(df.columns), path_obj.name)
        except Exception as exc:
            err = f"CSV load error: {exc}"
            logger.error(err)
            return _empty_error_result(dataset_name, source_path, err)

    if df.empty:
        err = "Dataset is empty (0 rows)."
        logger.warning(err)
        return _empty_error_result(dataset_name, source_path, err)

    row_count = len(df)

    # -----------------------------------------------------------------------
    # 2. Identify numeric and categorical columns
    # -----------------------------------------------------------------------
    numeric_cols = _find_numeric_columns(df)
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # -----------------------------------------------------------------------
    # 3. Numerical outliers — IQR
    # -----------------------------------------------------------------------
    iqr_outliers: list[NumericalOutlier] = []
    for col in numeric_cols:
        result = _detect_iqr_outliers(df[col], col)
        if result is not None and result.outlier_count > 0:
            iqr_outliers.append(result)

    # -----------------------------------------------------------------------
    # 4. Numerical outliers — Z-score (secondary)
    # -----------------------------------------------------------------------
    zscore_outliers: list[NumericalOutlier] = []
    if run_zscore:
        for col in numeric_cols:
            result = _detect_zscore_outliers(df[col], col)
            if result is not None and result.outlier_count > 0:
                zscore_outliers.append(result)

    all_numerical_outliers = iqr_outliers + zscore_outliers

    # -----------------------------------------------------------------------
    # 5. Distribution anomalies
    # -----------------------------------------------------------------------
    dist_anomalies = _detect_distribution_anomalies(df, categorical_cols)

    # -----------------------------------------------------------------------
    # 6. Business metric anomalies
    # -----------------------------------------------------------------------
    metric_anomalies, metric_meta = _detect_metric_anomalies(
        df, date_col=date_col, amount_col=amount_col, id_col=id_col
    )

    # -----------------------------------------------------------------------
    # 7. Aggregate
    # -----------------------------------------------------------------------
    all_items = all_numerical_outliers + dist_anomalies + metric_anomalies
    sev_summary = _count_by_severity(all_items)
    total_anomalies = len(all_items)
    anomalies_detected = total_anomalies > 0

    metadata: dict[str, Any] = {
        "numeric_columns_analyzed":      numeric_cols,
        "categorical_columns_analyzed":  categorical_cols,
        "iqr_outlier_columns":           [o.column for o in iqr_outliers],
        "zscore_run":                    run_zscore,
        "zscore_outlier_columns":        [o.column for o in zscore_outliers],
        "iqr_multiplier":                IQR_MULTIPLIER,
        "zscore_threshold":              ZSCORE_THRESHOLD,
        "min_zscore_rows":               MIN_ZSCORE_ROWS,
        **metric_meta,
    }

    return AnomalyResult(
        dataset_name=dataset_name,
        source_path=source_path,
        row_count=row_count,
        anomalies_detected=anomalies_detected,
        total_anomalies=total_anomalies,
        numerical_outliers=all_numerical_outliers,
        distribution_anomalies=dist_anomalies,
        metric_anomalies=metric_anomalies,
        severity_summary=sev_summary,
        analysis_metadata=metadata,
        anomaly_error=None,
    )


# ---------------------------------------------------------------------------
# Error-result factory
# ---------------------------------------------------------------------------


def _empty_error_result(
    dataset_name: str,
    source_path:  str,
    error_message:str,
) -> AnomalyResult:
    return AnomalyResult(
        dataset_name=dataset_name,
        source_path=source_path,
        row_count=0,
        anomalies_detected=False,
        total_anomalies=0,
        numerical_outliers=[],
        distribution_anomalies=[],
        metric_anomalies=[],
        severity_summary={Severity.HIGH: 0, Severity.MEDIUM: 0, Severity.LOW: 0},
        analysis_metadata={},
        anomaly_error=error_message,
    )
