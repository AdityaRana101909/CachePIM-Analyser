"""Statistical analysis, hypothesis testing, and reporting utilities for CachePIM.

Analyzes experimental outcomes, computes effect sizes, bad-offload rates,
and paired statistical significance tests across policies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def analyze_baseline_results(raw_csv_path: str = "results/raw/baseline_comparison_raw.csv") -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on raw baseline comparison runs."""
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw results file not found at {raw_csv_path}")

    df = pd.read_csv(path)

    # 1. Overall Policy Means
    summary = df.groupby("policy_name").agg({
        "selected_model_ns": ["mean", "std"],
        "regret_ns": ["mean", "std"],
        "normalized_regret": ["mean", "std"],
        "bad_offload": ["mean", "sum"],
    })

    # Flatten column MultiIndex
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary_dict = summary.to_dict(orient="index")

    # 2. Paired t-test between Reuse-aware and Cost-based
    pivot_lat = df.pivot_table(
        index=["master_seed", "trace_id", "workload_type"],
        columns="policy_name",
        values="selected_model_ns",
    ).dropna()

    paired_test = {}
    if "Reuse-aware" in pivot_lat.columns and "Cost-based" in pivot_lat.columns:
        diff = pivot_lat["Cost-based"] - pivot_lat["Reuse-aware"]
        t_stat, p_val = stats.ttest_1samp(diff, 0.0)
        paired_test = {
            "mean_latency_diff_ns": float(diff.mean()),
            "std_latency_diff_ns": float(diff.std(ddof=1)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "reuse_aware_faster_pct": float((diff > 0).mean() * 100),
        }

    # 3. Bad Offload Rate Comparison
    bad_offload_rates = df.groupby("policy_name")["bad_offload"].mean().to_dict()

    return {
        "num_runs": len(df),
        "policy_summary": summary_dict,
        "paired_test_reuse_vs_cost": paired_test,
        "bad_offload_rates": bad_offload_rates,
    }


def format_paper_table(summary_csv_path: str = "results/summary/baseline_comparison_summary.csv") -> pd.DataFrame:
    """Format paper-ready LaTeX / Markdown table of seed-aggregated metrics."""
    path = Path(summary_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary results file not found at {summary_csv_path}")

    df = pd.read_csv(path)
    # Pivot for clean display
    display_df = df.pivot_table(
        index=["workload_type", "policy_name"],
        columns="metric_name",
        values="mean",
    )
    return display_df
