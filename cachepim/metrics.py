"""Statistical metric calculation and aggregation for CachePIM.

Computes summary tables, seed-level aggregations, paired comparisons,
and confidence intervals conforming to Section 17 schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def calculate_seed_aggregates(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate per-trace rows to master-seed averages, then compute cross-seed statistics.

    Avoids treating multiple traces from the same seed as fully independent observations.
    Conforms to Section 14.14 guidelines.
    """
    if df.empty:
        return pd.DataFrame()

    if metrics is None:
        metrics = [
            "selected_model_ns",
            "cache_hit_rate",
            "regret_ns",
            "normalized_regret",
            "bad_offload",
        ]

    # Calculate Speedup relative to CPU-cache for each trace
    df = df.copy()
    if "speedup" not in df.columns:
        # Match each row to the CPU-cache baseline on the same trace
        cpu_times = df[df["policy_name"] == "CPU-cache"].set_index(["trace_id"])["selected_model_ns"]
        df["cpu_cache_baseline_ns"] = df["trace_id"].map(cpu_times)
        df["speedup"] = df["cpu_cache_baseline_ns"] / df["selected_model_ns"]
        metrics.append("speedup")

    # Step 1: Mean per (policy_name, workload_type, seed)
    group_cols = ["policy_name", "workload_type", "seed"]
    available_metrics = [m for m in metrics if m in df.columns]

    seed_level = df.groupby(group_cols)[available_metrics].mean().reset_index()

    # Step 2: Compute summary across master seeds
    summary_rows: List[Dict[str, Any]] = []

    for (pol, wl), sub_df in seed_level.groupby(["policy_name", "workload_type"]):
        n_seeds = len(sub_df)
        for m in available_metrics:
            vals = sub_df[m].to_numpy()
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0
            median_val = float(np.median(vals))

            # 95% Confidence Interval using Student's t distribution
            ci_low = mean_val
            ci_high = mean_val
            ci_method = "none"
            if n_seeds > 1 and std_val > 1e-12:
                ci_method = "students_t_95"
                sem = std_val / np.sqrt(n_seeds)
                t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
                ci_low = float(mean_val - t_crit * sem)
                ci_high = float(mean_val + t_crit * sem)

            summary_rows.append({
                "metric_name": m,
                "policy_name": pol,
                "workload_type": wl,
                "configuration_group": "default",
                "number_of_master_seeds": n_seeds,
                "mean": mean_val,
                "sample_std": std_val,
                "median": median_val,
                "confidence_interval_low": ci_low,
                "confidence_interval_high": ci_high,
                "ci_method": ci_method,
            })

    return pd.DataFrame(summary_rows)


def paired_policy_comparison(
    df: pd.DataFrame,
    test_policy: str = "Reuse-aware",
    baseline_policy: str = "Cost-based",
    metric: str = "selected_model_ns",
) -> Dict[str, Any]:
    """Compute paired statistics between two policies on identical traces."""
    pivot = df.pivot_table(
        index=["seed", "trace_id", "workload_type"],
        columns="policy_name",
        values=metric,
    ).dropna()

    if test_policy not in pivot.columns or baseline_policy not in pivot.columns:
        return {"error": f"Policies {test_policy} or {baseline_policy} not present in data"}

    diffs = pivot[baseline_policy] - pivot[test_policy]  # Positive means test_policy is faster/better
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    p_value = 1.0
    if n > 1 and std_diff > 1e-12:
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        p_value = float(p_val)

    return {
        "test_policy": test_policy,
        "baseline_policy": baseline_policy,
        "metric": metric,
        "num_pairs": n,
        "mean_difference": mean_diff,
        "sample_std_difference": std_diff,
        "p_value": p_value,
    }
