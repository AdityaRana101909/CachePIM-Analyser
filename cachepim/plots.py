"""Publication-quality figure generation for CachePIM.

Generates all 8 required figures directly from actual experimental result CSV files:
1. Cache hit/miss rate vs working-set size
2. CPU vs PIM execution latency vs working-set size (crossover characterization)
3. Performance comparison across all execution policies
4. Cache capacity vs transfer bandwidth crossover heatmap
5. Policy speedup relative to CPU-cache across workload types
6. Decision regret and bad-offload rate
7. Reuse-aware ablation results across lambda values
8. Multi-seed variability analysis with error bars / distributions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_paper_style() -> None:
    """Set publication-grade formatting conventions for matplotlib figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.35,
    })


def plot_cache_hit_miss_vs_workingset(
    crossover_csv: str = "results/raw/crossover_sweep_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 1: Cache hit rate and miss rate vs working-set size."""
    set_paper_style()
    df = pd.read_csv(crossover_csv)
    sub = df[(df["policy_name"] == "CPU-cache") & (df["cache_capacity_kb"] == 32) & (df["workload_type"] == "REUSE")]
    grouped = sub.groupby("working_set_kb").agg({"cache_hit_rate": "mean", "cache_miss_rate": "mean"}).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(grouped["working_set_kb"], grouped["cache_hit_rate"] * 100, "o-", color="#1f77b4", label="Cache Hit Rate (%)")
    ax.plot(grouped["working_set_kb"], grouped["cache_miss_rate"] * 100, "s--", color="#d62728", label="Cache Miss Rate (%)")

    ax.axvline(32, color="gray", linestyle=":", linewidth=1.5, label="LLC Capacity (32 KiB)")
    ax.set_xlabel("Working-Set Size (KiB)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Figure 1: Cache Hit and Miss Rates vs. Working-Set Size")
    ax.set_ylim(-5, 105)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    out_p = Path(output_dir) / "fig1_cache_hit_miss_vs_workingset.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def plot_cpu_pim_crossover(
    crossover_csv: str = "results/raw/crossover_sweep_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 2: CPU vs PIM execution latency vs working-set size."""
    set_paper_style()
    df = pd.read_csv(crossover_csv)
    sub = df[(df["cache_capacity_kb"] == 32) & (df["workload_type"] == "STREAM")]

    cpu_sub = sub[sub["policy_name"] == "CPU-cache"].groupby("working_set_kb")["selected_model_ns"].mean().reset_index()
    pim_sub = sub[sub["policy_name"] == "PIM-always"].groupby("working_set_kb")["selected_model_ns"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(cpu_sub["working_set_kb"], cpu_sub["selected_model_ns"] / 1000.0, "o-", color="#1f77b4", label="CPU (Modeled LLC)")
    ax.plot(pim_sub["working_set_kb"], pim_sub["selected_model_ns"] / 1000.0, "^-", color="#2ca02c", label="PIM (Modeled 16-Bank)")

    ax.axvline(32, color="gray", linestyle=":", linewidth=1.5, label="LLC Capacity (32 KiB)")
    ax.set_xlabel("Working-Set Size (KiB)")
    ax.set_ylabel("Execution Time (μs)")
    ax.set_title("Figure 2: CPU vs. PIM Latency and Crossover Boundary")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    out_p = Path(output_dir) / "fig2_cpu_pim_time_vs_workingset.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def plot_policy_performance_comparison(
    baseline_csv: str = "results/raw/baseline_comparison_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 3: Performance comparison across all 6 execution policies."""
    set_paper_style()
    df = pd.read_csv(baseline_csv)

    order = ["CPU-cache", "CPU-no-cache", "PIM-always", "Cost-based", "Reuse-aware", "Oracle"]
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    grouped = df.groupby("policy_name")["selected_model_ns"].mean().reindex(order) / 1000.0

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(grouped.index, grouped.values, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_ylabel("Mean Execution Latency (μs)")
    ax.set_title("Figure 3: Overall Execution Latency Across Offloading Policies")
    ax.set_xticks(range(len(grouped.index)))
    ax.set_xticklabels(grouped.index, rotation=20, ha="right")

    for b in bars:
        height = b.get_height()
        ax.annotate(f"{height:.1f}us",
                    xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_p = Path(output_dir) / "fig3_policy_performance_comparison.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def plot_crossover_heatmap(
    crossover_csv: str = "results/raw/crossover_sweep_raw.csv",
    transfer_csv: str = "results/raw/transfer_sensitivity_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 4: Cache Capacity vs Transfer Bandwidth Crossover Winner Heatmap."""
    set_paper_style()
    # Compute relative speedup grid of PIM vs CPU across Cache Capacity and Transfer Bandwidth
    caps = [16, 32, 64, 128]
    bws = [4.0, 8.0, 16.0, 32.0, 64.0]
    grid = np.zeros((len(bws), len(caps)))

    # Compute modeled PIM speedup for a representative 64 KiB STREAM workload
    from cachepim.config import CachePIMConfig
    from cachepim.simulator import Simulator
    from cachepim.workloads import WorkloadGenerator

    gen = WorkloadGenerator(64)
    trace, desc = gen.generate("STREAM", seed=42, working_set_bytes=64 * 1024, access_count=2048)

    for i, bw in enumerate(bws):
        for j, cap in enumerate(caps):
            cfg = CachePIMConfig()
            cfg.cache.capacity_bytes = cap * 1024
            cfg.transfer.host_to_pim_bandwidth_bytes_per_ns = bw
            cfg.transfer.pim_to_host_bandwidth_bytes_per_ns = bw
            sim = Simulator(cfg)
            res = sim.run_all_policies_on_trace(trace, desc)
            res_dict = {r.policy_name: r.selected_model_ns for r in res}
            # Speedup of PIM over CPU-cache
            speedup = res_dict["CPU-cache"] / res_dict["PIM-always"]
            grid[i, j] = speedup

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(grid, cmap="RdYlGn", origin="lower", aspect="auto")

    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([f"{c} KiB" for c in caps])
    ax.set_yticks(range(len(bws)))
    ax.set_yticklabels([f"{int(b)} GB/s" for b in bws])

    ax.set_xlabel("CPU Cache Capacity")
    ax.set_ylabel("Host-to-PIM Bandwidth")
    ax.set_title("Figure 4: PIM vs. CPU Speedup Heatmap (64 KiB Stream)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Speedup (PIM / CPU-cache)")

    for i in range(len(bws)):
        for j in range(len(caps)):
            ax.text(j, i, f"{grid[i, j]:.2f}x", ha="center", va="center",
                    color="black" if 0.8 < grid[i, j] < 2.5 else "white", fontweight="bold")

    fig.tight_layout()
    out_p = Path(output_dir) / "fig4_crossover_heatmap.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def plot_policy_speedup_by_workload(
    baseline_csv: str = "results/raw/baseline_comparison_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 5: Policy Speedup relative to CPU-cache across workload types."""
    set_paper_style()
    df = pd.read_csv(baseline_csv)

    # Compute speedup relative to CPU-cache per trace
    cpu_baselines = df[df["policy_name"] == "CPU-cache"].set_index(["trace_id"])["selected_model_ns"]
    df = df.copy()
    df["cpu_base"] = df["trace_id"].map(cpu_baselines)
    df["speedup"] = df["cpu_base"] / df["selected_model_ns"]

    policies = ["PIM-always", "Cost-based", "Reuse-aware", "Oracle"]
    workloads = ["STREAM", "REUSE", "RANDOM", "MIXED_PHASE"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(workloads))
    width = 0.2

    colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    for idx, (pol, color) in enumerate(zip(policies, colors)):
        sub = df[df["policy_name"] == pol]
        means = sub.groupby("workload_type")["speedup"].mean().reindex(workloads).values
        ax.bar(x + (idx - 1.5) * width, means, width, label=pol, color=color, edgecolor="black", linewidth=0.7)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="CPU-cache Baseline (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylabel("Speedup vs. CPU-cache (Higher is Better)")
    ax.set_title("Figure 5: Policy Speedup Across Workload Access Patterns")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out_p = Path(output_dir) / "fig5_policy_speedup_by_workload.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def plot_regret_and_bad_offload(
    baseline_csv: str = "results/raw/baseline_comparison_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 6: Decision Regret and Bad-Offload Rate across policies."""
    set_paper_style()
    df = pd.read_csv(baseline_csv)

    eval_policies = ["CPU-cache", "PIM-always", "Cost-based", "Reuse-aware"]
    sub = df[df["policy_name"].isin(eval_policies)]

    regret_mean = sub.groupby("policy_name")["regret_ns"].mean().reindex(eval_policies) / 1000.0
    bad_offload_rate = sub.groupby("policy_name")["bad_offload"].mean().reindex(eval_policies) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Subplot 1: Regret
    bars1 = ax1.bar(eval_policies, regret_mean.values, color="#d62728", edgecolor="black", width=0.55)
    ax1.set_ylabel("Mean Decision Regret (us)")
    ax1.set_title("Decision Regret relative to Oracle")
    ax1.set_xticks(range(len(eval_policies)))
    ax1.set_xticklabels(eval_policies, rotation=20, ha="right")
    for b in bars1:
        ax1.annotate(f"{b.get_height():.2f}us",
                     xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    # Subplot 2: Bad-Offload Rate
    bars2 = ax2.bar(eval_policies, bad_offload_rate.values, color="#e377c2", edgecolor="black", width=0.55)
    ax2.set_ylabel("Bad-Offload Rate (%)")
    ax2.set_title("Bad-Offload Decisions")
    ax2.set_xticks(range(len(eval_policies)))
    ax2.set_xticklabels(eval_policies, rotation=20, ha="right")
    for b in bars2:
        ax2.annotate(f"{b.get_height():.1f}%",
                     xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    fig.suptitle("Figure 6: Offloading Decision Quality Metrics", fontsize=13)
    fig.tight_layout()

    out_p = Path(output_dir) / "fig6_regret_and_bad_offload.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def plot_reuse_ablation(
    ablation_csv: str = "results/raw/ablation_sweep_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 7: Reuse-aware ablation across lambda weights."""
    set_paper_style()
    df = pd.read_csv(ablation_csv)

    reuse_sub = df[(df["policy_name"] == "Reuse-aware") & (df["workload_type"] == "REUSE")]
    grouped = reuse_sub.groupby("lambda_weight").agg({
        "selected_model_ns": "mean",
        "regret_ns": "mean",
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(7.0, 4.3))
    ax2 = ax1.twinx()

    l1 = ax1.plot(grouped["lambda_weight"], grouped["selected_model_ns"] / 1000.0, "o-", color="#1f77b4", label="Execution Latency (us)")
    l2 = ax2.plot(grouped["lambda_weight"], grouped["regret_ns"] / 1000.0, "s--", color="#d62728", label="Decision Regret (us)")

    ax1.set_xlabel("Reuse Penalty Weight (lambda)")
    ax1.set_ylabel("Mean Execution Latency (us)", color="#1f77b4")
    ax2.set_ylabel("Mean Decision Regret (us)", color="#d62728")
    ax1.set_title("Figure 7: Reuse-Aware Policy Ablation and Sensitivity to lambda")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)
    fig.tight_layout()

    out_p = Path(output_dir) / "fig7_reuse_ablation.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def plot_multi_seed_variability(
    baseline_csv: str = "results/raw/baseline_comparison_raw.csv",
    output_dir: str = "results/figures",
) -> Path:
    """Figure 8: Execution latency variability across master seeds 42-51."""
    set_paper_style()
    df = pd.read_csv(baseline_csv)

    eval_policies = ["CPU-cache", "PIM-always", "Cost-based", "Reuse-aware", "Oracle"]
    sub = df[df["policy_name"].isin(eval_policies)].copy()
    sub["latency_us"] = sub["selected_model_ns"] / 1000.0

    # Group by seed and policy
    seed_means = sub.groupby(["master_seed", "policy_name"])["latency_us"].mean().unstack()[eval_policies]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    data_to_plot = [seed_means[p].dropna().values for p in eval_policies]
    box = ax.boxplot(data_to_plot,
                     tick_labels=eval_policies,
                     patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Seed Mean Execution Latency (us)")
    ax.set_title("Figure 8: Empirical Variability Across Master Seeds (42-51)")
    ax.set_xticks(range(1, len(eval_policies) + 1))
    ax.set_xticklabels(eval_policies, rotation=15, ha="right")
    fig.tight_layout()

    out_p = Path(output_dir) / "fig8_multi_seed_variability.png"
    fig.savefig(out_p)
    fig.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)
    return out_p


def generate_all_plots(results_dir: str = "results") -> None:
    """Batch generate all 8 publication figures from executed simulation CSV results."""
    raw_dir = Path(results_dir) / "raw"
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 1: Cache Hit/Miss Rate vs. Working-Set Size...")
    plot_cache_hit_miss_vs_workingset(str(raw_dir / "crossover_sweep_raw.csv"), str(fig_dir))

    print("Generating Figure 2: CPU vs. PIM Latency and Crossover Boundary...")
    plot_cpu_pim_crossover(str(raw_dir / "crossover_sweep_raw.csv"), str(fig_dir))

    print("Generating Figure 3: Overall Execution Latency Across Policies...")
    plot_policy_performance_comparison(str(raw_dir / "baseline_comparison_raw.csv"), str(fig_dir))

    print("Generating Figure 4: PIM vs. CPU Speedup Crossover Heatmap...")
    plot_crossover_heatmap(str(raw_dir / "crossover_sweep_raw.csv"), str(raw_dir / "transfer_sensitivity_raw.csv"), str(fig_dir))

    print("Generating Figure 5: Policy Speedup Across Workload Types...")
    plot_policy_speedup_by_workload(str(raw_dir / "baseline_comparison_raw.csv"), str(fig_dir))

    print("Generating Figure 6: Decision Regret and Bad-Offload Metrics...")
    plot_regret_and_bad_offload(str(raw_dir / "baseline_comparison_raw.csv"), str(fig_dir))

    print("Generating Figure 7: Reuse-Aware Policy Ablation (lambda sweep)...")
    plot_reuse_ablation(str(raw_dir / "ablation_sweep_raw.csv"), str(fig_dir))

    print("Generating Figure 8: Multi-Seed Empirical Variability...")
    plot_multi_seed_variability(str(raw_dir / "baseline_comparison_raw.csv"), str(fig_dir))

    print(f"All 8 publication figures saved to {fig_dir} (PNG and PDF formats).")

