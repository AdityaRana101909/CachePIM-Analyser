"""Experimental runner and parameter sweep framework for CachePIM.

Executes systematic parameter sweeps, multi-seed baseline comparisons,
ablation studies, and held-out test evaluations. Exports raw per-run CSVs,
seed-aggregated statistical summaries, and reproducibility manifests.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cachepim.config import CachePIMConfig
from cachepim.metrics import calculate_seed_aggregates, paired_policy_comparison
from cachepim.simulator import Simulator
from cachepim.workloads import WorkloadGenerator


class ExperimentRunner:
    """Manages execution of experiment matrices and data persistence."""

    def __init__(self, config: CachePIMConfig, output_dir: str = "results"):
        config.validate()
        self.config = config
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.summary_dir = self.output_dir / "summary"
        self.figures_dir = self.output_dir / "figures"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_baseline_comparison(
        self,
        seeds: Optional[List[int]] = None,
        traces_per_seed: Optional[int] = None,
        patterns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run all 6 policies over all workload patterns and reproducible master seeds."""
        if seeds is None:
            seeds = self.config.experiment.master_seeds
        if traces_per_seed is None:
            traces_per_seed = self.config.experiment.traces_per_seed
        if patterns is None:
            patterns = self.config.experiment.default_workload_patterns

        gen = WorkloadGenerator(line_size_bytes=self.config.cache.line_size_bytes)
        sim = Simulator(self.config)

        all_results = []
        run_count = 0

        for seed in seeds:
            for pattern in patterns:
                for trace_idx in range(traces_per_seed):
                    run_count += 1
                    trace_seed = seed * 1000 + trace_idx
                    trace_id = f"baseline_{pattern}_s{seed}_t{trace_idx}"

                    # Varied working-set size based on trace index
                    ws_sizes = [16384, 32768, 65536, 131072, 262144]
                    ws_bytes = ws_sizes[trace_idx % len(ws_sizes)]

                    trace, desc = gen.generate(
                        pattern=pattern,
                        seed=trace_seed,
                        working_set_bytes=ws_bytes,
                        access_count=2048,
                    )
                    res_list = sim.run_all_policies_on_trace(
                        trace=trace,
                        descriptor=desc,
                        run_id=f"run_{run_count}",
                        trace_id=trace_id,
                    )
                    for r in res_list:
                        d = r.to_dict()
                        d["master_seed"] = seed
                        all_results.append(d)

        df = pd.DataFrame(all_results)
        csv_path = self.raw_dir / "baseline_comparison_raw.csv"
        df.to_csv(csv_path, index=False)

        summary_df = calculate_seed_aggregates(df)
        summary_path = self.summary_dir / "baseline_comparison_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        return df

    def run_crossover_sweep(
        self,
        working_set_sizes_kb: Optional[List[int]] = None,
        cache_capacities_kb: Optional[List[int]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sweep working-set sizes against cache capacities to map the CPU/PIM crossover."""
        if working_set_sizes_kb is None:
            working_set_sizes_kb = [8, 16, 24, 32, 48, 64, 96, 128, 256]
        if cache_capacities_kb is None:
            cache_capacities_kb = [16, 32, 64, 128]

        all_results = []
        gen = WorkloadGenerator(line_size_bytes=self.config.cache.line_size_bytes)

        for cap_kb in cache_capacities_kb:
            cfg = CachePIMConfig.from_dict(self.config.to_dict())
            cfg.cache.capacity_bytes = cap_kb * 1024
            sim = Simulator(cfg)

            for ws_kb in working_set_sizes_kb:
                ws_bytes = ws_kb * 1024
                for pattern in ["STREAM", "REUSE"]:
                    trace, desc = gen.generate(
                        pattern=pattern,
                        seed=seed,
                        working_set_bytes=ws_bytes,
                        access_count=2048,
                    )
                    trace_id = f"crossover_cap{cap_kb}k_ws{ws_kb}k_{pattern}"
                    res_list = sim.run_all_policies_on_trace(trace, desc, trace_id=trace_id)
                    for r in res_list:
                        d = r.to_dict()
                        d["cache_capacity_kb"] = cap_kb
                        d["working_set_kb"] = ws_kb
                        all_results.append(d)

        df = pd.DataFrame(all_results)
        df.to_csv(self.raw_dir / "crossover_sweep_raw.csv", index=False)
        return df

    def run_transfer_sensitivity(
        self,
        bandwidths_gb_s: Optional[List[float]] = None,
        working_set_kb: int = 64,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sweep host-to-PIM transfer bandwidth to analyze interconnect sensitivity."""
        if bandwidths_gb_s is None:
            bandwidths_gb_s = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

        all_results = []
        gen = WorkloadGenerator(line_size_bytes=self.config.cache.line_size_bytes)

        for bw in bandwidths_gb_s:
            cfg = CachePIMConfig.from_dict(self.config.to_dict())
            cfg.transfer.host_to_pim_bandwidth_bytes_per_ns = bw
            cfg.transfer.pim_to_host_bandwidth_bytes_per_ns = bw
            sim = Simulator(cfg)

            for pattern in ["STREAM", "REUSE", "RANDOM"]:
                trace, desc = gen.generate(
                    pattern=pattern,
                    seed=seed,
                    working_set_bytes=working_set_kb * 1024,
                    access_count=2048,
                )
                trace_id = f"transfer_bw{bw}g_{pattern}"
                res_list = sim.run_all_policies_on_trace(trace, desc, trace_id=trace_id)
                for r in res_list:
                    d = r.to_dict()
                    d["transfer_bandwidth_gb_s"] = bw
                    d["working_set_kb"] = working_set_kb
                    all_results.append(d)

        df = pd.DataFrame(all_results)
        df.to_csv(self.raw_dir / "transfer_sensitivity_raw.csv", index=False)
        return df

    def run_ablation_sweep(
        self,
        lambda_weights: Optional[List[float]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Ablation sweep varying reuse-penalty weight lambda from 0.0 (no penalty) to 4.0."""
        if lambda_weights is None:
            lambda_weights = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

        all_results = []
        gen = WorkloadGenerator(line_size_bytes=self.config.cache.line_size_bytes)

        for lam in lambda_weights:
            cfg = CachePIMConfig.from_dict(self.config.to_dict())
            cfg.policy.reuse_penalty_weight = lam
            sim = Simulator(cfg)

            for pattern in ["REUSE", "STREAM", "MIXED_PHASE"]:
                for ws_kb in [16, 32, 64, 128]:
                    trace, desc = gen.generate(
                        pattern=pattern,
                        seed=seed,
                        working_set_bytes=ws_kb * 1024,
                        access_count=2048,
                    )
                    trace_id = f"ablation_lam{lam}_{pattern}_ws{ws_kb}k"
                    res_list = sim.run_all_policies_on_trace(trace, desc, trace_id=trace_id)
                    for r in res_list:
                        d = r.to_dict()
                        d["lambda_weight"] = lam
                        d["working_set_kb"] = ws_kb
                        all_results.append(d)

        df = pd.DataFrame(all_results)
        df.to_csv(self.raw_dir / "ablation_sweep_raw.csv", index=False)
        return df

    def run_held_out_evaluation(self) -> Dict[str, Any]:
        """Evaluate development seeds vs held-out test seeds without re-tuning."""
        dev_seeds = self.config.experiment.dev_seeds
        held_out_seeds = self.config.experiment.held_out_seeds

        df_dev = self.run_baseline_comparison(seeds=dev_seeds, traces_per_seed=3)
        df_test = self.run_baseline_comparison(seeds=held_out_seeds, traces_per_seed=3)

        paired_dev = paired_policy_comparison(df_dev, "Reuse-aware", "Cost-based")
        paired_test = paired_policy_comparison(df_test, "Reuse-aware", "Cost-based")

        eval_summary = {
            "dev_seeds": dev_seeds,
            "held_out_seeds": held_out_seeds,
            "dev_comparison": paired_dev,
            "test_comparison": paired_test,
        }

        with open(self.summary_dir / "held_out_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        return eval_summary

    def run_full_suite(self) -> None:
        """Run complete experimental test suite and write manifest."""
        print("[1/5] Running Baseline Comparison across master seeds 42-51...")
        self.run_baseline_comparison()

        print("[2/5] Running CPU/PIM Crossover Sweep...")
        self.run_crossover_sweep()

        print("[3/5] Running Transfer Cost Sensitivity Sweep...")
        self.run_transfer_sensitivity()

        print("[4/5] Running Reuse-Aware Policy Ablation Sweep...")
        self.run_ablation_sweep()

        print("[5/5] Running Held-out Seed Evaluation...")
        self.run_held_out_evaluation()

        # Save reproducibility manifest
        manifest = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config_hash": self.config.get_hash(),
            "model_version": self.config.version,
            "master_seeds": self.config.experiment.master_seeds,
            "dev_seeds": self.config.experiment.dev_seeds,
            "held_out_seeds": self.config.experiment.held_out_seeds,
            "raw_dir": str(self.raw_dir),
            "summary_dir": str(self.summary_dir),
        }
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Experimental suite completed! Results saved to {self.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CachePIM: Batch Experiment Runner")
    parser.add_argument("--config", type=str, default="config/experiment.yaml", help="Path to experiment config YAML")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save experimental results")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config/default.yaml")

    config = CachePIMConfig.from_yaml(cfg_path)
    runner = ExperimentRunner(config, output_dir=args.output_dir)
    runner.run_full_suite()


if __name__ == "__main__":
    main()
