"""Unified simulation engine for CachePIM.

Executes workload traces through cache, CPU, PIM models, evaluates policies,
and compiles structured simulation results matching the Section 17 schema.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

from cachepim.cache_model import CacheModel
from cachepim.config import CachePIMConfig
from cachepim.cpu_model import CPUModel
from cachepim.pim_model import PIMModel
from cachepim.policies import (
    BasePolicy,
    CostBasedPolicy,
    CPUCachePolicy,
    CPUNoCachePolicy,
    OraclePolicy,
    PIMAlwaysPolicy,
    ReuseAwarePolicy,
)
from cachepim.types import (
    ExecutionPath,
    PolicyDecision,
    SimulationResult,
    TraceRecord,
    WorkloadDescriptor,
)


class Simulator:
    """Core simulation engine."""

    def __init__(self, config: CachePIMConfig):
        config.validate()
        self.config = config
        self.cache_model = CacheModel(config.cache)
        self.cpu_model = CPUModel(config.cpu)
        self.pim_model = PIMModel(config.pim, config.transfer)

    def simulate_trace(
        self,
        trace: List[TraceRecord],
        descriptor: WorkloadDescriptor,
        policy: BasePolicy,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> SimulationResult:
        """Simulate a single trace execution under the given policy."""
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        if trace_id is None:
            trace_id = f"trace_{descriptor.workload_type}_{descriptor.seed}"

        # 1. Simulate LLC CPU cache path
        self.cache_model.reset()
        cache_stats = self.cache_model.run(trace)

        # 2. Compute actual CPU modeled cost
        cpu_breakdown = self.cpu_model.evaluate(cache_stats, descriptor.operation_count)
        cpu_actual_ns = cpu_breakdown.total_cpu_time_ns

        # 3. Compute actual PIM modeled cost if eligible
        pim_breakdown = self.pim_model.evaluate(descriptor)
        pim_actual_ns = pim_breakdown.total_pim_time_ns if pim_breakdown else None

        # 4. Invoke policy decision
        decision: PolicyDecision = policy.decide(
            descriptor=descriptor,
            config=self.config,
            actual_cpu_ns=cpu_actual_ns,
            actual_pim_ns=pim_actual_ns,
        )

        # 5. Determine actual execution time for the chosen path
        if decision.selected_path == ExecutionPath.CPU.value:
            if policy.name == "CPU-no-cache":
                no_cache_breakdown = self.cpu_model.evaluate_no_cache(
                    descriptor.operation_count, descriptor.operation_count
                )
                selected_model_ns = no_cache_breakdown.total_cpu_time_ns
            else:
                selected_model_ns = cpu_actual_ns
        else:
            assert pim_actual_ns is not None, "Policy selected PIM for ineligible workload"
            selected_model_ns = pim_actual_ns

        # Add any policy decision overhead
        selected_model_ns += decision.policy_overhead_ns

        # 6. Oracle evaluation and Regret accounting
        if pim_actual_ns is not None:
            oracle_optimal_ns = min(cpu_actual_ns, pim_actual_ns)
            bad_offload = 1 if (decision.selected_path == ExecutionPath.PIM.value and cpu_actual_ns < pim_actual_ns) else 0
        else:
            oracle_optimal_ns = cpu_actual_ns
            bad_offload = 0

        # Regret invariant: non-negative (tolerance for floating-point inaccuracy)
        raw_regret = selected_model_ns - oracle_optimal_ns
        regret_ns = max(0.0, raw_regret) if raw_regret >= -1e-6 else 0.0

        normalized_regret = (regret_ns / oracle_optimal_ns) if oracle_optimal_ns > 0 else 0.0

        # 7. Data transfer accounting
        if decision.selected_path == ExecutionPath.PIM.value:
            host_to_pim_bytes = descriptor.bytes_read
            pim_to_host_bytes = descriptor.bytes_written
        else:
            host_to_pim_bytes = 0
            pim_to_host_bytes = 0

        bank_imbalance = pim_breakdown.bank_imbalance_factor if pim_breakdown else 1.0

        return SimulationResult(
            run_id=run_id,
            seed=descriptor.seed,
            trace_id=trace_id,
            workload_type=descriptor.workload_type,
            workload_parameters_json=descriptor.to_json(),
            config_hash=self.config.get_hash(),
            policy_name=policy.name,
            selected_path=decision.selected_path,
            decision_reason=decision.decision_reason,
            cpu_estimated_ns=decision.cpu_estimated_ns,
            pim_estimated_ns=decision.pim_estimated_ns,
            cpu_actual_model_ns=cpu_actual_ns,
            pim_actual_model_ns=pim_actual_ns,
            selected_model_ns=selected_model_ns,
            cache_accesses=cache_stats.accesses,
            cache_hits=cache_stats.hits,
            cache_misses=cache_stats.misses,
            cache_evictions=cache_stats.evictions,
            cache_hit_rate=cache_stats.hit_rate,
            cache_miss_rate=cache_stats.miss_rate,
            input_bytes=descriptor.bytes_read,
            output_bytes=descriptor.bytes_written,
            host_to_pim_bytes=host_to_pim_bytes,
            pim_to_host_bytes=pim_to_host_bytes,
            bank_count=self.config.pim.bank_count,
            bank_imbalance=bank_imbalance,
            policy_overhead_ns=decision.policy_overhead_ns,
            regret_ns=regret_ns,
            normalized_regret=normalized_regret,
            bad_offload=bad_offload,
            model_version=self.config.version,
        )

    def run_all_policies_on_trace(
        self,
        trace: List[TraceRecord],
        descriptor: WorkloadDescriptor,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> List[SimulationResult]:
        """Run all 6 baseline and proposed policies on the identical trace and workload descriptor."""
        policies: List[BasePolicy] = [
            CPUCachePolicy(),
            CPUNoCachePolicy(),
            PIMAlwaysPolicy(),
            CostBasedPolicy(),
            ReuseAwarePolicy(),
            OraclePolicy(),
        ]
        results: List[SimulationResult] = []
        for pol in policies:
            res = self.simulate_trace(
                trace=trace,
                descriptor=descriptor,
                policy=pol,
                run_id=run_id,
                trace_id=trace_id,
            )
            results.append(res)
        return results
