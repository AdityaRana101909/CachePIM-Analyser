"""Execution offloading policies for CachePIM.

Implements six distinct execution policies sharing a unified interface:
1. CPU-cache: Always execute through the CPU cache.
2. CPU-no-cache: Direct-memory baseline without cache reuse.
3. PIM-always: Offload all supported kernels to PIM.
4. Cost-based: Analytical cost comparison using estimated parameters.
5. Reuse-aware: Candidate policy incorporating a dimensionally consistent reuse penalty.
6. Oracle: Optimal offline reference using actual simulated costs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from cachepim.config import CachePIMConfig
from cachepim.cpu_model import CPUModel
from cachepim.pim_model import PIMModel
from cachepim.types import (
    CacheStats,
    ExecutionPath,
    PolicyDecision,
    WorkloadDescriptor,
)


def estimate_cache_hits_and_misses(
    descriptor: WorkloadDescriptor,
    capacity_bytes: int,
    line_size_bytes: int,
) -> tuple[int, int]:
    """Analytically estimate hits and misses from workload descriptor without executing trace."""
    total_accesses = descriptor.operation_count
    if total_accesses <= 0:
        return 0, 0

    working_set_bytes = max(line_size_bytes, descriptor.working_set_bytes)
    unique_lines = max(1, working_set_bytes // line_size_bytes)
    cache_lines = max(1, capacity_bytes // line_size_bytes)

    # Compulsory misses
    compulsory_misses = min(total_accesses, unique_lines)

    if working_set_bytes <= capacity_bytes:
        # Working set fits completely in cache: compulsory misses only, remainder hits
        estimated_misses = compulsory_misses
        estimated_hits = total_accesses - estimated_misses
    else:
        # Working set exceeds cache capacity: capacity thrashing occurs
        if descriptor.reusable_bytes > 0:
            # Hot-set partially fits in cache
            hot_lines = max(1, descriptor.reusable_bytes // line_size_bytes)
            if hot_lines <= cache_lines:
                # Hot set fits in cache: accesses to hot set hit after compulsory miss
                hot_ratio = descriptor.hot_cold_ratio if descriptor.hot_cold_ratio is not None else 0.8
                hot_accesses = int(total_accesses * hot_ratio)
                cold_accesses = total_accesses - hot_accesses

                hot_hits = max(0, hot_accesses - hot_lines)
                estimated_hits = hot_hits
                estimated_misses = total_accesses - estimated_hits
            else:
                # Even hot set thrashes
                fit_ratio = cache_lines / hot_lines
                estimated_hits = int((total_accesses - compulsory_misses) * fit_ratio * 0.5)
                estimated_misses = total_accesses - estimated_hits
        else:
            # Low/no reuse (streaming or uniform random exceeding cache)
            estimated_hits = 0
            estimated_misses = total_accesses

    return max(0, estimated_hits), max(0, estimated_misses)


class BasePolicy(ABC):
    """Abstract base class for all offloading policies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def decide(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        actual_cpu_ns: Optional[float] = None,
        actual_pim_ns: Optional[float] = None,
    ) -> PolicyDecision:
        """Make an offloading decision between CPU and PIM."""
        pass


class CPUCachePolicy(BasePolicy):
    """Always execute through CPU cache."""

    def __init__(self) -> None:
        super().__init__("CPU-cache")

    def decide(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        actual_cpu_ns: Optional[float] = None,
        actual_pim_ns: Optional[float] = None,
    ) -> PolicyDecision:
        hits_est, misses_est = estimate_cache_hits_and_misses(
            descriptor, config.cache.capacity_bytes, config.cache.line_size_bytes
        )
        stats = CacheStats(
            accesses=descriptor.operation_count,
            hits=hits_est,
            misses=misses_est,
            dirty_evictions=0,
        )
        cpu_cost = CPUModel(config.cpu).evaluate(stats, descriptor.operation_count)

        pim_model = PIMModel(config.pim, config.transfer)
        pim_cost = pim_model.evaluate(descriptor)
        pim_est = pim_cost.total_pim_time_ns if pim_cost else None

        return PolicyDecision(
            policy_name=self.name,
            selected_path=ExecutionPath.CPU.value,
            decision_reason="ALWAYS_CPU_CACHE",
            cpu_estimated_ns=cpu_cost.total_cpu_time_ns,
            pim_estimated_ns=pim_est if pim_est is not None else 0.0,
            policy_overhead_ns=0.0,
        )


class CPUNoCachePolicy(BasePolicy):
    """Baseline executing directly through memory without cache reuse."""

    def __init__(self) -> None:
        super().__init__("CPU-no-cache")

    def decide(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        actual_cpu_ns: Optional[float] = None,
        actual_pim_ns: Optional[float] = None,
    ) -> PolicyDecision:
        cpu_cost = CPUModel(config.cpu).evaluate_no_cache(
            descriptor.operation_count, descriptor.operation_count
        )
        pim_model = PIMModel(config.pim, config.transfer)
        pim_cost = pim_model.evaluate(descriptor)
        pim_est = pim_cost.total_pim_time_ns if pim_cost else None

        return PolicyDecision(
            policy_name=self.name,
            selected_path=ExecutionPath.CPU.value,
            decision_reason="DIRECT_MEMORY_BASELINE",
            cpu_estimated_ns=cpu_cost.total_cpu_time_ns,
            pim_estimated_ns=pim_est if pim_est is not None else 0.0,
            policy_overhead_ns=0.0,
        )


class PIMAlwaysPolicy(BasePolicy):
    """Always offload eligible kernels to PIM; fallback to CPU if unsupported."""

    def __init__(self) -> None:
        super().__init__("PIM-always")

    def decide(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        actual_cpu_ns: Optional[float] = None,
        actual_pim_ns: Optional[float] = None,
    ) -> PolicyDecision:
        hits_est, misses_est = estimate_cache_hits_and_misses(
            descriptor, config.cache.capacity_bytes, config.cache.line_size_bytes
        )
        stats = CacheStats(
            accesses=descriptor.operation_count,
            hits=hits_est,
            misses=misses_est,
            dirty_evictions=0,
        )
        cpu_cost = CPUModel(config.cpu).evaluate(stats, descriptor.operation_count)

        if not descriptor.pim_eligible:
            return PolicyDecision(
                policy_name=self.name,
                selected_path=ExecutionPath.CPU.value,
                decision_reason="UNSUPPORTED_BY_PIM",
                cpu_estimated_ns=cpu_cost.total_cpu_time_ns,
                pim_estimated_ns=0.0,
                policy_overhead_ns=0.0,
            )

        pim_model = PIMModel(config.pim, config.transfer)
        pim_cost = pim_model.evaluate(descriptor)
        pim_est = pim_cost.total_pim_time_ns if pim_cost else 0.0

        return PolicyDecision(
            policy_name=self.name,
            selected_path=ExecutionPath.PIM.value,
            decision_reason="ALWAYS_PIM",
            cpu_estimated_ns=cpu_cost.total_cpu_time_ns,
            pim_estimated_ns=pim_est,
            policy_overhead_ns=0.0,
        )


class CostBasedPolicy(BasePolicy):
    """Offload to PIM if estimated PIM latency < estimated CPU latency; ties choose CPU."""

    def __init__(self) -> None:
        super().__init__("Cost-based")

    def decide(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        actual_cpu_ns: Optional[float] = None,
        actual_pim_ns: Optional[float] = None,
    ) -> PolicyDecision:
        hits_est, misses_est = estimate_cache_hits_and_misses(
            descriptor, config.cache.capacity_bytes, config.cache.line_size_bytes
        )
        stats = CacheStats(
            accesses=descriptor.operation_count,
            hits=hits_est,
            misses=misses_est,
            dirty_evictions=0,
        )
        cpu_cost = CPUModel(config.cpu).evaluate(stats, descriptor.operation_count)
        cpu_est = cpu_cost.total_cpu_time_ns

        if not descriptor.pim_eligible:
            return PolicyDecision(
                policy_name=self.name,
                selected_path=ExecutionPath.CPU.value,
                decision_reason="UNSUPPORTED_BY_PIM",
                cpu_estimated_ns=cpu_est,
                pim_estimated_ns=0.0,
                policy_overhead_ns=0.0,
            )

        pim_model = PIMModel(config.pim, config.transfer)
        pim_cost = pim_model.evaluate(descriptor)
        pim_est = pim_cost.total_pim_time_ns if pim_cost else float("inf")

        if pim_est < cpu_est:
            selected = ExecutionPath.PIM.value
            reason = "COST_PIM_LOWER"
        else:
            selected = ExecutionPath.CPU.value
            reason = "COST_CPU_LOWER_OR_EQUAL"

        return PolicyDecision(
            policy_name=self.name,
            selected_path=selected,
            decision_reason=reason,
            cpu_estimated_ns=cpu_est,
            pim_estimated_ns=pim_est,
            policy_overhead_ns=0.0,
        )


class ReuseAwarePolicy(BasePolicy):
    """Candidate policy incorporating dimensionally consistent reuse preservation cost.

    T_PIM_adjusted = T_PIM_estimated + lambda * R_cost
    R_cost = (effective_reusable_bytes / cache_capacity_bytes) * estimated_CPU_reuse_benefit_ns
    """

    def __init__(self) -> None:
        super().__init__("Reuse-aware")

    def calculate_reuse_cost(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        estimated_hits: int,
    ) -> float:
        """Compute dimensionally consistent reuse preservation cost in nanoseconds."""
        cap_bytes = config.cache.capacity_bytes
        if cap_bytes <= 0 or descriptor.working_set_bytes <= 0 or estimated_hits <= 0:
            return 0.0

        # Cache coverage ratio: how much of the working set can be retained in cache
        fit_ratio = min(1.0, cap_bytes / descriptor.working_set_bytes)
        effective_reusable_bytes = descriptor.reusable_bytes * fit_ratio

        # Time saved per hit in nanoseconds
        hit_benefit_per_access = config.cpu.miss_service_ns - config.cpu.hit_service_ns
        estimated_cpu_reuse_benefit_ns = estimated_hits * hit_benefit_per_access

        # Normalization by cache capacity ensures dimension is in nanoseconds:
        # (bytes / bytes) * nanoseconds = nanoseconds
        reusable_fraction = min(1.0, effective_reusable_bytes / cap_bytes)
        r_cost = reusable_fraction * estimated_cpu_reuse_benefit_ns
        return max(0.0, r_cost)

    def decide(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        actual_cpu_ns: Optional[float] = None,
        actual_pim_ns: Optional[float] = None,
    ) -> PolicyDecision:
        hits_est, misses_est = estimate_cache_hits_and_misses(
            descriptor, config.cache.capacity_bytes, config.cache.line_size_bytes
        )
        stats = CacheStats(
            accesses=descriptor.operation_count,
            hits=hits_est,
            misses=misses_est,
            dirty_evictions=0,
        )
        cpu_cost = CPUModel(config.cpu).evaluate(stats, descriptor.operation_count)
        cpu_est = cpu_cost.total_cpu_time_ns

        if not descriptor.pim_eligible:
            return PolicyDecision(
                policy_name=self.name,
                selected_path=ExecutionPath.CPU.value,
                decision_reason="UNSUPPORTED_BY_PIM",
                cpu_estimated_ns=cpu_est,
                pim_estimated_ns=0.0,
                policy_overhead_ns=0.0,
            )

        pim_model = PIMModel(config.pim, config.transfer)
        pim_cost = pim_model.evaluate(descriptor)
        pim_est = pim_cost.total_pim_time_ns if pim_cost else float("inf")

        lam = config.policy.reuse_penalty_weight
        r_cost = self.calculate_reuse_cost(descriptor, config, hits_est)
        pim_adjusted = pim_est + (lam * r_cost)

        if pim_adjusted < cpu_est:
            selected = ExecutionPath.PIM.value
            reason = "REUSE_AWARE_PIM_LOWER"
        else:
            selected = ExecutionPath.CPU.value
            if pim_est < cpu_est:
                reason = "REUSE_AWARE_PRESERVED_CPU_DUE_TO_REUSE"
            else:
                reason = "REUSE_AWARE_CPU_LOWER_NATURALLY"

        return PolicyDecision(
            policy_name=self.name,
            selected_path=selected,
            decision_reason=reason,
            cpu_estimated_ns=cpu_est,
            pim_estimated_ns=pim_est,
            policy_overhead_ns=0.0,  # Zero runtime overhead for analytical descriptor inspection
        )


class OraclePolicy(BasePolicy):
    """Optimal offline reference selecting min(T_CPU_actual, T_PIM_actual)."""

    def __init__(self) -> None:
        super().__init__("Oracle")

    def decide(
        self,
        descriptor: WorkloadDescriptor,
        config: CachePIMConfig,
        actual_cpu_ns: Optional[float] = None,
        actual_pim_ns: Optional[float] = None,
    ) -> PolicyDecision:
        if actual_cpu_ns is None or actual_pim_ns is None:
            raise ValueError("OraclePolicy requires actual modeled costs to make optimal decision")

        if not descriptor.pim_eligible:
            return PolicyDecision(
                policy_name=self.name,
                selected_path=ExecutionPath.CPU.value,
                decision_reason="ORACLE_UNSUPPORTED_BY_PIM",
                cpu_estimated_ns=actual_cpu_ns,
                pim_estimated_ns=0.0,
                policy_overhead_ns=0.0,
            )

        if actual_pim_ns < actual_cpu_ns:
            selected = ExecutionPath.PIM.value
            reason = "ORACLE_PIM_OPTIMAL"
        else:
            selected = ExecutionPath.CPU.value
            reason = "ORACLE_CPU_OPTIMAL"

        return PolicyDecision(
            policy_name=self.name,
            selected_path=selected,
            decision_reason=reason,
            cpu_estimated_ns=actual_cpu_ns,
            pim_estimated_ns=actual_pim_ns,
            policy_overhead_ns=0.0,
        )
