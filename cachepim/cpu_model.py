"""CPU analytical cost model for CachePIM.

Calculates execution latency using cache statistics, operation counts,
and architectural timing parameters (in nanoseconds):
T_CPU = N_hit * hit_service_ns + N_miss * miss_service_ns + N_ops * compute_ns_per_op + T_writeback
"""

from __future__ import annotations

from typing import Optional

from cachepim.config import CPUConfig
from cachepim.types import CacheStats, CPUCostBreakdown


class CPUModel:
    """Analytical CPU execution cost model."""

    def __init__(self, config: CPUConfig):
        config.validate()
        self.config = config
        self.hit_service_ns: float = config.hit_service_ns
        self.miss_service_ns: float = config.miss_service_ns
        self.compute_ns_per_op: float = config.compute_ns_per_op

    def evaluate(
        self,
        cache_stats: CacheStats,
        operation_count: int,
        dirty_evictions: Optional[int] = None,
    ) -> CPUCostBreakdown:
        """Compute CPU cost breakdown for execution through the CPU cache.

        miss_service_ns represents the complete service time for a miss
        and replaces (rather than adds to) hit service time.
        """
        if dirty_evictions is None:
            dirty_evictions = cache_stats.dirty_evictions

        cache_service_time_ns = (
            (cache_stats.hits * self.hit_service_ns)
            + (cache_stats.misses * self.miss_service_ns)
        )
        compute_time_ns = operation_count * self.compute_ns_per_op
        # Writeback cost for dirty evicted lines requiring memory writeout
        writeback_time_ns = dirty_evictions * self.miss_service_ns

        total_cpu_time_ns = cache_service_time_ns + compute_time_ns + writeback_time_ns

        return CPUCostBreakdown(
            cache_service_time_ns=cache_service_time_ns,
            compute_time_ns=compute_time_ns,
            writeback_time_ns=writeback_time_ns,
            total_cpu_time_ns=total_cpu_time_ns,
        )

    def evaluate_no_cache(
        self,
        access_count: int,
        operation_count: int,
    ) -> CPUCostBreakdown:
        """Compute CPU cost breakdown for the CPU-no-cache baseline.

        Models direct memory service for each access without cache reuse.
        """
        cache_service_time_ns = access_count * self.miss_service_ns
        compute_time_ns = operation_count * self.compute_ns_per_op
        writeback_time_ns = 0.0
        total_cpu_time_ns = cache_service_time_ns + compute_time_ns

        return CPUCostBreakdown(
            cache_service_time_ns=cache_service_time_ns,
            compute_time_ns=compute_time_ns,
            writeback_time_ns=writeback_time_ns,
            total_cpu_time_ns=total_cpu_time_ns,
        )
