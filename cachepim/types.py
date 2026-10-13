"""Core data types, enums, and schema definitions for CachePIM.

Provides structured types for traces, cache statistics, cost breakdowns,
policy decisions, and simulation results conforming to the specification schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OperationType(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


class KernelType(str, Enum):
    VECTOR_OP = "VECTOR_OP"
    REDUCTION = "REDUCTION"
    FILTER = "FILTER"
    UNSUPPORTED_KERNEL = "UNSUPPORTED_KERNEL"


class WorkloadPattern(str, Enum):
    STREAM = "STREAM"
    REUSE = "REUSE"
    RANDOM = "RANDOM"
    MIXED_PHASE = "MIXED_PHASE"


class ExecutionPath(str, Enum):
    CPU = "CPU"
    PIM = "PIM"


@dataclass(frozen=True)
class TraceRecord:
    """Individual memory access record in a synthetic trace."""
    sequence_number: int
    address: int
    operation: str  # "READ" or "WRITE"
    task_id: str
    phase_id: str


@dataclass
class WorkloadDescriptor:
    """Metadata describing a generated computational kernel and memory trace."""
    workload_type: str
    seed: int
    kernel_type: str
    element_count: int
    bytes_read: int
    bytes_written: int
    operation_count: int
    read_write_ratio: float
    working_set_bytes: int
    trace_length: int
    pim_eligible: bool
    reusable_bytes: int = 0
    hot_cold_ratio: Optional[float] = None
    phase_mix: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)


@dataclass
class CacheEvent:
    """Record produced for each memory access processed by the cache model."""
    line_id: int
    hit: bool
    miss: bool
    evicted_line_id: Optional[int]
    operation: str
    sequence_number: int


@dataclass
class CacheStats:
    """Aggregated cache statistics for a simulated trace."""
    accesses: int = 0
    reads: int = 0
    writes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    unique_lines: int = 0
    bytes_fetched_from_memory: int = 0
    dirty_evictions: int = 0


@dataclass
class CPUCostBreakdown:
    """Component-level analytical cost breakdown for CPU execution in nanoseconds."""
    cache_service_time_ns: float
    compute_time_ns: float
    writeback_time_ns: float
    total_cpu_time_ns: float


@dataclass
class PIMCostBreakdown:
    """Component-level analytical cost breakdown for PIM execution in nanoseconds."""
    h2p_transfer_ns: float
    compute_ns: float
    p2h_transfer_ns: float
    launch_overhead_ns: float
    sync_overhead_ns: float
    imbalance_overhead_ns: float
    total_pim_time_ns: float
    bank_imbalance_factor: float
    per_bank_work: List[int]


@dataclass
class PolicyDecision:
    """Offloading decision produced by an execution policy."""
    policy_name: str
    selected_path: str  # "CPU" or "PIM"
    decision_reason: str
    cpu_estimated_ns: float
    pim_estimated_ns: float
    policy_overhead_ns: float = 0.0


@dataclass
class SimulationResult:
    """Full execution row conforming to Section 17 result schema."""
    run_id: str
    seed: int
    trace_id: str
    workload_type: str
    workload_parameters_json: str
    config_hash: str
    policy_name: str
    selected_path: str
    decision_reason: str
    cpu_estimated_ns: float
    pim_estimated_ns: Optional[float]
    cpu_actual_model_ns: float
    pim_actual_model_ns: Optional[float]
    selected_model_ns: float
    cache_accesses: int
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    cache_hit_rate: float
    cache_miss_rate: float
    input_bytes: int
    output_bytes: int
    host_to_pim_bytes: int
    pim_to_host_bytes: int
    bank_count: int
    bank_imbalance: float
    policy_overhead_ns: float
    regret_ns: float
    normalized_regret: float
    bad_offload: int
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
