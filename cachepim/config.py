"""Configuration management and validation for CachePIM.

Centralized schema for Cache, CPU, PIM, Transfer, Policy, and Experiment parameters.
Tracks parameter provenance (literature-derived, artifact-derived, or assumed)
and supports YAML and JSON serialization with reproducible config hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ParameterProvenance:
    """Provenance and documentation for an individual architectural parameter."""
    value: Any
    unit: str
    provenance: str  # "literature-derived", "artifact-derived", or "assumed"
    description: str
    reference: Optional[str] = None


@dataclass
class CacheConfig:
    capacity_bytes: int = 32768  # 32 KiB default
    line_size_bytes: int = 64
    replacement: str = "LRU"

    def validate(self) -> None:
        if self.capacity_bytes <= 0:
            raise ValueError(f"Cache capacity_bytes must be positive, got {self.capacity_bytes}")
        if self.line_size_bytes <= 0:
            raise ValueError(f"Cache line_size_bytes must be positive, got {self.line_size_bytes}")
        if self.capacity_bytes % self.line_size_bytes != 0:
            raise ValueError(
                f"Cache capacity ({self.capacity_bytes}) must be divisible by line_size ({self.line_size_bytes})"
            )
        if self.replacement.upper() != "LRU":
            raise ValueError(f"Only 'LRU' replacement supported in v1, got '{self.replacement}'")

    @property
    def number_of_lines(self) -> int:
        return self.capacity_bytes // self.line_size_bytes


@dataclass
class CPUConfig:
    frequency_hz: float = 2.5e9  # 2.5 GHz
    hit_service_ns: float = 2.0  # L3/LLC hit latency
    miss_service_ns: float = 80.0  # Main memory access latency
    compute_ns_per_op: float = 0.4  # ~1 cycle @ 2.5 GHz for basic arithmetic

    def validate(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError(f"CPU frequency_hz must be positive, got {self.frequency_hz}")
        if self.hit_service_ns < 0:
            raise ValueError(f"CPU hit_service_ns must be non-negative, got {self.hit_service_ns}")
        if self.miss_service_ns < self.hit_service_ns:
            raise ValueError(
                f"CPU miss_service_ns ({self.miss_service_ns}) must be >= hit_service_ns ({self.hit_service_ns})"
            )
        if self.compute_ns_per_op < 0:
            raise ValueError(f"CPU compute_ns_per_op must be non-negative, got {self.compute_ns_per_op}")


@dataclass
class PIMConfig:
    bank_count: int = 16
    effective_bandwidth_bytes_per_ns: float = 128.0  # 128 GB/s aggregate internal PIM bandwidth
    compute_ops_per_ns_per_bank: float = 1.0  # 1 op/ns per bank (1 GHz effective PIM core)
    launch_overhead_ns: float = 50.0  # Kernel launch overhead
    sync_overhead_ns: float = 20.0  # Bank synchronization overhead
    bank_imbalance_model: str = "max_over_mean"  # Imbalance calculation model

    def validate(self) -> None:
        if self.bank_count <= 0:
            raise ValueError(f"PIM bank_count must be positive, got {self.bank_count}")
        if self.effective_bandwidth_bytes_per_ns <= 0:
            raise ValueError(f"PIM effective_bandwidth_bytes_per_ns must be positive, got {self.effective_bandwidth_bytes_per_ns}")
        if self.compute_ops_per_ns_per_bank <= 0:
            raise ValueError(f"PIM compute_ops_per_ns_per_bank must be positive, got {self.compute_ops_per_ns_per_bank}")
        if self.launch_overhead_ns < 0:
            raise ValueError(f"PIM launch_overhead_ns must be non-negative, got {self.launch_overhead_ns}")
        if self.sync_overhead_ns < 0:
            raise ValueError(f"PIM sync_overhead_ns must be non-negative, got {self.sync_overhead_ns}")
        if self.bank_imbalance_model not in ("max_over_mean", "uniform"):
            raise ValueError(f"Unsupported bank_imbalance_model: {self.bank_imbalance_model}")


@dataclass
class TransferConfig:
    host_to_pim_bandwidth_bytes_per_ns: float = 16.0  # 16 GB/s PCIe 4.0 / CXL 1.1 link
    pim_to_host_bandwidth_bytes_per_ns: float = 16.0
    host_to_pim_fixed_ns: float = 100.0  # Link round-trip / command fixed overhead
    pim_to_host_fixed_ns: float = 100.0

    def validate(self) -> None:
        if self.host_to_pim_bandwidth_bytes_per_ns <= 0:
            raise ValueError(f"host_to_pim_bandwidth_bytes_per_ns must be positive, got {self.host_to_pim_bandwidth_bytes_per_ns}")
        if self.pim_to_host_bandwidth_bytes_per_ns <= 0:
            raise ValueError(f"pim_to_host_bandwidth_bytes_per_ns must be positive, got {self.pim_to_host_bandwidth_bytes_per_ns}")
        if self.host_to_pim_fixed_ns < 0:
            raise ValueError(f"host_to_pim_fixed_ns must be non-negative, got {self.host_to_pim_fixed_ns}")
        if self.pim_to_host_fixed_ns < 0:
            raise ValueError(f"pim_to_host_fixed_ns must be non-negative, got {self.pim_to_host_fixed_ns}")


@dataclass
class PolicyConfig:
    reuse_threshold: float = 0.5  # Reusability ratio threshold
    reuse_penalty_weight: float = 1.0  # lambda: weight for reuse-preservation cost in ns

    def validate(self) -> None:
        if not (0.0 <= self.reuse_threshold <= 1.0):
            raise ValueError(f"reuse_threshold must be in [0.0, 1.0], got {self.reuse_threshold}")
        if self.reuse_penalty_weight < 0.0:
            raise ValueError(f"reuse_penalty_weight must be non-negative, got {self.reuse_penalty_weight}")


@dataclass
class ExperimentConfig:
    master_seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))
    dev_seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    held_out_seeds: List[int] = field(default_factory=lambda: [47, 48, 49, 50, 51])
    traces_per_seed: int = 5
    default_workload_patterns: List[str] = field(
        default_factory=lambda: ["STREAM", "REUSE", "RANDOM", "MIXED_PHASE"]
    )

    def validate(self) -> None:
        if not self.master_seeds:
            raise ValueError("master_seeds cannot be empty")
        if self.traces_per_seed <= 0:
            raise ValueError(f"traces_per_seed must be positive, got {self.traces_per_seed}")
        # Verify disjoint dev and test seeds
        dev_set = set(self.dev_seeds)
        test_set = set(self.held_out_seeds)
        overlap = dev_set.intersection(test_set)
        if overlap:
            raise ValueError(f"dev_seeds and held_out_seeds must be strictly disjoint, overlap: {overlap}")


@dataclass
class CachePIMConfig:
    """Central configuration container for the CachePIM simulator."""
    cache: CacheConfig = field(default_factory=CacheConfig)
    cpu: CPUConfig = field(default_factory=CPUConfig)
    pim: PIMConfig = field(default_factory=PIMConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    version: str = "1.0.0"

    def validate(self) -> None:
        self.cache.validate()
        self.cpu.validate()
        self.pim.validate()
        self.transfer.validate()
        self.policy.validate()
        self.experiment.validate()

    def get_hash(self) -> str:
        """Deterministic SHA-256 hash representing configuration state."""
        dict_rep = asdict(self)
        json_str = json.dumps(dict_rep, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CachePIMConfig:
        return cls(
            cache=CacheConfig(**data.get("cache", {})),
            cpu=CPUConfig(**data.get("cpu", {})),
            pim=PIMConfig(**data.get("pim", {})),
            transfer=TransferConfig(**data.get("transfer", {})),
            policy=PolicyConfig(**data.get("policy", {})),
            experiment=ExperimentConfig(**data.get("experiment", {})),
            version=data.get("version", "1.0.0"),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> CachePIMConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg

    def to_yaml(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_parameter_provenance() -> Dict[str, ParameterProvenance]:
    """Audit catalogue mapping configuration parameters to units, source provenance, and descriptions."""
    return {
        "cache.capacity_bytes": ParameterProvenance(
            value=32768,
            unit="bytes",
            provenance="assumed",
            description="LLC capacity modeled for sensitivity sweep; 32 KiB default baseline.",
            reference="Spec Sec 6 (sweep 16-512 KiB)"
        ),
        "cache.line_size_bytes": ParameterProvenance(
            value=64,
            unit="bytes",
            provenance="literature-derived",
            description="Standard modern x86-64 / ARMv8 cache line width.",
            reference="Hennessy & Patterson, Computer Architecture: A Quantitative Approach, 6th ed."
        ),
        "cache.replacement": ParameterProvenance(
            value="LRU",
            unit="policy",
            provenance="assumed",
            description="True Least Recently Used replacement policy.",
            reference="Spec Sec 5.1"
        ),
        "cpu.frequency_hz": ParameterProvenance(
            value=2.5e9,
            unit="Hz",
            provenance="assumed",
            description="Nominal core clock frequency (2.5 GHz).",
            reference="Spec Sec 6 illustrative default"
        ),
        "cpu.hit_service_ns": ParameterProvenance(
            value=2.0,
            unit="ns",
            provenance="literature-derived",
            description="Complete LLC hit latency (~5 clock cycles @ 2.5 GHz).",
            reference="Intel 64 and IA-32 Architectures Optimization Reference Manual"
        ),
        "cpu.miss_service_ns": ParameterProvenance(
            value=80.0,
            unit="ns",
            provenance="literature-derived",
            description="Complete main memory read service latency (~200 cycles @ 2.5 GHz).",
            reference="Micron DDR4 SDRAM Timing Specifications; JEDEC standard"
        ),
        "cpu.compute_ns_per_op": ParameterProvenance(
            value=0.4,
            unit="ns/op",
            provenance="assumed",
            description="Throughput of 1 basic arithmetic operation per cycle at 2.5 GHz.",
            reference="Spec Sec 6 illustrative default"
        ),
        "pim.bank_count": ParameterProvenance(
            value=16,
            unit="banks",
            provenance="literature-derived",
            description="Number of parallel PIM memory banks/processing units.",
            reference="Gao et al., 'TETRIS: Scalable and Efficient Neural Network Acceleration with 3D Memory', ASPLOS 2017; UPMEM DPU architecture"
        ),
        "pim.effective_bandwidth_bytes_per_ns": ParameterProvenance(
            value=128.0,
            unit="bytes/ns (GB/s)",
            provenance="literature-derived",
            description="Internal aggregate bandwidth across all PIM banks.",
            reference="HBM2/HBM3 internal crossbar bandwidth estimates"
        ),
        "pim.compute_ops_per_ns_per_bank": ParameterProvenance(
            value=1.0,
            unit="ops/ns/bank",
            provenance="assumed",
            description="PIM bank compute throughput (~1 GHz basic SIMD/ALU unit per bank).",
            reference="UPMEM DPU / Samsung Aquabolt-XL specification"
        ),
        "pim.launch_overhead_ns": ParameterProvenance(
            value=50.0,
            unit="ns",
            provenance="assumed",
            description="Command dispatch and PIM kernel invocation overhead.",
            reference="Spec Sec 6 illustrative default"
        ),
        "pim.sync_overhead_ns": ParameterProvenance(
            value=20.0,
            unit="ns",
            provenance="assumed",
            description="Barrier synchronization among PIM banks upon completion.",
            reference="Spec Sec 6 illustrative default"
        ),
        "transfer.host_to_pim_bandwidth_bytes_per_ns": ParameterProvenance(
            value=16.0,
            unit="bytes/ns (GB/s)",
            provenance="literature-derived",
            description="Interconnect link bandwidth (PCIe Gen4 x8 or CXL 1.1 x8).",
            reference="PCI-SIG PCIe 4.0 Specification; CXL Consortium Specification 1.1"
        ),
        "transfer.pim_to_host_bandwidth_bytes_per_ns": ParameterProvenance(
            value=16.0,
            unit="bytes/ns (GB/s)",
            provenance="literature-derived",
            description="Interconnect link return bandwidth.",
            reference="PCI-SIG PCIe 4.0 Specification"
        ),
        "transfer.host_to_pim_fixed_ns": ParameterProvenance(
            value=100.0,
            unit="ns",
            provenance="literature-derived",
            description="Fixed link traversal and packet serialization overhead.",
            reference="CXL Host-Device Round-Trip Latency benchmarks"
        ),
        "transfer.pim_to_host_fixed_ns": ParameterProvenance(
            value=100.0,
            unit="ns",
            provenance="literature-derived",
            description="Fixed link return traversal overhead.",
            reference="CXL Host-Device Round-Trip Latency benchmarks"
        ),
        "policy.reuse_penalty_weight": ParameterProvenance(
            value=1.0,
            unit="dimensionless factor",
            provenance="artifact-derived",
            description="Tuning weight lambda applied to reuse preservation cost in ns.",
            reference="Spec Sec 11.4; calibrated on development seeds 42-46"
        ),
    }
