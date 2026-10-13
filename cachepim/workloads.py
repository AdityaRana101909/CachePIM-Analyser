"""Workload and trace generation for CachePIM.

Provides deterministic, reproducible synthetic memory-access generators for:
1. STREAM: Sequential streaming over contiguous memory arrays.
2. REUSE: Hot-set access with controlled temporal reuse and cold background stream.
3. RANDOM: Uniform or skewed random sampling over a target working set.
4. MIXED_PHASE: Concatenated execution phases with differing working sets and locality.

Also defines computational kernel descriptors: Vector Operations, Reductions,
Streaming Filters, and unsupported kernels for fallback testing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cachepim.types import (
    KernelType,
    OperationType,
    TraceRecord,
    WorkloadDescriptor,
    WorkloadPattern,
)


class WorkloadGenerator:
    """Generates synthetic memory access traces and metadata deterministically."""

    def __init__(self, line_size_bytes: int = 64):
        self.line_size_bytes = line_size_bytes

    def generate(
        self,
        pattern: str = "STREAM",
        seed: int = 42,
        working_set_bytes: int = 32768,
        access_count: int = 2048,
        kernel_type: str = "VECTOR_OP",
        read_write_ratio: float = 1.0,  # 1.0 means all READ, 0.0 means all WRITE, 0.5 balanced
        hot_cold_ratio: float = 0.8,
        hot_set_fraction: float = 0.2,
        passes: int = 2,
        pim_eligible: Optional[bool] = None,
        task_id: str = "task_0",
        phase_mix: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[TraceRecord], WorkloadDescriptor]:
        """Generate a trace and descriptor for the specified workload pattern."""
        rng = np.random.default_rng(seed)

        # Check PIM eligibility: kernels marked UNSUPPORTED_KERNEL are not PIM eligible
        if pim_eligible is None:
            pim_eligible = (kernel_type != KernelType.UNSUPPORTED_KERNEL.value)

        pattern_upper = pattern.upper()

        if pattern_upper == WorkloadPattern.STREAM.value:
            records, descriptor = self._generate_stream(
                rng=rng,
                seed=seed,
                working_set_bytes=working_set_bytes,
                passes=passes,
                kernel_type=kernel_type,
                read_write_ratio=read_write_ratio,
                pim_eligible=pim_eligible,
                task_id=task_id,
            )
        elif pattern_upper == WorkloadPattern.REUSE.value:
            records, descriptor = self._generate_reuse(
                rng=rng,
                seed=seed,
                working_set_bytes=working_set_bytes,
                access_count=access_count,
                hot_cold_ratio=hot_cold_ratio,
                hot_set_fraction=hot_set_fraction,
                kernel_type=kernel_type,
                read_write_ratio=read_write_ratio,
                pim_eligible=pim_eligible,
                task_id=task_id,
            )
        elif pattern_upper == WorkloadPattern.RANDOM.value:
            records, descriptor = self._generate_random(
                rng=rng,
                seed=seed,
                working_set_bytes=working_set_bytes,
                access_count=access_count,
                kernel_type=kernel_type,
                read_write_ratio=read_write_ratio,
                pim_eligible=pim_eligible,
                task_id=task_id,
            )
        elif pattern_upper == WorkloadPattern.MIXED_PHASE.value:
            records, descriptor = self._generate_mixed_phase(
                rng=rng,
                seed=seed,
                working_set_bytes=working_set_bytes,
                access_count=access_count,
                kernel_type=kernel_type,
                read_write_ratio=read_write_ratio,
                pim_eligible=pim_eligible,
                task_id=task_id,
                phase_mix=phase_mix,
            )
        else:
            raise ValueError(f"Unknown workload pattern: {pattern}")

        return records, descriptor

    def _determine_op(self, rng: np.random.Generator, read_write_ratio: float) -> str:
        """Helper to sample READ or WRITE based on read_write_ratio."""
        if read_write_ratio >= 1.0:
            return OperationType.READ.value
        if read_write_ratio <= 0.0:
            return OperationType.WRITE.value
        return OperationType.READ.value if rng.random() < read_write_ratio else OperationType.WRITE.value

    def _generate_stream(
        self,
        rng: np.random.Generator,
        seed: int,
        working_set_bytes: int,
        passes: int,
        kernel_type: str,
        read_write_ratio: float,
        pim_eligible: bool,
        task_id: str,
    ) -> Tuple[List[TraceRecord], WorkloadDescriptor]:
        """Contiguous streaming trace over working_set_bytes for multiple passes."""
        element_size = 8  # 8 bytes per double / 64-bit word
        num_elements = max(1, working_set_bytes // element_size)
        base_address = 0x10000000

        records: List[TraceRecord] = []
        seq = 0
        phase_id = "phase_stream"

        for p in range(passes):
            for i in range(num_elements):
                seq += 1
                addr = base_address + (i * element_size)
                op = self._determine_op(rng, read_write_ratio)
                records.append(
                    TraceRecord(
                        sequence_number=seq,
                        address=addr,
                        operation=op,
                        task_id=task_id,
                        phase_id=f"{phase_id}_{p}",
                    )
                )

        bytes_read = sum(element_size for r in records if r.operation == OperationType.READ.value)
        bytes_written = sum(element_size for r in records if r.operation == OperationType.WRITE.value)
        op_count = len(records)
        # In multi-pass streaming, passes > 1 offer temporal reuse across passes
        reusable_bytes = working_set_bytes if passes > 1 else 0

        desc = WorkloadDescriptor(
            workload_type=WorkloadPattern.STREAM.value,
            seed=seed,
            kernel_type=kernel_type,
            element_count=num_elements,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            operation_count=op_count,
            read_write_ratio=read_write_ratio,
            working_set_bytes=working_set_bytes,
            trace_length=len(records),
            pim_eligible=pim_eligible,
            reusable_bytes=reusable_bytes,
            params={
                "pattern": "STREAM",
                "seed": seed,
                "working_set_bytes": working_set_bytes,
                "passes": passes,
                "kernel_type": kernel_type,
                "read_write_ratio": read_write_ratio,
            },
        )
        return records, desc

    def _generate_reuse(
        self,
        rng: np.random.Generator,
        seed: int,
        working_set_bytes: int,
        access_count: int,
        hot_cold_ratio: float,
        hot_set_fraction: float,
        kernel_type: str,
        read_write_ratio: float,
        pim_eligible: bool,
        task_id: str,
    ) -> Tuple[List[TraceRecord], WorkloadDescriptor]:
        """Hot-set reuse trace where hot_cold_ratio of accesses target hot_set_fraction of lines."""
        element_size = 8
        total_lines = max(2, working_set_bytes // self.line_size_bytes)
        hot_lines = max(1, int(total_lines * hot_set_fraction))
        cold_lines = max(1, total_lines - hot_lines)

        base_hot_addr = 0x20000000
        base_cold_addr = base_hot_addr + (hot_lines * self.line_size_bytes)

        records: List[TraceRecord] = []
        phase_id = "phase_reuse"

        for seq in range(1, access_count + 1):
            is_hot = rng.random() < hot_cold_ratio
            if is_hot:
                line_idx = rng.integers(0, hot_lines)
                offset = int(rng.integers(0, self.line_size_bytes // element_size)) * element_size
                addr = base_hot_addr + (int(line_idx) * self.line_size_bytes) + offset
            else:
                line_idx = rng.integers(0, cold_lines)
                offset = int(rng.integers(0, self.line_size_bytes // element_size)) * element_size
                addr = base_cold_addr + (int(line_idx) * self.line_size_bytes) + offset

            op = self._determine_op(rng, read_write_ratio)
            records.append(
                TraceRecord(
                    sequence_number=seq,
                    address=addr,
                    operation=op,
                    task_id=task_id,
                    phase_id=phase_id,
                )
            )

        bytes_read = sum(element_size for r in records if r.operation == OperationType.READ.value)
        bytes_written = sum(element_size for r in records if r.operation == OperationType.WRITE.value)
        reusable_bytes = hot_lines * self.line_size_bytes

        desc = WorkloadDescriptor(
            workload_type=WorkloadPattern.REUSE.value,
            seed=seed,
            kernel_type=kernel_type,
            element_count=working_set_bytes // element_size,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            operation_count=access_count,
            read_write_ratio=read_write_ratio,
            working_set_bytes=working_set_bytes,
            trace_length=len(records),
            pim_eligible=pim_eligible,
            reusable_bytes=reusable_bytes,
            hot_cold_ratio=hot_cold_ratio,
            params={
                "pattern": "REUSE",
                "seed": seed,
                "working_set_bytes": working_set_bytes,
                "access_count": access_count,
                "hot_cold_ratio": hot_cold_ratio,
                "hot_set_fraction": hot_set_fraction,
                "kernel_type": kernel_type,
                "read_write_ratio": read_write_ratio,
            },
        )
        return records, desc

    def _generate_random(
        self,
        rng: np.random.Generator,
        seed: int,
        working_set_bytes: int,
        access_count: int,
        kernel_type: str,
        read_write_ratio: float,
        pim_eligible: bool,
        task_id: str,
    ) -> Tuple[List[TraceRecord], WorkloadDescriptor]:
        """Uniform random sampling across the working set."""
        element_size = 8
        total_lines = max(1, working_set_bytes // self.line_size_bytes)
        base_addr = 0x30000000

        records: List[TraceRecord] = []
        phase_id = "phase_random"

        for seq in range(1, access_count + 1):
            line_idx = rng.integers(0, total_lines)
            offset = int(rng.integers(0, self.line_size_bytes // element_size)) * element_size
            addr = base_addr + (int(line_idx) * self.line_size_bytes) + offset
            op = self._determine_op(rng, read_write_ratio)
            records.append(
                TraceRecord(
                    sequence_number=seq,
                    address=addr,
                    operation=op,
                    task_id=task_id,
                    phase_id=phase_id,
                )
            )

        bytes_read = sum(element_size for r in records if r.operation == OperationType.READ.value)
        bytes_written = sum(element_size for r in records if r.operation == OperationType.WRITE.value)

        desc = WorkloadDescriptor(
            workload_type=WorkloadPattern.RANDOM.value,
            seed=seed,
            kernel_type=kernel_type,
            element_count=working_set_bytes // element_size,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            operation_count=access_count,
            read_write_ratio=read_write_ratio,
            working_set_bytes=working_set_bytes,
            trace_length=len(records),
            pim_eligible=pim_eligible,
            reusable_bytes=0,  # Negligible predictable reuse
            params={
                "pattern": "RANDOM",
                "seed": seed,
                "working_set_bytes": working_set_bytes,
                "access_count": access_count,
                "kernel_type": kernel_type,
                "read_write_ratio": read_write_ratio,
            },
        )
        return records, desc

    def _generate_mixed_phase(
        self,
        rng: np.random.Generator,
        seed: int,
        working_set_bytes: int,
        access_count: int,
        kernel_type: str,
        read_write_ratio: float,
        pim_eligible: bool,
        task_id: str,
        phase_mix: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[TraceRecord], WorkloadDescriptor]:
        """Concatenate alternating streaming and high-reuse phases."""
        phase_count = 2
        per_phase_accesses = max(16, access_count // phase_count)
        element_size = 8
        base_addr = 0x40000000

        records: List[TraceRecord] = []
        seq = 0

        # Phase 1: High Reuse Phase (Hot-set)
        hot_lines = max(1, (working_set_bytes // 2) // self.line_size_bytes)
        for i in range(per_phase_accesses):
            seq += 1
            line_idx = rng.integers(0, hot_lines)
            offset = int(rng.integers(0, self.line_size_bytes // element_size)) * element_size
            addr = base_addr + (int(line_idx) * self.line_size_bytes) + offset
            op = self._determine_op(rng, read_write_ratio)
            records.append(
                TraceRecord(
                    sequence_number=seq,
                    address=addr,
                    operation=op,
                    task_id=task_id,
                    phase_id="phase_reuse_0",
                )
            )

        # Phase 2: Sequential Streaming Phase
        stream_lines = max(1, (working_set_bytes // 2) // self.line_size_bytes)
        stream_base = base_addr + (hot_lines * self.line_size_bytes)
        for i in range(per_phase_accesses):
            seq += 1
            line_idx = i % stream_lines
            offset = int(rng.integers(0, self.line_size_bytes // element_size)) * element_size
            addr = stream_base + (line_idx * self.line_size_bytes) + offset
            op = self._determine_op(rng, read_write_ratio)
            records.append(
                TraceRecord(
                    sequence_number=seq,
                    address=addr,
                    operation=op,
                    task_id=task_id,
                    phase_id="phase_stream_1",
                )
            )

        bytes_read = sum(element_size for r in records if r.operation == OperationType.READ.value)
        bytes_written = sum(element_size for r in records if r.operation == OperationType.WRITE.value)
        reusable_bytes = hot_lines * self.line_size_bytes

        desc = WorkloadDescriptor(
            workload_type=WorkloadPattern.MIXED_PHASE.value,
            seed=seed,
            kernel_type=kernel_type,
            element_count=working_set_bytes // element_size,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            operation_count=len(records),
            read_write_ratio=read_write_ratio,
            working_set_bytes=working_set_bytes,
            trace_length=len(records),
            pim_eligible=pim_eligible,
            reusable_bytes=reusable_bytes,
            phase_mix={"phases": ["REUSE", "STREAM"], "split": 0.5},
            params={
                "pattern": "MIXED_PHASE",
                "seed": seed,
                "working_set_bytes": working_set_bytes,
                "access_count": len(records),
                "kernel_type": kernel_type,
                "read_write_ratio": read_write_ratio,
            },
        )
        return records, desc
