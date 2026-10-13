"""Fully associative Last-Level Cache (LLC) simulator with exact LRU replacement.

Models line-level mapping, cache capacity constraints, LRU eviction order,
hit/miss/eviction accounting, and memory fetch bytes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from cachepim.config import CacheConfig
from cachepim.types import CacheEvent, CacheStats, OperationType, TraceRecord


@dataclass
class CacheLineEntry:
    line_id: int
    is_dirty: bool = False
    task_id: Optional[str] = None


class CacheModel:
    """Fully associative cache model with exact Least-Recently-Used (LRU) replacement policy."""

    def __init__(self, config: CacheConfig):
        config.validate()
        self.config = config
        self.capacity_bytes: int = config.capacity_bytes
        self.line_size_bytes: int = config.line_size_bytes
        self.number_of_lines: int = config.number_of_lines

        # State storage: OrderedDict maintains access recency (front = LRU, end = MRU)
        self._lines: OrderedDict[int, CacheLineEntry] = OrderedDict()
        self._unique_lines_seen: Set[int] = set()

        # Running statistics
        self._seq: int = 0
        self._accesses: int = 0
        self._reads: int = 0
        self._writes: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._dirty_evictions: int = 0

    def reset(self) -> None:
        """Reset cache to cold initial state and clear all counters."""
        self._lines.clear()
        self._unique_lines_seen.clear()
        self._seq = 0
        self._accesses = 0
        self._reads = 0
        self._writes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._dirty_evictions = 0

    def address_to_line_id(self, address: int) -> int:
        """Convert a byte address to an aligned cache line identifier."""
        return address // self.line_size_bytes

    def access(
        self,
        address: int,
        operation: str = "READ",
        task_id: Optional[str] = None,
    ) -> CacheEvent:
        """Process a single memory access and update LRU recency and statistics."""
        self._seq += 1
        seq = self._seq
        line_id = self.address_to_line_id(address)
        self._unique_lines_seen.add(line_id)
        self._accesses += 1

        is_write = operation.upper() == OperationType.WRITE.value
        if is_write:
            self._writes += 1
        else:
            self._reads += 1

        evicted_line_id: Optional[int] = None

        if line_id in self._lines:
            # Cache HIT: mark line as Most Recently Used (move to end)
            entry = self._lines[line_id]
            if is_write:
                entry.is_dirty = True
            if task_id:
                entry.task_id = task_id
            self._lines.move_to_end(line_id, last=True)
            self._hits += 1

            return CacheEvent(
                line_id=line_id,
                hit=True,
                miss=False,
                evicted_line_id=None,
                operation=operation,
                sequence_number=seq,
            )
        else:
            # Cache MISS
            self._misses += 1

            # Check capacity; if full, evict the least recently used line (front of OrderedDict)
            if len(self._lines) >= self.number_of_lines:
                evicted_id, evicted_entry = self._lines.popitem(last=False)
                evicted_line_id = evicted_id
                self._evictions += 1
                if evicted_entry.is_dirty:
                    self._dirty_evictions += 1

            # Insert new line as MRU
            new_entry = CacheLineEntry(line_id=line_id, is_dirty=is_write, task_id=task_id)
            self._lines[line_id] = new_entry

            # Enforce capacity invariant
            assert len(self._lines) <= self.number_of_lines, (
                f"Cache invariant violated: resident lines ({len(self._lines)}) > capacity ({self.number_of_lines})"
            )

            return CacheEvent(
                line_id=line_id,
                hit=False,
                miss=True,
                evicted_line_id=evicted_line_id,
                operation=operation,
                sequence_number=seq,
            )

    def run(self, trace: Iterable[TraceRecord]) -> CacheStats:
        """Process an entire stream of trace records and return aggregated CacheStats."""
        for rec in trace:
            self.access(rec.address, rec.operation, rec.task_id)
        return self.get_stats()

    def get_stats(self) -> CacheStats:
        """Return the current cumulative cache statistics."""
        if self._accesses > 0:
            hit_rate = self._hits / self._accesses
            miss_rate = self._misses / self._accesses
        else:
            hit_rate = 0.0
            miss_rate = 0.0

        return CacheStats(
            accesses=self._accesses,
            reads=self._reads,
            writes=self._writes,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            unique_lines=len(self._unique_lines_seen),
            bytes_fetched_from_memory=self._misses * self.line_size_bytes,
            dirty_evictions=self._dirty_evictions,
        )

    def snapshot(self) -> Dict[str, Any]:
        """Capture diagnostic snapshot of cache residency and recency order."""
        return {
            "capacity_bytes": self.capacity_bytes,
            "line_size_bytes": self.line_size_bytes,
            "number_of_lines": self.number_of_lines,
            "resident_count": len(self._lines),
            "resident_line_ids_lru_order": list(self._lines.keys()),
            "stats": self.get_stats(),
        }
