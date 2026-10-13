"""Tests for CacheModel verifying LRU correctness, capacity invariants, and reset behavior."""

import pytest
from cachepim.cache_model import CacheModel
from cachepim.config import CacheConfig
from cachepim.types import TraceRecord


def test_empty_cache_stats():
    config = CacheConfig(capacity_bytes=256, line_size_bytes=64)
    model = CacheModel(config)
    stats = model.get_stats()

    assert stats.accesses == 0
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.evictions == 0
    assert stats.hit_rate == 0.0
    assert stats.miss_rate == 0.0
    assert stats.unique_lines == 0
    assert stats.bytes_fetched_from_memory == 0


def test_hand_calculated_lru_trace():
    """Verify exact hand-calculated sequence on a 3-line cache (192 bytes, 64B lines).

    Line size = 64 bytes.
    Line 0: address 0
    Line 1: address 64
    Line 2: address 128
    Line 3: address 192

    Sequence:
    1. Access 0 (Line 0): MISS, cache=[0]
    2. Access 64 (Line 1): MISS, cache=[0, 1]
    3. Access 128 (Line 2): MISS, cache=[0, 1, 2] (Cache now full)
    4. Access 0 (Line 0): HIT, cache=[1, 2, 0] (Line 0 moved to MRU)
    5. Access 192 (Line 3): MISS, evicts Line 1 (LRU), cache=[2, 0, 3]
    6. Access 64 (Line 1): MISS, evicts Line 2 (LRU), cache=[0, 3, 1]
    7. Access 0 (Line 0): HIT, cache=[3, 1, 0]
    """
    config = CacheConfig(capacity_bytes=192, line_size_bytes=64)
    assert config.number_of_lines == 3

    model = CacheModel(config)

    # 1. Access 0
    e1 = model.access(0, "READ")
    assert e1.miss and not e1.hit
    assert e1.line_id == 0
    assert e1.evicted_line_id is None

    # 2. Access 64
    e2 = model.access(64, "READ")
    assert e2.miss and not e2.hit
    assert e2.line_id == 1
    assert e2.evicted_line_id is None

    # 3. Access 128
    e3 = model.access(128, "READ")
    assert e3.miss and not e3.hit
    assert e3.line_id == 2
    assert e3.evicted_line_id is None

    # 4. Access 0 (hit, promotes line 0 to MRU)
    e4 = model.access(0, "READ")
    assert e4.hit and not e4.miss
    assert e4.line_id == 0
    assert e4.evicted_line_id is None

    # 5. Access 192 (miss, evicts LRU which is line 1)
    e5 = model.access(192, "READ")
    assert e5.miss and not e5.hit
    assert e5.line_id == 3
    assert e5.evicted_line_id == 1

    # 6. Access 64 (miss, evicts LRU which is line 2)
    e6 = model.access(64, "READ")
    assert e6.miss and not e6.hit
    assert e6.line_id == 1
    assert e6.evicted_line_id == 2

    # 7. Access 0 (hit)
    e7 = model.access(0, "READ")
    assert e7.hit and not e7.miss
    assert e7.line_id == 0
    assert e7.evicted_line_id is None

    stats = model.get_stats()
    assert stats.accesses == 7
    assert stats.hits == 2
    assert stats.misses == 5
    assert stats.evictions == 2
    assert pytest.approx(stats.hit_rate) == 2 / 7
    assert pytest.approx(stats.miss_rate) == 5 / 7
    assert pytest.approx(stats.hit_rate + stats.miss_rate) == 1.0
    assert stats.bytes_fetched_from_memory == 5 * 64


def test_capacity_invariant():
    """Verify resident line count never exceeds configured capacity."""
    config = CacheConfig(capacity_bytes=256, line_size_bytes=64)  # 4 lines
    model = CacheModel(config)

    # Access 100 distinct lines
    for i in range(100):
        addr = i * 64
        model.access(addr, "READ")
        snap = model.snapshot()
        assert snap["resident_count"] <= 4

    stats = model.get_stats()
    assert stats.accesses == 100
    assert stats.misses == 100
    assert stats.evictions == 96
    assert stats.unique_lines == 100


def test_reset_behavior():
    config = CacheConfig(capacity_bytes=128, line_size_bytes=64)
    model = CacheModel(config)

    trace = [
        TraceRecord(sequence_number=1, address=0, operation="READ", task_id="t", phase_id="p"),
        TraceRecord(sequence_number=2, address=64, operation="READ", task_id="t", phase_id="p"),
        TraceRecord(sequence_number=3, address=0, operation="READ", task_id="t", phase_id="p"),
    ]

    stats1 = model.run(trace)
    assert stats1.hits == 1
    assert stats1.misses == 2

    model.reset()
    assert model.get_stats().accesses == 0
    assert model.snapshot()["resident_count"] == 0

    stats2 = model.run(trace)
    assert stats2.hits == stats1.hits
    assert stats2.misses == stats1.misses
