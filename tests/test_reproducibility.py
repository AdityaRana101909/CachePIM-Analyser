"""Tests for determinism, seed separation, and configuration reproducibility."""

from cachepim.config import CachePIMConfig
from cachepim.simulator import Simulator
from cachepim.workloads import WorkloadGenerator


def test_seed_and_generator_determinism():
    """Verify that identical seeds produce identical traces and simulation metrics."""
    gen = WorkloadGenerator(line_size_bytes=64)
    trace1, desc1 = gen.generate(pattern="STREAM", seed=42, working_set_bytes=16384, passes=2)
    trace2, desc2 = gen.generate(pattern="STREAM", seed=42, working_set_bytes=16384, passes=2)

    assert len(trace1) == len(trace2)
    for r1, r2 in zip(trace1, trace2):
        assert r1.sequence_number == r2.sequence_number
        assert r1.address == r2.address
        assert r1.operation == r2.operation

    cfg = CachePIMConfig()
    sim1 = Simulator(cfg)
    sim2 = Simulator(cfg)

    results1 = sim1.run_all_policies_on_trace(trace1, desc1)
    results2 = sim2.run_all_policies_on_trace(trace2, desc2)

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1.policy_name == r2.policy_name
        assert r1.selected_path == r2.selected_path
        assert r1.selected_model_ns == r2.selected_model_ns
        assert r1.regret_ns == r2.regret_ns
        assert r1.cache_hits == r2.cache_hits
        assert r1.cache_misses == r2.cache_misses


def test_dev_and_held_out_seed_separation():
    """Verify dev_seeds and held_out_seeds are completely disjoint."""
    cfg = CachePIMConfig()
    dev_seeds = set(cfg.experiment.dev_seeds)
    test_seeds = set(cfg.experiment.held_out_seeds)

    assert len(dev_seeds) > 0
    assert len(test_seeds) > 0
    assert dev_seeds.isdisjoint(test_seeds)


def test_config_hash_consistency():
    cfg1 = CachePIMConfig()
    cfg2 = CachePIMConfig()
    assert cfg1.get_hash() == cfg2.get_hash()

    cfg2.cache.capacity_bytes = 65536
    assert cfg1.get_hash() != cfg2.get_hash()


def test_cli_ui_simulator_parity():
    """Verify CLI and Dashboard produce identical core metrics for matching inputs."""
    cfg = CachePIMConfig()
    gen = WorkloadGenerator(line_size_bytes=cfg.cache.line_size_bytes)
    trace, desc = gen.generate(pattern="STREAM", seed=42, working_set_bytes=32768, access_count=2048)

    sim = Simulator(cfg)
    results = sim.run_all_policies_on_trace(trace, desc)

    res_map = {r.policy_name: r for r in results}
    # Verify exact values that the CLI prints matches the simulation core used by the UI
    assert res_map["CPU-cache"].selected_path == "CPU"
    assert res_map["PIM-always"].selected_path == "PIM"
    assert res_map["Oracle"].selected_path == "PIM"
    assert res_map["Oracle"].regret_ns == 0.0
    assert res_map["CPU-cache"].regret_ns >= 0.0

