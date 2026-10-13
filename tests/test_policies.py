"""Tests for execution policies, regret invariants, and fallback behavior."""

import pytest
from cachepim.config import CachePIMConfig
from cachepim.policies import (
    CostBasedPolicy,
    CPUCachePolicy,
    CPUNoCachePolicy,
    OraclePolicy,
    PIMAlwaysPolicy,
    ReuseAwarePolicy,
)
from cachepim.types import ExecutionPath, KernelType, WorkloadDescriptor


@pytest.fixture
def base_config():
    return CachePIMConfig()


def test_pim_always_fallback_for_unsupported_kernel(base_config):
    desc = WorkloadDescriptor(
        workload_type="STREAM",
        seed=42,
        kernel_type=KernelType.UNSUPPORTED_KERNEL.value,
        element_count=100,
        bytes_read=100,
        bytes_written=0,
        operation_count=100,
        read_write_ratio=1.0,
        working_set_bytes=100,
        trace_length=100,
        pim_eligible=False,
    )
    policy = PIMAlwaysPolicy()
    decision = policy.decide(desc, base_config)

    assert decision.selected_path == ExecutionPath.CPU.value
    assert decision.decision_reason == "UNSUPPORTED_BY_PIM"


def test_oracle_consistency(base_config):
    desc = WorkloadDescriptor(
        workload_type="STREAM",
        seed=42,
        kernel_type=KernelType.VECTOR_OP.value,
        element_count=100,
        bytes_read=100,
        bytes_written=0,
        operation_count=100,
        read_write_ratio=1.0,
        working_set_bytes=100,
        trace_length=100,
        pim_eligible=True,
    )
    policy = OraclePolicy()

    # Case 1: CPU faster
    dec1 = policy.decide(desc, base_config, actual_cpu_ns=200.0, actual_pim_ns=500.0)
    assert dec1.selected_path == ExecutionPath.CPU.value
    assert dec1.decision_reason == "ORACLE_CPU_OPTIMAL"

    # Case 2: PIM faster
    dec2 = policy.decide(desc, base_config, actual_cpu_ns=600.0, actual_pim_ns=300.0)
    assert dec2.selected_path == ExecutionPath.PIM.value
    assert dec2.decision_reason == "ORACLE_PIM_OPTIMAL"


def test_reuse_aware_policy_dimensional_consistency(base_config):
    desc = WorkloadDescriptor(
        workload_type="REUSE",
        seed=42,
        kernel_type=KernelType.VECTOR_OP.value,
        element_count=1000,
        bytes_read=8000,
        bytes_written=0,
        operation_count=1000,
        read_write_ratio=1.0,
        working_set_bytes=16384,  # Fits in 32 KiB cache
        trace_length=1000,
        pim_eligible=True,
        reusable_bytes=16384,
    )
    policy = ReuseAwarePolicy()
    r_cost = policy.calculate_reuse_cost(desc, base_config, estimated_hits=500)

    # R_cost must be non-negative and in nanoseconds
    assert r_cost >= 0.0
    assert isinstance(r_cost, float)

    # When lambda = 0 (ablation), decision should match cost-based
    config_lam0 = CachePIMConfig()
    config_lam0.policy.reuse_penalty_weight = 0.0
    dec_lam0 = policy.decide(desc, config_lam0)
    cost_pol = CostBasedPolicy()
    dec_cost = cost_pol.decide(desc, config_lam0)

    assert dec_lam0.selected_path == dec_cost.selected_path
