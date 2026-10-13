"""Processing-in-Memory (PIM) analytical cost model for CachePIM.

Calculates execution latency using analytical timing components:
T_PIM = T_H2P + T_compute_PIM + T_P2H + T_launch + T_sync + T_imbalance

Models host-to-PIM / PIM-to-host data transfers, bank parallelism,
deterministic bank load distribution, and kernel launch/sync overheads.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from cachepim.config import PIMConfig, TransferConfig
from cachepim.types import PIMCostBreakdown, WorkloadDescriptor


class PIMModel:
    """Analytical PIM execution and data movement model."""

    def __init__(self, pim_config: PIMConfig, transfer_config: TransferConfig):
        pim_config.validate()
        transfer_config.validate()
        self.pim_config = pim_config
        self.transfer_config = transfer_config

        self.bank_count: int = pim_config.bank_count
        self.compute_ops_per_ns_per_bank: float = pim_config.compute_ops_per_ns_per_bank
        self.launch_overhead_ns: float = pim_config.launch_overhead_ns
        self.sync_overhead_ns: float = pim_config.sync_overhead_ns
        self.imbalance_model: str = pim_config.bank_imbalance_model

        self.h2p_bandwidth: float = transfer_config.host_to_pim_bandwidth_bytes_per_ns
        self.p2h_bandwidth: float = transfer_config.pim_to_host_bandwidth_bytes_per_ns
        self.h2p_fixed_ns: float = transfer_config.host_to_pim_fixed_ns
        self.p2h_fixed_ns: float = transfer_config.pim_to_host_fixed_ns

    def distribute_work(
        self,
        operation_count: int,
        skew_factor: float = 0.0,
    ) -> Tuple[List[int], float]:
        """Deterministically distribute operations across banks and calculate imbalance factor.

        imbalance_factor = max_b(W_b) / mean_b(W_b) when mean_b(W_b) > 0, else 1.0.
        """
        B = self.bank_count
        if operation_count <= 0 or B <= 0:
            return [0] * B, 1.0

        if skew_factor == 0.0:
            # Deterministic round-robin / block distribution
            base = operation_count // B
            rem = operation_count % B
            per_bank = [base + (1 if b < rem else 0) for b in range(B)]
        else:
            # Deterministic exponential skew across banks
            weights = [1.0 + (skew_factor * (b / max(1, B - 1))) for b in range(B)]
            sum_w = sum(weights)
            per_bank = [int(operation_count * (w / sum_w)) for w in weights]
            diff = operation_count - sum(per_bank)
            per_bank[0] += diff

        max_work = max(per_bank)
        mean_work = sum(per_bank) / B

        imbalance_factor = (max_work / mean_work) if mean_work > 0 else 1.0
        return per_bank, imbalance_factor

    def evaluate(
        self,
        descriptor: WorkloadDescriptor,
        input_data_resident: bool = False,
    ) -> Optional[PIMCostBreakdown]:
        """Compute PIM cost breakdown for the given workload descriptor.

        Returns None if workload is not eligible for PIM offloading.
        Input transfer is charged in full unless input_data_resident=True.
        """
        if not descriptor.pim_eligible:
            return None

        # Data transfers
        input_bytes = 0 if input_data_resident else descriptor.bytes_read
        output_bytes = descriptor.bytes_written

        t_h2p = (input_bytes / self.h2p_bandwidth) + (self.h2p_fixed_ns if input_bytes > 0 else 0.0)
        t_p2h = (output_bytes / self.p2h_bandwidth) + (self.p2h_fixed_ns if output_bytes > 0 else 0.0)

        # Work distribution across banks
        per_bank_work, imbalance_factor = self.distribute_work(descriptor.operation_count)

        # Base compute time assuming perfectly balanced banks
        t_compute_base = descriptor.operation_count / (
            self.bank_count * self.compute_ops_per_ns_per_bank
        )

        # Imbalance overhead: extra time taken by the bottleneck bank
        # t_bottleneck = max(W_b) / ops_per_ns = t_compute_base * imbalance_factor
        t_imbalance = t_compute_base * (imbalance_factor - 1.0)

        t_launch = self.launch_overhead_ns
        t_sync = self.sync_overhead_ns

        total_pim_time_ns = t_h2p + t_compute_base + t_p2h + t_launch + t_sync + t_imbalance

        return PIMCostBreakdown(
            h2p_transfer_ns=t_h2p,
            compute_ns=t_compute_base,
            p2h_transfer_ns=t_p2h,
            launch_overhead_ns=t_launch,
            sync_overhead_ns=t_sync,
            imbalance_overhead_ns=t_imbalance,
            total_pim_time_ns=total_pim_time_ns,
            bank_imbalance_factor=imbalance_factor,
            per_bank_work=per_bank_work,
        )
