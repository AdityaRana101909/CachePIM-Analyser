# CachePIM: A Statistical Simulator for Cache-Aware CPU–Processing-in-Memory Execution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Testing](https://img.shields.io/badge/pytest-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CachePIM is a reproducible, trace-driven analytical research simulator designed to investigate the architectural conditions under which computational workloads should execute on a host CPU through a capacity-limited cache versus offloading to Processing-in-Memory (PIM).

---

## 1. Research Question

> **"How do cache capacity, workload reuse, and host-to-PIM transfer overhead jointly influence the performance crossover between CPU execution and Processing-in-Memory (PIM), and can a lightweight reuse-aware policy improve offloading decisions over conventional cost-based selection?"**

CachePIM evaluates this question experimentally using deterministic trace generation, an exact LRU last-level cache model, component-level analytical latency models, and an offline oracle reference.

---

## 2. Repository Structure

```
prototype/
├── cachepim/                     # Core simulation package
│   ├── __init__.py               # Package metadata and version
│   ├── config.py                 # Configuration dataclasses, validation, and provenance
│   ├── types.py                  # Dataclasses and enums for traces, costs, and results
│   ├── workloads.py              # Trace generators (STREAM, REUSE, RANDOM, MIXED_PHASE)
│   ├── cache_model.py            # Fully associative LRU cache model
│   ├── cpu_model.py              # Analytical CPU execution cost model
│   ├── pim_model.py              # Analytical PIM bank parallelism & transfer model
│   ├── policies.py               # Six offloading policies (CPU, PIM, Cost, Reuse, Oracle)
│   ├── simulator.py              # Unified simulation engine
│   ├── metrics.py                # Statistical aggregations, seed summaries & CI
│   ├── experiments.py            # Standalone experimental sweep runner
│   ├── analysis.py               # Statistical hypothesis testing and paired t-tests
│   ├── plots.py                  # Publication-ready Matplotlib figure generation
│   └── run.py                    # Single-run CLI entry point
├── config/
│   ├── default.yaml              # Default architectural parameters
│   └── experiment.yaml           # Parameter sweep ranges and experimental settings
├── tests/
│   ├── test_cache_model.py       # Hand-verified LRU trace and capacity tests
│   ├── test_cost_models.py       # CPU and PIM cost accounting tests
│   ├── test_policies.py          # Policy fallback, regret invariants, and consistency
│   └── test_reproducibility.py   # Seed determinism, config hashing, and UI/CLI parity
├── results/
│   ├── raw/                      # Raw simulation result CSVs
│   ├── summary/                  # Seed-aggregated statistical summaries
│   ├── figures/                  # Publication figures (PNG & PDF)
│   └── manifest.json             # Reproducibility run manifest
├── app.py                        # Interactive Streamlit dashboard
├── python.bat                    # Windows wrapper for project Python environment
├── streamlit.bat                 # Windows wrapper for Streamlit dashboard
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Documentation and reproduction guide
```

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10 or later (tested on Python 3.12).
- Virtual environment support.

### Setup Instructions
```bash
# 1. Clone repository and navigate to workspace
git clone <repo-url> CachePIM
cd CachePIM

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution Commands

### A. Run an Individual Simulation via CLI
Simulate a specific workload pattern under a deterministic seed:
```bash
python -m cachepim.run --config config/default.yaml --workload STREAM --seed 42
```
Optional arguments:
- `--working-set-kb <int>`: Working set in KiB (default: 32).
- `--access-count <int>`: Number of memory accesses (default: 2048).
- `--output <path>`: Optional path to save CSV results.

### B. Run the Full Experimental Suite
Execute all parameter sweeps, baseline comparisons across master seeds 42–51, ablation studies, and held-out test evaluations:
```bash
python -m cachepim.experiments --config config/experiment.yaml
```

### C. Generate Publication Figures
Generate all 8 publication figures directly from executed simulation results:
```bash
python -c "from cachepim.plots import generate_all_plots; generate_all_plots()"
```

### D. Launch the Interactive Dashboard
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
*(On Windows with the provided wrapper: `.\streamlit.bat run app.py`)*

### E. Run the Correctness Test Suite
```bash
pytest -v tests/
```

---

## 5. Architectural Models & Mathematical Formulation

### A. CPU Cache & Execution Model
- **Cache Mapping**: Line ID $L = \lfloor \text{address} / \text{line\_size\_bytes} \rfloor$.
- **LRU Eviction**: Maintained via an exact recency map. When resident lines reach capacity $N_{\text{lines}} = \text{capacity\_bytes} / \text{line\_size\_bytes}$, the least recently used line is evicted.
- **CPU Latency Equation**:
  $$T_{\text{CPU}} = N_{\text{hit}} \times t_{\text{hit}} + N_{\text{miss}} \times t_{\text{miss}} + N_{\text{ops}} \times t_{\text{compute}} + T_{\text{writeback}}$$
  where:
  - $t_{\text{miss}}$ is the complete service time for a cache miss (replaces, rather than adds to, hit latency).
  - $T_{\text{writeback}} = N_{\text{dirty\_evictions}} \times t_{\text{miss}}$ (zero for clean/read-only inputs).

- **CPU-no-cache Baseline**:
  $$T_{\text{CPU-no-cache}} = N_{\text{accesses}} \times t_{\text{miss}} + N_{\text{ops}} \times t_{\text{compute}}$$

### B. Processing-in-Memory (PIM) Model
- **PIM Latency Equation**:
  $$T_{\text{PIM}} = T_{\text{H2P}} + T_{\text{compute\_PIM}} + T_{\text{P2H}} + T_{\text{launch}} + T_{\text{sync}} + T_{\text{imbalance}}$$
  where:
  - $T_{\text{H2P}} = \frac{\text{input\_bytes}}{\text{BW}_{\text{H2P}}} + t_{\text{fixed\_H2P}}$
  - $T_{\text{P2H}} = \frac{\text{output\_bytes}}{\text{BW}_{\text{P2H}}} + t_{\text{fixed\_P2H}}$
  - $T_{\text{compute\_base}} = \frac{N_{\text{ops}}}{B \times \text{OPS\_per\_ns\_per\_bank}}$ (across $B$ banks)
  - Bank load distribution: Operations assigned deterministically across banks $b \in [0, B-1]$ yielding per-bank work $W_b$.
  - $\text{imbalance\_factor} = \frac{\max_b(W_b)}{\text{mean}_b(W_b)}$
  - $T_{\text{imbalance}} = T_{\text{compute\_base}} \times (\text{imbalance\_factor} - 1.0)$
  - Total compute time $T_{\text{compute\_base}} + T_{\text{imbalance}} = \frac{\max_b(W_b)}{\text{OPS\_per\_ns\_per\_bank}}$ (accounts for bottleneck bank without double counting parallelism).

### C. Execution Policies
1. **CPU-cache**: Always execute via modeled CPU cache.
2. **CPU-no-cache**: Direct-memory access baseline.
3. **PIM-always**: Always offload supported kernels to PIM; falls back to CPU if unsupported with reason `UNSUPPORTED_BY_PIM`.
4. **Cost-based**: Estimates $T_{\text{CPU}}$ and $T_{\text{PIM}}$ using policy-visible parameters. Offloads to PIM if $T_{\text{PIM\_est}} < T_{\text{CPU\_est}}$.
5. **Reuse-aware** (Proposed candidate): Incorporates a dimensionally consistent reuse-preservation cost in nanoseconds:
   $$T_{\text{PIM\_adjusted}} = T_{\text{PIM\_est}} + \lambda \times R_{\text{cost}}$$
   $$R_{\text{cost}} = \left(\frac{\text{effective\_reusable\_bytes}}{\text{cache\_capacity\_bytes}}\right) \times \text{estimated\_CPU\_reuse\_benefit\_ns}$$
   where $\text{estimated\_CPU\_reuse\_benefit\_ns} = N_{\text{hits\_est}} \times (t_{\text{miss}} - t_{\text{hit}})$.
   Offloads to PIM only if $T_{\text{PIM\_adjusted}} < T_{\text{CPU\_est}}$.
6. **Oracle**: Optimal offline baseline selecting $\min(T_{\text{CPU\_actual}}, T_{\text{PIM\_actual}})$.

---

## 6. Architectural Parameters & Provenance

Every parameter in CachePIM is cataloged and documented:

| Parameter | Default Value | Unit | Provenance | Source Reference |
| :--- | :--- | :--- | :--- | :--- |
| `cache.capacity_bytes` | 32768 (32 KiB) | bytes | Assumed | Sweep baseline across 16–512 KiB |
| `cache.line_size_bytes` | 64 | bytes | Literature-derived | Standard x86-64 / ARMv8 cache line |
| `cache.replacement` | LRU | policy | Assumed | True Least-Recently Used replacement |
| `cpu.frequency_hz` | 2.5e9 (2.5 GHz) | Hz | Assumed | Nominal core clock frequency |
| `cpu.hit_service_ns` | 2.0 | ns | Literature-derived | LLC hit latency (~5 cycles @ 2.5 GHz) |
| `cpu.miss_service_ns` | 80.0 | ns | Literature-derived | DRAM row-buffer hit/miss latency (JEDEC DDR4) |
| `cpu.compute_ns_per_op`| 0.4 | ns/op | Assumed | 1 op/cycle ALU throughput @ 2.5 GHz |
| `pim.bank_count` | 16 | banks | Literature-derived | HBM2 / UPMEM DPU bank architectures |
| `pim.effective_bw` | 128.0 | GB/s | Literature-derived | Aggregate internal crossbar memory BW |
| `pim.compute_ops_per_bank`| 1.0 | ops/ns/bank| Assumed | PIM bank ALU throughput (~1 GHz) |
| `pim.launch_overhead_ns`| 50.0 | ns | Assumed | Kernel dispatch and command serialization |
| `pim.sync_overhead_ns` | 20.0 | ns | Assumed | Inter-bank barrier synchronization |
| `transfer.h2p_bw` | 16.0 | GB/s | Literature-derived | PCIe 4.0 x8 / CXL 1.1 link bandwidth |
| `transfer.p2h_bw` | 16.0 | GB/s | Literature-derived | Return link bandwidth |
| `transfer.fixed_overhead_ns` | 100.0 | ns | Literature-derived | Host-device round-trip traversal latency |
| `policy.reuse_penalty_weight` | 1.0 | factor | Artifact-derived | Calibrated on development seeds 42–46 |

---

## 7. Model Scope & Known Limitations

1. **Analytical vs. Cycle-Accurate**: CachePIM uses transparent analytical timing equations rather than cycle-by-cycle microarchitectural simulation (e.g., gem5).
2. **Cache Hierarchy**: Evaluates a single-level LLC. Multi-level L1/L2 caches, prefetchers, out-of-order instruction scheduling, and memory-level parallelism (MSHRs) are abstracted.
3. **Synthetic Traces**: Memory addresses are generated from parameterized statistical patterns rather than hardware OS pin-traces.
4. **Data Coherence**: PIM and CPU operate under a clean/read-only input model with explicit dirty-line writebacks; non-blocking hardware cache-coherence protocols are not modeled.
