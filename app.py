"""CachePIM Interactive Research Dashboard.

Streamlit dashboard calling the exact same core simulation models as the CLI.
Provides interactive controls, component-level cost breakdowns, crossover heatmaps,
parameter provenance audits, and CSV export capabilities.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cachepim.config import CachePIMConfig, get_parameter_provenance
from cachepim.simulator import Simulator
from cachepim.workloads import WorkloadGenerator

st.set_page_config(
    page_title="CachePIM: Research Simulator Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ CachePIM: Cache-Aware CPU–PIM Simulator")
st.markdown(
    "> **Notice:** All metrics displayed are *modeled analytical simulation estimates*, "
    "not physical hardware measurements. Designed for architectural sensitivity and crossover characterization."
)

# -------------------------------------------------------------
# Sidebar Configuration Controls
# -------------------------------------------------------------
st.sidebar.header("Workload Parameters")
pattern = st.sidebar.selectbox("Access Pattern", ["STREAM", "REUSE", "RANDOM", "MIXED_PHASE"], index=1)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=99999, value=42, step=1)
working_set_kb = st.sidebar.slider("Working-Set Size (KiB)", min_value=8, max_value=256, value=32, step=8)
access_count = st.sidebar.slider("Access Count", min_value=512, max_value=8192, value=2048, step=512)

hot_cold_ratio = 0.8
hot_set_fraction = 0.2
if pattern == "REUSE":
    hot_cold_ratio = st.sidebar.slider("Hot Access Ratio", 0.1, 0.99, 0.8, 0.05)
    hot_set_fraction = st.sidebar.slider("Hot Set Fraction", 0.05, 0.5, 0.2, 0.05)

st.sidebar.header("CPU & Cache Architecture")
capacity_kb = st.sidebar.select_slider("LLC Cache Capacity (KiB)", options=[16, 32, 64, 128, 256, 512], value=32)
line_size_bytes = st.sidebar.selectbox("Line Size (Bytes)", [32, 64, 128], index=1)
hit_service_ns = st.sidebar.number_input("Cache Hit Service (ns)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
miss_service_ns = st.sidebar.number_input("Cache Miss Service (ns)", min_value=20.0, max_value=200.0, value=80.0, step=5.0)

st.sidebar.header("PIM & Interconnect Architecture")
bank_count = st.sidebar.select_slider("PIM Bank Count", options=[4, 8, 16, 32, 64], value=16)
pim_bandwidth_gb_s = st.sidebar.slider("Internal Aggregate BW (GB/s)", 32.0, 512.0, 128.0, 16.0)
transfer_bw_gb_s = st.sidebar.slider("Host–PIM Link BW (GB/s)", 2.0, 64.0, 16.0, 2.0)
launch_overhead_ns = st.sidebar.number_input("Kernel Launch Overhead (ns)", 0.0, 500.0, 50.0, 10.0)

st.sidebar.header("Policy Tuning")
reuse_penalty_weight = st.sidebar.slider("Reuse Penalty Weight (λ)", 0.0, 5.0, 1.0, 0.25)

# Construct Config
config = CachePIMConfig()
config.cache.capacity_bytes = capacity_kb * 1024
config.cache.line_size_bytes = line_size_bytes
config.cpu.hit_service_ns = float(hit_service_ns)
config.cpu.miss_service_ns = float(miss_service_ns)
config.pim.bank_count = int(bank_count)
config.pim.effective_bandwidth_bytes_per_ns = float(pim_bandwidth_gb_s)
config.pim.launch_overhead_ns = float(launch_overhead_ns)
config.transfer.host_to_pim_bandwidth_bytes_per_ns = float(transfer_bw_gb_s)
config.transfer.pim_to_host_bandwidth_bytes_per_ns = float(transfer_bw_gb_s)
config.policy.reuse_penalty_weight = float(reuse_penalty_weight)
config.validate()

# -------------------------------------------------------------
# Main Simulation Execution
# -------------------------------------------------------------
gen = WorkloadGenerator(line_size_bytes=line_size_bytes)
trace, desc = gen.generate(
    pattern=pattern,
    seed=int(seed),
    working_set_bytes=working_set_kb * 1024,
    access_count=int(access_count),
    hot_cold_ratio=hot_cold_ratio,
    hot_set_fraction=hot_set_fraction,
)

sim = Simulator(config)
results = sim.run_all_policies_on_trace(trace, desc)
res_dict = {r.policy_name: r for r in results}
df_results = pd.DataFrame([r.to_dict() for r in results])

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Interactive Simulation",
    "🗺️ Crossover Analysis",
    "🧪 Batch Experiments",
    "📚 Architectural Provenance & Assumptions",
])

with tab1:
    # Top KPI Metrics for Reuse-Aware Policy
    target_res = res_dict["Reuse-aware"]
    cpu_res = res_dict["CPU-cache"]
    oracle_res = res_dict["Oracle"]

    speedup = cpu_res.selected_model_ns / target_res.selected_model_ns

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Selected Path", target_res.selected_path, delta=target_res.decision_reason)
    col2.metric("Execution Time", f"{target_res.selected_model_ns / 1000.0:.2f} μs")
    col3.metric("Speedup vs. CPU-cache", f"{speedup:.2f}x")
    col4.metric("Cache Hit Rate", f"{target_res.cache_hit_rate * 100.0:.1f}%")
    col5.metric("Decision Regret", f"{target_res.regret_ns / 1000.0:.2f} μs")

    st.subheader("Policy Comparison Summary")
    summary_table = df_results[[
        "policy_name", "selected_path", "selected_model_ns",
        "cache_hit_rate", "regret_ns", "bad_offload", "decision_reason"
    ]].copy()
    summary_table["Latency (μs)"] = (summary_table["selected_model_ns"] / 1000.0).round(2)
    summary_table["Hit Rate (%)"] = (summary_table["cache_hit_rate"] * 100.0).round(1)
    summary_table["Regret (μs)"] = (summary_table["regret_ns"] / 1000.0).round(2)
    st.dataframe(
        summary_table[["policy_name", "selected_path", "Latency (μs)", "Hit Rate (%)", "Regret (μs)", "bad_offload", "decision_reason"]],
        use_container_width=True,
    )

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        # Latency Bar Chart
        plot_df = summary_table.copy()
        fig_lat = px.bar(
            plot_df,
            x="policy_name",
            y="Latency (μs)",
            color="selected_path",
            title="Execution Latency by Policy",
            color_discrete_map={"CPU": "#1f77b4", "PIM": "#2ca02c"},
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    with c2:
        # Cache breakdown
        pie_data = pd.DataFrame({
            "Category": ["Hits", "Misses"],
            "Count": [target_res.cache_hits, target_res.cache_misses],
        })
        fig_pie = px.pie(
            pie_data,
            names="Category",
            values="Count",
            title=f"CPU LLC Event Breakdown ({target_res.cache_accesses} accesses)",
            color="Category",
            color_discrete_map={"Hits": "#2ca02c", "Misses": "#d62728"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Component-level cost breakdown
    st.subheader("Component-Level Cost Breakdown")
    pim_breakdown = sim.pim_model.evaluate(desc)
    cpu_breakdown = sim.cpu_model.evaluate(
        sim.cache_model.get_stats(), desc.operation_count
    )

    c3, c4 = st.columns(2)
    with c3:
        cpu_cost_df = pd.DataFrame({
            "Component": ["Cache Service", "Arithmetic Compute", "Writeback"],
            "Latency (μs)": [
                cpu_breakdown.cache_service_time_ns / 1000.0,
                cpu_breakdown.compute_time_ns / 1000.0,
                cpu_breakdown.writeback_time_ns / 1000.0,
            ],
        })
        fig_cpu_cost = px.bar(
            cpu_cost_df, x="Component", y="Latency (μs)", title="CPU Cost Breakdown", color="Component"
        )
        st.plotly_chart(fig_cpu_cost, use_container_width=True)

    with c4:
        if pim_breakdown:
            pim_cost_df = pd.DataFrame({
                "Component": [
                    "Host->PIM Link", "Bank Compute", "PIM->Host Link",
                    "Launch Overhead", "Sync Overhead", "Bank Imbalance"
                ],
                "Latency (μs)": [
                    pim_breakdown.h2p_transfer_ns / 1000.0,
                    pim_breakdown.compute_ns / 1000.0,
                    pim_breakdown.p2h_transfer_ns / 1000.0,
                    pim_breakdown.launch_overhead_ns / 1000.0,
                    pim_breakdown.sync_overhead_ns / 1000.0,
                    pim_breakdown.imbalance_overhead_ns / 1000.0,
                ],
            })
            fig_pim_cost = px.bar(
                pim_cost_df, x="Component", y="Latency (μs)", title="PIM Cost Breakdown", color="Component"
            )
            st.plotly_chart(fig_pim_cost, use_container_width=True)

    # Bank Load Distribution
    if pim_breakdown:
        st.subheader("PIM Bank Workload Distribution")
        bank_df = pd.DataFrame({
            "Bank ID": [f"Bank {i}" for i in range(len(pim_breakdown.per_bank_work))],
            "Operations": pim_breakdown.per_bank_work,
        })
        fig_bank = px.bar(bank_df, x="Bank ID", y="Operations", title="Per-Bank Work Distribution")
        st.plotly_chart(fig_bank, use_container_width=True)

with tab2:
    st.subheader("🗺️ 2D CPU/PIM Crossover Characterization")
    st.write(
        "Explore how the crossover boundary shifts as Cache Capacity and Host-to-PIM Link Bandwidth vary."
    )

    caps = [16, 32, 64, 128, 256]
    bws = [4.0, 8.0, 16.0, 32.0, 64.0]
    heatmap_matrix = []

    for b in bws:
        row = []
        for c in caps:
            t_cfg = CachePIMConfig.from_dict(config.to_dict())
            t_cfg.cache.capacity_bytes = c * 1024
            t_cfg.transfer.host_to_pim_bandwidth_bytes_per_ns = b
            t_cfg.transfer.pim_to_host_bandwidth_bytes_per_ns = b
            t_sim = Simulator(t_cfg)
            res = t_sim.run_all_policies_on_trace(trace, desc)
            rd = {r.policy_name: r.selected_model_ns for r in res}
            sp = rd["CPU-cache"] / rd["PIM-always"]
            row.append(sp)
        heatmap_matrix.append(row)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap_matrix,
            x=[f"{c} KiB" for c in caps],
            y=[f"{int(b)} GB/s" for b in bws],
            colorscale="Viridis",
            text=[[f"{val:.2f}x" for val in r] for r in heatmap_matrix],
            texttemplate="%{text}",
            colorbar_title="PIM Speedup",
        )
    )
    fig_heat.update_layout(
        title="Speedup of PIM over CPU-cache Across Architecture Space",
        xaxis_title="CPU LLC Capacity",
        yaxis_title="Host-to-PIM Link Bandwidth",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("🧪 Run Batch Parameter Sweeps")
    st.write("Execute batch experiments using the same engine as the CLI runner.")

    if st.button("🚀 Run Mini Sweep (Working-Set Sizes 8–128 KiB)"):
        with st.spinner("Simulating parameter sweep..."):
            ws_list = [8, 16, 32, 64, 128]
            sweep_rows = []
            for ws in ws_list:
                tr, dsc = gen.generate(pattern=pattern, seed=int(seed), working_set_bytes=ws * 1024, access_count=2048)
                res_list = sim.run_all_policies_on_trace(tr, dsc)
                for r in res_list:
                    d = r.to_dict()
                    d["sweep_working_set_kb"] = ws
                    sweep_rows.append(d)
            st.session_state["mini_sweep_df"] = pd.DataFrame(sweep_rows)
            st.success("Sweep completed!")

    if "mini_sweep_df" in st.session_state:
        sw_df = st.session_state["mini_sweep_df"]
        st.dataframe(sw_df.head(20), use_container_width=True)
        csv_data = sw_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sweep CSV",
            data=csv_data,
            file_name=f"cachepim_sweep_{pattern}.csv",
            mime="text/csv",
        )

with tab4:
    st.subheader("📚 Architectural Parameter Audit & Provenance")
    st.write(
        "Every parameter in CachePIM is classified into **literature-derived**, "
        "**artifact-derived**, or **assumed** with explicit units and citations."
    )
    prov_dict = get_parameter_provenance()
    prov_rows = []
    for k, v in prov_dict.items():
        prov_rows.append({
            "Parameter": k,
            "Default Value": v.value,
            "Unit": v.unit,
            "Provenance": v.provenance,
            "Description": v.description,
            "Literature / Specification Reference": v.reference,
        })
    prov_df = pd.DataFrame(prov_rows)
    st.dataframe(prov_df, use_container_width=True)
