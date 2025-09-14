#!/usr/bin/env python3
"""
Demo: compare a baseline (VRAM-only) plan vs an improved (Carbon + Travel) plan.

This version prints non-technical, business-readable summaries:
- Executive Summary with key improvements
- Side-by-side comparison table
- Short explanations of "why" the improved plan chose certain machines
"""

import asyncio
from src.utils.model_utils import (
    download_config,
    estimate_layer_vram,
    calculate_max_layers_for_peer,
    distribute_layers_across_peers,
    optimized_distribute_layers_across_peers,
)

# Toggle to show or hide low-level capacity prints
VERBOSE = False

def _pct_change(old: float, new: float) -> float:
    return 0.0 if old == 0 else ((old - new) / old) * 100.0

def _print_hr():
    print("-" * 70)

def _print_exec_summary(baseline, improved, preference):
    b = baseline["optimization_info"]
    i = improved["optimization_info"]
    carbon_drop = b["avg_carbon_intensity"] - i["avg_carbon_intensity"]
    dist_drop = b["tour_distance_km"] - i["tour_distance_km"]
    carbon_pct = _pct_change(b["avg_carbon_intensity"], i["avg_carbon_intensity"])
    dist_pct = _pct_change(b["tour_distance_km"], i["tour_distance_km"])

    print("\n📣 EXECUTIVE SUMMARY")
    _print_hr()
    print(f"• Lower emissions: -{carbon_drop:.1f} gCO₂/kWh  ({carbon_pct:.1f}% improvement)")
    print(f"• Less inter-machine travel: -{dist_drop:.1f} km  ({dist_pct:.1f}% improvement)")
    if preference == "Low Latency":
        print("• We prioritized lower latency between machines while still fitting the model in memory and keeping emissions low.")
    else:
        print("• We prioritized lower carbon emissions while still fitting the model in memory and keeping data hops short.")

def _print_side_by_side(baseline, improved):
    b = baseline["optimization_info"]
    i = improved["optimization_info"]

    print("\n📊 SIDE-BY-SIDE COMPARISON")
    _print_hr()
    print(f"{'Metric':35s} {'Baseline (VRAM only)':>22s}   {'Improved (Carbon + Travel)':>26s}")
    _print_hr()
    print(f"{'Machines used':35s} {str(len(baseline['distribution'])):>22s}   {str(len(improved['distribution'])):>26s}")
    print(f"{'Can fit entire model':35s} {str(baseline['can_fit_model']):>22s}   {str(improved['can_fit_model']):>26s}")
    print(f"{'Avg grid emissions (gCO₂/kWh)':35s} {b['avg_carbon_intensity']:>22.1f}   {i['avg_carbon_intensity']:>26.1f}")
    print(f"{'Total data travel (km)':35s} {b['tour_distance_km']:>22.1f}   {i['tour_distance_km']:>26.1f}")
    print(f"{'Selected machines':35s} {', '.join(b['selected_peers'])[:60]:>22s}   {', '.join(i['selected_peers'])[:60]:>26s}")
    _print_hr()

def _print_why_different(baseline, improved, peers_vram, peers_ci):
    b_set = set(baseline["optimization_info"]["selected_peers"])
    i_set = set(improved["optimization_info"]["selected_peers"])
    only_b = sorted(b_set - i_set)
    only_i = sorted(i_set - b_set)

    print("\n🔍 WHAT CHANGED AND WHY")
    _print_hr()
    if only_b or only_i:
        if only_b:
            print("Machines dropped in the improved plan:")
            for pid in only_b:
                print(f"  - {pid}  (grid: {peers_ci.get(pid, 'n/a')} gCO₂/kWh, free VRAM: {peers_vram.get(pid, 'n/a')} GB)")
        if only_i:
            print("New machines added in the improved plan:")
            for pid in only_i:
                print(f"  + {pid}  (cleaner grid: {peers_ci.get(pid, 'n/a')} gCO₂/kWh, free VRAM: {peers_vram.get(pid, 'n/a')} GB)")
    else:
        print("Same set of machines, reordered to shorten data travel.")

async def demo_layer_calculations(preference=None):
    print("🚀 Layer Distribution Demo")
    _print_hr()

    model_id = "meta-llama/Llama-3.2-1B"
    qbits = 16

    try:
        print(f"Fetching model details for {model_id} ...")
        config = await download_config(model_id)
        print(f"Model layers: {config.get('num_hidden_layers')}")

        vram_per_layer, embedding_vram, total_vram = estimate_layer_vram(config, qbits)
        print(f"Approximate memory per layer: {vram_per_layer:.3f} GB  •  Embeddings: {embedding_vram:.3f} GB  •  Total: {total_vram:.3f} GB")

        # --- Demo peers (as provided) ---
        """peers_info_demo = {
            "peer_oslo_3060":   {"free_vram_gb": 0.164, "lat": 59.9139, "lon": 10.7522, "carbon_intensity_g_per_kwh": 18.0},
            "peer_paris_a6000": {"free_vram_gb": 0.625, "lat": 48.8566, "lon": 2.3522,  "carbon_intensity_g_per_kwh": 44.0},
            "peer_stockholm_4090": {"free_vram_gb": 0.362,"lat": 59.3293,"lon": 18.0686,"carbon_intensity_g_per_kwh": 19.0},
            "peer_berlin_3080": {"free_vram_gb":  0.132, "lat": 52.5200, "lon": 13.4050, "carbon_intensity_g_per_kwh": 419.0},
            "peer_warsaw_t4x2": {"free_vram_gb": 0.428, "lat": 52.2297, "lon": 21.0122, "carbon_intensity_g_per_kwh": 692.0},
            "peer_saopaulo_a100": {"free_vram_gb": 1.217,"lat": -23.5505,"lon": -46.6333,"carbon_intensity_g_per_kwh": 97.0},
            "peer_sfo_v100":    {"free_vram_gb": 0.493, "lat": 37.7749, "lon": -122.4194,"carbon_intensity_g_per_kwh": 384.0},
            "peer_sydney_h100": {"free_vram_gb": 1.250, "lat": -33.8688,"lon": 151.2093, "carbon_intensity_g_per_kwh": 626.0},
            "peer_delhi_3070":  {"free_vram_gb":  0.099, "lat": 28.6139, "lon": 77.2090, "carbon_intensity_g_per_kwh": 632.0},
            "peer_shanghai_a10":{"free_vram_gb": 0.329, "lat": 31.2304, "lon": 121.4737,"carbon_intensity_g_per_kwh": 560.0},
        }
        """
        peers_info_demo = {
    "peer_oslo_3060":      {"free_vram_gb": 0.35, "lat": 59.9139,  "lon": 10.7522,   "carbon_intensity_g_per_kwh": 18.0},
    "peer_stockholm_4090": {"free_vram_gb": 0.85, "lat": 59.3293,  "lon": 18.0686,   "carbon_intensity_g_per_kwh": 19.0},
    "peer_paris_a6000":    {"free_vram_gb": 1.10, "lat": 48.8566,  "lon": 2.3522,    "carbon_intensity_g_per_kwh": 44.0},
    "peer_berlin_3080":    {"free_vram_gb": 0.25, "lat": 52.5200,  "lon": 13.4050,   "carbon_intensity_g_per_kwh": 419.0},
    "peer_warsaw_t4x2":    {"free_vram_gb": 0.75, "lat": 52.2297,  "lon": 21.0122,   "carbon_intensity_g_per_kwh": 692.0},
    "peer_saopaulo_a100":  {"free_vram_gb": 1.20, "lat": -23.5505, "lon": -46.6333,  "carbon_intensity_g_per_kwh": 97.0},
    "peer_sfo_v100":       {"free_vram_gb": 0.80, "lat": 37.7749,  "lon": -122.4194, "carbon_intensity_g_per_kwh": 384.0},
    "peer_sydney_h100":    {"free_vram_gb": 1.25, "lat": -33.8688, "lon": 151.2093,  "carbon_intensity_g_per_kwh": 626.0},
    "peer_delhi_3070":     {"free_vram_gb": 0.20, "lat": 28.6139,  "lon": 77.2090,   "carbon_intensity_g_per_kwh": 632.0},
    "peer_shanghai_a10":   {"free_vram_gb": 0.45, "lat": 31.2304,  "lon": 121.4737,  "carbon_intensity_g_per_kwh": 560.0},
    "peer_london_3090":    {"free_vram_gb": 0.60, "lat": 51.5074,  "lon": -0.1278,   "carbon_intensity_g_per_kwh": 215.0},
    "peer_madrid_3080":    {"free_vram_gb": 0.40, "lat": 40.4168,  "lon": -3.7038,   "carbon_intensity_g_per_kwh": 230.0},
    "peer_toronto_a5000":  {"free_vram_gb": 0.95, "lat": 43.6532,  "lon": -79.3832,  "carbon_intensity_g_per_kwh": 120.0},
    "peer_tokyo_a10g":     {"free_vram_gb": 0.55, "lat": 35.6762,  "lon": 139.6503,  "carbon_intensity_g_per_kwh": 430.0},
    "peer_cape_town_3070": {"free_vram_gb": 0.30, "lat": -33.9249, "lon": 18.4241,   "carbon_intensity_g_per_kwh": 820.0},
}


        server_location = (42.3601, -71.0589) # MIT

        example_peers = {pid: info["free_vram_gb"] for pid, info in peers_info_demo.items()}
        example_locations = {pid: (info["lat"], info["lon"]) for pid, info in peers_info_demo.items()}
        example_carbon = {pid: info["carbon_intensity_g_per_kwh"] for pid, info in peers_info_demo.items()}

        if VERBOSE:
            print("\n🖥️ Machines & available memory (GB):")
            for pid, v in example_peers.items():
                print(f"  {pid}: {v:.1f} GB")

        print("\nCalculating max layers per machine...")
        print(f"{'='*20}Baseline: VRAM-only{'='*20}")
        # --- Baseline: VRAM-only ---
        baseline = distribute_layers_across_peers(
            config, example_peers, example_carbon, example_locations, qbits
        )

        print(f"\n{'='*20}Improved: Carbon + Travel{'='*20}")
        # --- Improved: Carbon + Travel ---
        improved = optimized_distribute_layers_across_peers(
            config, example_peers, example_carbon, example_locations, qbits, server_location=server_location, preference=preference
        )

        # --- Executive view: key outcomes ---
        _print_exec_summary(baseline, improved, preference)
        _print_side_by_side(baseline, improved)
        _print_why_different(baseline, improved, example_peers, example_carbon)

        # Optional: show the final detailed assignments (still readable)
        print("\n📋 FINAL IMPROVED PLAN – MACHINE ASSIGNMENTS")
        _print_hr()
        for pid, a in improved["distribution"].items():
            emb = " (stores model dictionary)" if a["handles_embeddings"] else ""
            print(f"- {pid}{emb}: {a['assigned_layers']} layers  •  Utilization: {a['vram_utilization_percent']}%  •  Grid: {a['carbon_intensity']} gCO₂/kWh")

        print("\nFinished. ")
        if preference == "Low Latency":
            print("The improved plan lowers latency between machines and reduces carbon emissions while keeping the model within memory limits.")
        else:
            print("The improved plan lowers carbon emissions and reduces data travel while keeping the model within memory limits.")
        return True

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return False


if __name__ == "__main__":
    print("🎮 Starting Layer Distribution Demo")
    ok = asyncio.run(demo_layer_calculations("Low Carbon Emissions"))
    print("\n✅ Demo completed successfully!" if ok else "\n❌ Demo failed!")
