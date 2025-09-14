#!/usr/bin/env python3
"""
Demo script showing how to calculate layer distribution across GPU peers.

This script demonstrates:
1. How to calculate VRAM per layer for a model
2. How many layers can fit in a specific peer
3. How to distribute layers optimally across multiple peers
"""

import asyncio
from src.utils.model_utils import (
    download_config,
    estimate_layer_vram,
    calculate_max_layers_for_peer,
    distribute_layers_across_peers,
)

async def demo_layer_calculations():
    """Demonstrate layer calculation functionality"""
    print("🚀 Layer Calculation Demo")
    print("=" * 50)

    # Example: Llama-2-7B model
    model_id = "meta-llama/Llama-3.2-1B" #"meta-llama/Llama-2-70b-hf"
    qbits = 16  # FP16 quantization

    try:
        # 1. Download model configuration
        print(f"📥 Downloading config for {model_id}...")
        config = await download_config(model_id)
        print(f"✅ Model has {config.get('num_hidden_layers')} layers")

        # 2. Calculate VRAM per layer
        print(f"\n🔢 Calculating VRAM requirements ({qbits}-bit)...")
        vram_per_layer, embedding_vram, total_vram = estimate_layer_vram(config, qbits)
        print(f"   VRAM per layer: {vram_per_layer:.3f} GB")
        print(f"   Embedding VRAM: {embedding_vram:.3f} GB")
        print(f"   Total model VRAM: {total_vram:.3f} GB")

        # Small capacities force multiple peers to be picked
        example_peers = {
            "peer_green_1": 1.4,   # low CI
            "peer_green_2": 1.3,   # low CI
            "peer_mid_1":   0.7,   # mid CI
            "peer_brown_1": 1.6,   # high CI
        }

        example_carbon_intensity = {
            "peer_green_1": 120.0,  # e.g., hydro-heavy grid
            "peer_green_2": 140.0,
            "peer_mid_1":   300.0,
            "peer_brown_1": 480.0,
        }

        example_locations = {
            "peer_green_1": (64.1265, -21.8174),  # Reykjavik
            "peer_green_2": (59.9139, 10.7522),   # Oslo
            "peer_mid_1":   (52.5200, 13.4050),   # Berlin
            "peer_brown_1": (39.0997, -94.5786),  # Kansas City
        }


        print("\n🖥️  Example GPU Peers:")
        for peer_id, vram in example_peers.items():
            print(f"   {peer_id}: {vram} GB free VRAM")

        # 4. Calculate capacity for each peer individually
        print("\n📊 Individual Peer Capacities:")
        for peer_id, available_vram in example_peers.items():
            calculation = calculate_max_layers_for_peer(config, available_vram, qbits)
            print(f"\n   {peer_id} ({available_vram} GB free):")
            print(f"      Max layers (layers only): {calculation['max_layers_only']}")
            print(
                f"      Max layers (with embeddings): {calculation['max_layers_with_embeddings']}"
            )
            print(
                f"      VRAM utilization: {calculation['vram_used_layers_only']:.2f} GB"
            )

        # 5. Compare VRAM-only vs Optimized approaches
        print("\n🔄 Comparing Distribution Approaches:")
        print("=" * 50)
        
        # VRAM-only approach (old method - simulate by using same carbon/location for all)
        print("\n📊 VRAM-Only Approach (Traditional):")
        vram_only_carbon = {peer_id: 400.0 for peer_id in example_peers.keys()}  # Same carbon for all
        vram_only_locations = {peer_id: (42.3601, -71.0942) for peer_id in example_peers.keys()}  # Same location for all
        
        vram_only_distribution = distribute_layers_across_peers(
            config, example_peers, vram_only_carbon, vram_only_locations, qbits
        )
        
        print(f"   Selected peers: {vram_only_distribution['optimization_info']['selected_peers']}")
        print(f"   Peer order: {' → '.join(vram_only_distribution['optimization_info']['ordered_peers'])}")
        print(f"   Average carbon intensity: {vram_only_distribution['optimization_info']['avg_carbon_intensity']} gCO2/kWh")
        print(f"   Travel distance: {vram_only_distribution['optimization_info']['tour_distance_km']} km")
        
        # Optimized approach (new method)
        print("\n🌱 Optimized Approach (Carbon + Travel Distance):")
        optimized_distribution = distribute_layers_across_peers(
            config, example_peers, example_carbon_intensity, example_locations, qbits
        )
        
        print(f"   Selected peers: {optimized_distribution['optimization_info']['selected_peers']}")
        print(f"   Peer order: {' → '.join(optimized_distribution['optimization_info']['ordered_peers'])}")
        print(f"   Average carbon intensity: {optimized_distribution['optimization_info']['avg_carbon_intensity']} gCO2/kWh")
        print(f"   Travel distance: {optimized_distribution['optimization_info']['tour_distance_km']} km")
        
        # Comparison summary
        print("\n📈 Comparison Summary:")
        carbon_improvement = vram_only_distribution['optimization_info']['avg_carbon_intensity'] - optimized_distribution['optimization_info']['avg_carbon_intensity']
        distance_improvement = vram_only_distribution['optimization_info']['tour_distance_km'] - optimized_distribution['optimization_info']['tour_distance_km']
        
        print(f"   Carbon intensity improvement: {carbon_improvement:.1f} gCO2/kWh ({carbon_improvement/vram_only_distribution['optimization_info']['avg_carbon_intensity']*100:.1f}% reduction)")
        print(f"   Travel distance improvement: {distance_improvement:.1f} km ({distance_improvement/vram_only_distribution['optimization_info']['tour_distance_km']*100:.1f}% reduction)")
        
        # Show peer selection differences
        print("\n🔍 Peer Selection Analysis:")
        vram_selected = set(vram_only_distribution['optimization_info']['selected_peers'])
        opt_selected = set(optimized_distribution['optimization_info']['selected_peers'])
        
        if vram_selected != opt_selected:
            print(f"   Different peer selection:")
            print(f"   VRAM-only selected: {sorted(vram_selected)}")
            print(f"   Optimized selected: {sorted(opt_selected)}")
            print(f"   Peers only in VRAM approach: {sorted(vram_selected - opt_selected)}")
            print(f"   Peers only in optimized approach: {sorted(opt_selected - vram_selected)}")
        else:
            print(f"   Same peer selection: {sorted(vram_selected)}")
            print(f"   But different ordering due to travel distance optimization")
        
        # Show why certain peers were preferred
        print("\n💡 Why the optimized approach chose different peers:")
        for peer_id in example_peers.keys():
            if peer_id in example_carbon_intensity:
                carbon = example_carbon_intensity[peer_id]
                vram = example_peers[peer_id]
                node_score = optimized_distribution['optimization_info']['node_scores'].get(peer_id, 'N/A')
                selected_status = "✅" if peer_id in opt_selected else "❌"
                print(f"   {peer_id}: {carbon} gCO2/kWh, {vram}GB VRAM, score: {node_score} {selected_status}")
        
        # Use the optimized distribution for detailed results
        distribution = optimized_distribution

        print(f"\n📋 Final Optimized Distribution Results:")
        print(f"   Can fit entire model: {distribution['can_fit_model']}")
        print(
            f"   Total layers to distribute: {distribution['model_info']['total_layers']}"
        )
        print(
            f"   Assigned layers: {distribution['model_info']['total_assigned_layers']}"
        )
        print(
            f"   Utilized peers: {distribution['utilized_peers']}/{distribution['total_peers']}"
        )

        print("\n   Detailed Distribution:")
        for peer_id, allocation in distribution["distribution"].items():
            handles_emb = "✅" if allocation["handles_embeddings"] else "❌"
            print(f"      {peer_id}:")
            print(f"         Assigned layers: {allocation['assigned_layers']}")
            print(f"         Handles embeddings: {handles_emb}")
            print(f"         VRAM usage: {allocation['estimated_vram_usage']} GB")
            print(f"         Utilization: {allocation['vram_utilization_percent']}%")
            print(f"         Carbon intensity: {allocation['carbon_intensity']} gCO2/kWh")
            print(f"         Location: {allocation['location']['lat']:.4f}, {allocation['location']['lon']:.4f}")
        
        # Show optimization results
        if "optimization_info" in distribution:
            opt_info = distribution["optimization_info"]
            print(f"\n🌱 Optimization Results:")
            print(f"   Average carbon intensity: {opt_info['avg_carbon_intensity']} gCO2/kWh")
            print(f"   Total travel distance: {opt_info['tour_distance_km']} km")
            print(f"   Optimized peer order: {' → '.join(opt_info['ordered_peers'])}")
            print(f"   Node scores: {opt_info['node_scores']}")

        # 6. Show what happens with insufficient VRAM
        print("\n⚠️  Example: Insufficient VRAM scenario")
        small_peers = {
            "peer_small_1": 2.0,  # Only 2GB free VRAM
            "peer_small_2": 3.0,  # Only 3GB free VRAM
        }
        
        small_carbon_intensity = {
            "peer_small_1": 400.0,
            "peer_small_2": 400.0,
        }
        
        small_locations = {
            "peer_small_1": (42.3601, -71.0942),  # MIT
            "peer_small_2": (42.3601, -71.0942),  # MIT
        }

        small_distribution = distribute_layers_across_peers(
            config, small_peers, small_carbon_intensity, small_locations, qbits
        )
        print(
            f"   Can fit model with small peers: {small_distribution['can_fit_model']}"
        )
        print(
            f"   Remaining unassigned layers: {small_distribution['remaining_layers']}"
        )

        return True

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return False


def demo_api_usage():
    """Show example API calls for the new endpoints"""
    print("\n📡 API Usage Examples:")
    print("=" * 30)

    print("1. Calculate layers for specific peer:")
    print("""
    POST /calculate_peer_layers
    {
        "model_id": "meta-llama/Llama-3.2-1B",
        "hf_token": "your_token",
        "qbits": 16,
        "peer_id": "peer_rtx_4090",
        "available_vram_gb": 22.0
    }
    """)

    print("2. Get current peer capacity:")
    print("""
    GET /peer_layer_capacity/peer_rtx_4090?model_id=meta-llama/Llama-2-7b-hf&qbits=16
    """)

    print("3. Create distribution plan:")
    print("""
    POST /distribute_model_layers
    {
        "model_id": "meta-llama/Llama-2-7b-hf", 
        "hf_token": "your_token",
        "qbits": 16
    }
    """)


if __name__ == "__main__":
    print("🎮 Starting Layer Calculation Demo")

    # Run the async demo
    success = asyncio.run(demo_layer_calculations())

    if success:
        # Show API usage examples
        # demo_api_usage()
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed!")
