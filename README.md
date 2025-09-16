# Prismn — HackMIT 2025
A carbon- and latency-aware placement engine that partitions LLM layers across a global GPU swarm to deliver faster responses, lower emissions, and higher utilization—without changing model quality.

> _By Yichen Liu, Anoushk Kharangate, and Megan Kulshekar_

<br>

![Prismn Banner](Prismn_Banner.png)

![HackMIT](https://img.shields.io/badge/HackMIT-2025-purple)
![Made with Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

## Overview
Prismn splits large language models into pipeline stages and distributes those layers across multiple GPU peers—optimizing for **lower carbon intensity**, **shorter geographic distance (as a latency proxy)**, and **available VRAM**. The result is **faster responses, fewer machines involved, and lower emissions**, all while maintaining model quality.


## Inspiration
The rapid growth of LLMs has real environmental costs: high energy use and significant carbon emissions. We set out to make model serving **more sustainable by default**. Building on the open-source **Tandemn** API—which already supports distributing LLM layers across GPUs—we added a **carbon-aware router** and **distance-aware ordering** so developers can make greener, lower-latency placements without changing their models or workflows.

## What Prismn Does
1. **Scores peers** using a weighted sum of:
   - **Carbon intensity** (gCO₂/kWh) — prefer cleaner grids  
   - **Geographic distance** to ingress — proxy for lower latency  
   - **Free VRAM** — ensure capacity and higher utilization
2. **Greedily selects** the smallest feasible set of peers (in **layer units**, including embeddings).
3. **Orders peers** via a Traveling Salesman heuristic (nearest-neighbor + **2-opt** refinement) to minimize inter-peer data travel.
4. **Shards the model** accordingly and returns an execution plan: ordered peers, per-peer layer counts, VRAM utilization, average grid emissions, and approximate “tour” distance.

## How We Built It
- Extended the **Tandemn** repository to implement **carbon- and distance-aware selection** plus **pipeline ordering**.
- Added **quantization-aware per-layer VRAM estimation** (weights + activations, embeddings accounted for) with safety margins to avoid OOMs.
- Built a lightweight **PyTinker** UI to compare **VRAM-only** vs **Carbon+Distance** policies side-by-side.
- Tested multi-peer scenarios in a Linux environment.

## Key Benefits
- **Latency:** Fewer hops and shorter paths reduce end-to-end latency.
- **Sustainability:** Prefers low-carbon regions automatically, lowering emissions per request.
- **Utilization:** Packs layers by accurate per-layer VRAM, reducing idle memory and stages.
- **Pragmatic & fast:** Greedy selection + 2-opt ordering give strong results with low overhead.

## Architecture
- **Coordinator (server):** peer discovery, scoring, ordering, and layer placement.
- **Peers (GPU nodes):** report VRAM/locality; execute assigned layers.
- **Database:** MongoDB for metrics and peer state.
- **Networking:** Iroh for peer-to-peer communication.

## What's Next
* **Live carbon data:** Integrate the [Electricity Maps API](https://portal.electricitymaps.com/developer-hub/api/reference#carbon-intensity-latest) for real-time regional carbon intensity.
* **Richer sourcing signals:** Incorporate GPU specs, model type, and energy source metadata.
* **Better latency modeling:** Add measured **ping/RTT** and throughput where available (distance remains a fallback).
* **Scale ordering:** Current TSP + 2-opt performs well up to hundreds of peers; add geographic **clustering** to accelerate larger deployments.
* **Alternative formulations:** Explore **min-cut / max-flow** style approaches to balance capacity vs. distance when placement resembles network flow.
* **UI improvements:** More preferences, presets, and additional LLM choices.
* **Broader testing:** Real GPUs distributed across regions/continents.

---

## Prerequisites

- Python 3.8+
- MongoDB
- NVIDIA GPU with CUDA support (for GPU metrics)
- Iroh node

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yliu00/prismn
cd prismn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (or create a .env file):
```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="iroh_tandem"
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

## Usage

1. Start the central server:
```bash
python src/server.py
```

2. On each peer machine, run:
```bash
python src/machine_runner.py
```

The server will provide a ticket that needs to be shared with all peer machines.

## License

MIT License 

---

<p align="center"><sub>Why the extra “n”? A little homage to <strong>Tandemn</strong> ✨</sub></p>
<p align="center">Built with <span title="love">❤️</span> and <span title="coffee">☕</span> — Prismn</p>
