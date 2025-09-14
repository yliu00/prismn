# Prismn - HackMIT 2025 Project By Yichen Liu, Anoushk Kharangate, and Megan Kulshekar

A carbon- and latency-aware placement engine that partitions LLM layers across a global GPU swarm to deliver faster responses, lower emissions, and higher utilization—without changing model quality.

## Inspiration

The use of LLMs in the past few years has impacted our environment significantly. LLMs are known for outputting lots of carbon and using up an extensive amount of energy. We wanted to find a solution to manage this problem because we believe the use of LLMs will continue to increase in the future. We discovered that Tandemn had an API that distributed the layers of an LLM on multiple GPUs to make more efficient, optimal decisions regarding usage. We decided to leverage this open source code and implement a carbon-aware router to make even more informed decisions that take into consideration the impacts of LLMs on the environment. Our goal was to create a tool for devs that promotes sustainability and helps reduce the carbon footprint of these models.

## What it does

Given a request to a specified LLM model, Prismn slices the model by layers and distributes the layers across GPU peers by taking into consideration not only the VRAM capacity of each peer but also the carbon effects of choosing each peer and the distance between peers. The algorithm consists first of a weighted sum of carbon efficiency, distance, and VRAM capacity, in order to select a list of the best peers needed to fulfill the request. Then, to decide the order in which the requests will be sent, we implemented a Travelling Salesman Problem algorithm to find the shortest path that traverses all selected peers, with an additional 2-opt refinement to further reduce the distance between peers. The algorithm then returns the list of peers in order, and the model is sharded accordingly.

## How we built it

We used the open source Tandemn API repository that was available on GitHub. We modified the deploy function to use a newly created function that optimizes the distribution of layers among GPU peers while taking into account the location of each peer. We also added and modified functions to source to a specific peer based on a priority of either low carbon emission or low latency. PyTinker was used for the frontend user interface. The testing of multiple peers was performed in a Linux environment.

## Individual Contributions

Megan - Learned about the Tandemn API and about GPUs more in depth, debugged with the team on problems related to adding location data to the pre-existing API, created the frontend user interface to output the data  
Yichen - Learned about model sharding and parallel computing, implemented the layer distribution algorithm, updated a pre-existing test demo to better compare the new algorithm with the previous approach, worked with the team to connect the user interface with backend  
Anoushk - Hosted Tandemn server and GPU peers, worked with the team to connect GPU peers to server to test our algorithm on real GPUs

## Challenges we ran into

The existing Tandemn codebase lacked documentation and contained incomplete functions, making it hard to follow the code logic sometimes, so we worked with the Tandemn representatives to better understand their architecture and design choices. There were also some issues in the existing code with the heartbeat functions that we debugged with the representatives.

## Accomplishments that we're proud of

We’re proud that we were able to navigate the Tandemn repository, make sense of it, and make a meaningful contribution to the existing work. The distribution logic that we added not only successfully reduces carbon emissions, but also improves travel latency.

## What we learned

We learned a lot about model sharding, GPUs, backend operations, the carbon impacts of various LLM models, the problem of choosing which GPUs to shard to and in which order, and so much more!

## What's next for our project

We would like to implement the carbon intensity API from Electricity Maps (https://portal.electricitymaps.com/developer-hub/api/reference#carbon-intensity-latest) to have a more accurate calculation of the carbon intensity of a certain region. We did not use this during the hackathon since the features we needed from the API required a paid subscription. In the future, we also plan on improving the overall algorithm for sourcing so that it includes more metrics like GPU specs, model type, and energy source. To improve latency optimization, in addition to geographic distance, we would also use ping RTT to get the actual time needed for a signal to travel between peers for a better estimate of latency. In addition, the current TSP algorithm with 2-opt refinement works well for smaller numbers of peers (<300), but as we scale up and the number of peers increases, we would add a layer of geographic clusters to improve the speed of the TSP algorithm. We could also replace the current TSP algorithm with a max-flow min-cut algorithm, since the goal is to maximize VRAM capacity (and the inverse of the carbon intensity) while minimizing distances between peers. We also plan on adding improved controls to the UI, including more preference choices and additional LLM model choices. Our testing would also be improved to use real GPUs scattered across the globe.


## Prerequisites

- Python 3.8+
- MongoDB
- NVIDIA GPU with CUDA support (for GPU metrics)
- Iroh node

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iroh-tandem.git
cd iroh-tandem
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

## Architecture

The system consists of:
- A central server that coordinates the distributed computation
- Multiple peer machines that perform the actual computation
- MongoDB for storing metrics and peer information
- Iroh for peer-to-peer communication

## License

MIT License 
