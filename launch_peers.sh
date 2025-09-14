#!/bin/bash

# Script to launch multiple peers with different environment files
# Usage: ./launch_peers.sh [number_of_peers]

NUM_PEERS=${1:-3}  # Default to 3 peers if no argument provided

echo "🚀 Launching $NUM_PEERS peers..."

# Function to launch a peer in the background
launch_peer() {
    local peer_num=$1
    local env_file="peer${peer_num}.env"
    
    if [ ! -f "$env_file" ]; then
        echo "❌ Environment file $env_file not found!"
        return 1
    fi
    
    echo "🔧 Starting peer $peer_num with $env_file..."
    source venv/bin/activate
    python src/machine_runner.py --env "$env_file" --peer-name "peer$peer_num" &
    
    # Store the PID
    echo $! > "peer${peer_num}.pid"
    echo "✅ Peer $peer_num started with PID $(cat peer${peer_num}.pid)"
}

# Launch peers
for i in $(seq 1 $NUM_PEERS); do
    launch_peer $i
    sleep 2  # Small delay between launches
done

echo "🎉 All $NUM_PEERS peers launched!"
echo "📋 To stop all peers, run: ./stop_peers.sh"
echo "📊 To check peer status, run: ps aux | grep machine_runner"
