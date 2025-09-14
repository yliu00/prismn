#!/bin/bash

# Script to stop all running peers
echo "🛑 Stopping all peers..."

# Kill all machine_runner processes
pkill -f "machine_runner.py"

# Remove PID files
rm -f peer*.pid

echo "✅ All peers stopped!"
