#!/usr/bin/env bash
# Optional helper to run a local Flower server and multiple local clients (non-TLS) for quick checks.
# This script runs server in background and spawns local clients.

python3 -m pip install flwr

# Start server in background (no TLS)
python3 - <<'PY'
import flwr as fl
from flwr.server.strategy import FedAvg
strategy = FedAvg()
fl.server.start_server(server_address="0.0.0.0:8080", config={"num_rounds":3}, strategy=strategy)
PY

# Spawn clients: adjust the data paths to processed/client_0.npz etc.
for i in 0 1 2 3; do
  python3 flower_implementation/flower_client_ssl.py --server 127.0.0.1:8080 --data processed/client_${i}.npz --model ffnn &
done

wait
