"""
Programmatic SuperNode launcher for Flower ≥ 1.24.
This will start the server (SuperNode) using your app.
"""

import os
from flwr.supernode.supernode import main as supernode_main

os.environ["FLWR_INSECURE"] = "1"

APP_PATH = "flower_implementation.flower_server_ssl:app"

print(f"🚀 Launching Flower SuperNode for {APP_PATH}")
supernode_main([
    "--insecure",
    "--grpc-rere",              # Fleet API
    "--node-config", APP_PATH,  # Path to your ServerApp
])
