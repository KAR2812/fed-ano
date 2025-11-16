# flower_implementation/flower_server_ssl.py

import os
import sys
import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

# Ensure local imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model_utils import get_model_by_name

# Global storage for final model weights
final_weights = None


class SaveModelStrategy(FedAvg):
    """Custom FedAvg strategy that saves global model weights."""

    def aggregate_fit(self, server_round, results, failures):
        global final_weights

        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None:
            params, _ = aggregated
            print(f"💾 Round {server_round}: Saving aggregated global weights")
            final_weights = fl.common.parameters_to_ndarrays(params)

        return aggregated


def main():
    global final_weights

    INPUT_DIM = 3
    MODEL_ARCH = "ffnn"
    ROUNDS = 3

    print("🚀 Starting Flower Server: 0.0.0.0:8080")

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    config = ServerConfig(num_rounds=ROUNDS)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

    if final_weights is not None:
        print("🎯 Training finished — exporting global FL model...")
        model = get_model_by_name(MODEL_ARCH, INPUT_DIM)
        model.set_weights(final_weights)
        np.savez("saved_global_model.npz", weights=np.array(model.get_weights(), dtype=object), allow_pickle=True)

        print("💾 Saved: saved_global_model.npz")
    else:
        print("⚠ No global weights received — Training might have failed!")


if __name__ == "__main__":
    main()
