# flower_implementation/flower_server_ssl.py

import os
import sys
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerConfig
import ssl
# Ensure local imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model_utils import get_model_by_name


# ---------------- WEIGHT + DIVERGENCE FUNCTION ---------------- #

def compute_divergence(params_local, params_global):
    """Compute model divergence from the global model."""
    return float(
        sum(np.linalg.norm(l - g) for l, g in zip(params_local, params_global))
    )


def calc_weight(metrics, divergence):
    """
    Weighted score combining:
      - accuracy
      - anomaly-type average recall
      - divergence penalty
    """
    acc = float(metrics.get("accuracy", 0.0))

    recalls = [
        float(metrics.get("recall_spike", 0.0)),
        float(metrics.get("recall_season", 0.0)),
        float(metrics.get("recall_outlier", 0.0)),
    ]
    avg_recall = sum(recalls) / 3.0

    # Normalize divergence so that small divergence ~1, high divergence ~0
    norm_div = max(0.0, 1.0 - (divergence / 10.0))

    # Final score: 50% accuracy, 40% avg recall, 10% divergence
    weight = 0.5 * acc + 0.4 * avg_recall + 0.1 * norm_div
    return float(weight)


# ---------------- CUSTOM STRATEGY ---------------- #

class WeightedFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_params = None  # track global model params (list of ndarrays)

    def aggregate_fit(self, server_round, results, failures):
        print("--------------------------------")
        print(f"\n ROUND {server_round} — Weighted Aggregation")

        client_weights = []
        client_params = []

        # Initialize global params from first client if not set
        if self.global_params is None and results:
            self.global_params = parameters_to_ndarrays(results[0][1].parameters)

        for client_proxy, fit_res in results:
            params_local = parameters_to_ndarrays(fit_res.parameters)
            metrics = fit_res.metrics or {}

            if self.global_params is None:
                divergence = 0.0
            else:
                divergence = compute_divergence(params_local, self.global_params)

            weight = calc_weight(metrics, divergence)

            client_params.append(params_local)
            client_weights.append(weight)

            print(
                f" ➤ Client {client_proxy.cid}: "
                f"weight={weight:.4f}, divergence={divergence:.4f}, metrics={metrics}"
            )

        if not client_params:
            print("⚠ No client parameters received this round.")
            return None, {}

        # Normalize weights
        total = sum(client_weights)
        if total == 0:
            weights = np.ones(len(client_weights)) / len(client_weights)
        else:
            weights = np.array(client_weights, dtype=float) / float(total)

        print(" Final normalized influence:", weights)

        # Weighted aggregation
        aggregated = []
        num_layers = len(client_params[0])
        for layer_i in range(num_layers):
            layer = np.zeros_like(client_params[0][layer_i])
            for cid in range(len(client_params)):
                layer += weights[cid] * client_params[cid][layer_i]
            aggregated.append(layer)

        # Update global model for next round
        self.global_params = [p.copy() for p in aggregated]

        return ndarrays_to_parameters(aggregated), {}


# ---------------- MAIN SERVER EXECUTION ---------------- #

def main():
    INPUT_DIM = 3
    MODEL_ARCH = "ffnn"
    ROUNDS = 3

    strategy = WeightedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    print(" Weighted Federated Server running on 0.0.0.0:8080")

    config = ServerConfig(num_rounds=ROUNDS)

    # Load SSL certificates
    from pathlib import Path
    
    certs_dir = Path(PROJECT_ROOT) / "certs"
    ca_crt = certs_dir / "ca_crt.pem"
    server_crt = certs_dir / "server_crt.pem"
    server_key = certs_dir / "server_key.pem"

    certificates = (
        ca_crt.read_bytes(),
        server_crt.read_bytes(),
        server_key.read_bytes(),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
        certificates=certificates,
    )

    # Save final global weights at end
    if strategy.global_params is not None:
        print(" Saving final global model weights...")

        model = get_model_by_name(MODEL_ARCH, INPUT_DIM)
        model_weights = model.get_weights()
        params = strategy.global_params

        # Handle mismatch in length (extra non-trainable/optimizer params)
        if len(params) != len(model_weights):
            print(
                f"⚠ global_params length ({len(params)}) != model weights length "
                f"({len(model_weights)}). Trimming to match."
            )
            params_to_use = params[: len(model_weights)]
        else:
            params_to_use = params

        model.set_weights(params_to_use)
        np.savez(
            "saved_global_model.npz",
            weights=np.array(model.get_weights(), dtype=object),
            allow_pickle=True,
        )
        print(" Saved final global model weights → saved_global_model.npz")
    else:
        print("⚠ Global weights not saved — aggregation failed or no rounds run.")


if __name__ == "__main__":
    main()
