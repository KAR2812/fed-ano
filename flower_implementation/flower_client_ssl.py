# flower_implementation/flower_client_ssl.py

"""
Classic Flower NumPyClient for the smart grid anomaly detection project.

Usage (after starting the server):

    python flower_implementation/flower_client_ssl.py \
        --server 127.0.0.1:8080 \
        --data processed/client_0.npz \
        --model ffnn
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.client import NumPyClient
from sklearn.metrics import classification_report

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model_utils import get_model_by_name


# ------------------ CLI Argument Parsing ------------------
parser = argparse.ArgumentParser(description="Flower Client for Federated Smart Grid Anomaly Detection")
parser.add_argument("--server", default="127.0.0.1:8080", help="Server address (default: 127.0.0.1:8080)")
parser.add_argument("--data", default="processed/client_0.npz", help="Path to client dataset (.npz)")
parser.add_argument("--model", default="ffnn",
                    choices=["logistic", "ffnn", "conv1d", "rnn", "lstm", "gru"],
                    help="Model architecture to train")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for local training")
parser.add_argument("--epochs", type=int, default=1, help="Local epochs per federated round")
args = parser.parse_args()


# ------------------ Data Loading ------------------
if not os.path.exists(args.data):
    raise FileNotFoundError(f"❌ Dataset not found: {args.data}")

d = np.load(args.data)
X, y = d["X"], d["y"]
input_dim = X.shape[1]

# Build selected model
model = get_model_by_name(args.model, input_dim)


# ------------------ Define NumPyClient ------------------
class FlowerSmartGridClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        X_train, y_train = X, y
        if args.model == "conv1d":
            X_train = X_train[..., np.newaxis]
        elif args.model in ("rnn", "lstm", "gru"):
            X_train = X_train[:, np.newaxis, :]

        model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )
        print(f"📈 Finished local training on {len(X_train)} samples.")

        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        X_eval, y_eval = X, y
        if args.model == "conv1d":
            X_eval = X_eval[..., np.newaxis]
        elif args.model in ("rnn", "lstm", "gru"):
            X_eval = X_eval[:, np.newaxis, :]

        # Keras metrics: loss and accuracy
        loss, acc = model.evaluate(X_eval, y_eval, verbose=0)

        # Get predictions and compute per-class metrics
        y_proba = model.predict(X_eval, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        report = classification_report(
            y_eval, y_pred,
            output_dict=True,
            zero_division=0,
        )

        def get(cls: int, key: str) -> float:
            return float(report.get(str(cls), {}).get(key, 0.0))

        metrics = {
            "accuracy": float(acc),
            "precision_spike": get(1, "precision"),
            "recall_spike": get(1, "recall"),
            "precision_season": get(2, "precision"),
            "recall_season": get(2, "recall"),
            "precision_outlier": get(3, "precision"),
            "recall_outlier": get(3, "recall"),
        }

        print("🧮 Multi-class anomaly report (local client):")
        print(f"   Overall accuracy        : {acc:.4f}")
        print(f"   Type 1 (Spike)         : P={metrics['precision_spike']:.3f}, R={metrics['recall_spike']:.3f}")
        print(f"   Type 2 (Seasonal break): P={metrics['precision_season']:.3f}, R={metrics['recall_season']:.3f}")
        print(f"   Type 3 (Outlier)       : P={metrics['precision_outlier']:.3f}, R={metrics['recall_outlier']:.3f}")

        return float(loss), len(X_eval), metrics



# ------------------ Start the Client ------------------
print("✅ Client initialized successfully.")
print(f"🌐 Connecting to Flower server at: {args.server}")
print("------------------------------------------------------------")
print("🧩 Launch sequence:")
print("   1️⃣ python flower_implementation/flower_server_ssl.py")
print(f"   2️⃣ python flower_implementation/flower_client_ssl.py --server {args.server} --data {args.data} --model {args.model}")
print("------------------------------------------------------------")

fl.client.start_numpy_client(
    server_address=args.server,
    client=FlowerSmartGridClient(),
)
