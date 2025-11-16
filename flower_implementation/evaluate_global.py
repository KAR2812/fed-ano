"""
Evaluate global anomaly detection model with type prediction.

Outputs:
 - Console table
 - CSV: evaluation_results.csv
"""

# ==========================================
# Fix Python path to allow importing models/
# ==========================================

import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("🔍 Python Path Updated:", sys.path[:3])

from models.model_utils import get_model_by_name
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report


# Debug print to verify
print("Python Path:", sys.path)

from models.model_utils import get_model_by_name
import numpy as np


# ==========================================
# CONFIG
# ==========================================
CLIENT_DATA = "processed/client_0.npz"   # pick any client dataset
MODEL_ARCH = "ffnn"                      # logistic/rnn/lstm/gru also allowed
WEIGHTS_PATH = "saved_global_model.npz"  # FL aggregated weights
OUTPUT_CSV = "evaluation_results.csv"

ANOMALY_NAME = {
    0: "NORMAL",
    1: "SPIKE",
    2: "SEASONAL",
    3: "OUTLIER",
}

# ==========================================

if not os.path.exists(CLIENT_DATA):
    raise FileNotFoundError(f"No dataset at {CLIENT_DATA}")

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"No model weights found at {WEIGHTS_PATH}\n"
        "Make sure to save the final model weights after FL rounds.\n"
        "Ask ChatGPT for instructions if needed."
    )

# Load data
d = np.load(CLIENT_DATA)
X, y_true = d["X"], d["y"]
print(f"📂 Loaded dataset: {CLIENT_DATA} — {X.shape[0]} samples")

input_dim = X.shape[1]

# Load model
model = get_model_by_name(MODEL_ARCH, input_dim)

# Load FL global weights
weights = np.load(WEIGHTS_PATH, allow_pickle=True)["weights"]
model.set_weights(weights)
print("🔄 Loaded global model weights!")

# Fix required shapes for sequential models
if MODEL_ARCH == "conv1d":
    X_eval = X[..., np.newaxis]
elif MODEL_ARCH in ("rnn", "lstm", "gru"):
    X_eval = X[:, np.newaxis, :]
else:
    X_eval = X

# Predictions
y_proba = model.predict(X_eval, verbose=0)
y_pred = np.argmax(y_proba, axis=1)

# Classification report printed
print("\n📊 Classification Report (Global Model):")
print(classification_report(
    y_true, y_pred,
    target_names=[ANOMALY_NAME[i] for i in range(4)],
    zero_division=0
))

# Build table
df = pd.DataFrame({
    "True": [ANOMALY_NAME[i] for i in y_true],
    "Pred": [ANOMALY_NAME[i] for i in y_pred],
    "GC_norm": X[:, 0],
    "Day sin": X[:, 1],
    "Day cos": X[:, 2]
})

# Save results
df.to_csv(OUTPUT_CSV, index=False)
print(f"📎 Results exported: {OUTPUT_CSV}")

# Show first 20 rows nicely
print("\n📝 Sample Predictions")
print(df.head(20).to_string(index=False))
