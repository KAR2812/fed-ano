"""
Evaluate a saved Keras model (if you saved weights externally) on processed/test_set.npz.
This is a small utility used to evaluate Keras models on the test set.
"""
import numpy as np
import argparse
from models.models_tff import ffnn_model, logistic_regression, conv1d_model, rnn_model, lstm_model, gru_model

MODEL_MAP = {
    'logistic': logistic_regression,
    'ffnn': ffnn_model,
    'conv1d': conv1d_model,
    'rnn': rnn_model,
    'lstm': lstm_model,
    'gru': gru_model
}

def load_test(processed_dir="processed"):
    d = np.load(processed_dir + "/test_set.npz")
    return d['X'], d['y']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ffnn")
    parser.add_argument("--processed", default="processed")
    parser.add_argument("--weights", default=None, help="Path to saved Keras weights (.h5)")
    args = parser.parse_args()

    X_test, y_test = load_test(args.processed)
    input_dim = X_test.shape[1]
    model_fn = MODEL_MAP.get(args.model)
    if not model_fn:
        raise ValueError("Unknown model")
    model = model_fn(input_dim)
    if args.weights:
        model.load_weights(args.weights)
    if args.model == 'conv1d':
        X_test = X_test[..., np.newaxis]
    if args.model in ('rnn','lstm','gru'):
        X_test = X_test[:, np.newaxis, :]
    res = model.evaluate(X_test, y_test, verbose=1)
    print("Test results:", res)

if __name__ == "__main__":
    main()
