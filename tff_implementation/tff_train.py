"""
TFF training script (simulation mode). Uses processed/ client partitions generated
by preprocessing/preprocess_ausgrid.py. Runs FedAvg with tff.learning.

Usage:
    python tff_implementation/tff_train.py --model ffnn --rounds 50 --clients_per_round 10 --clients_dir ../processed
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from glob import glob
import joblib
from models.models_tff import (
    logistic_regression, ffnn_model, conv1d_model,
    autoencoder_models, rnn_model, lstm_model, gru_model
)
from tff_implementation.tff_utils import make_tf_dataset_from_numpy

def load_client_npz_list(processed_dir):
    files = sorted(glob(os.path.join(processed_dir, "client_*.npz")))
    client_data = []
    for f in files:
        d = np.load(f)
        client_data.append((d['X'], d['y']))
    return client_data

def build_keras_model_by_name(name, input_dim):
    name = name.lower()
    if name == 'logistic':
        return logistic_regression(input_dim)
    if name == 'ffnn':
        return ffnn_model(input_dim)
    if name == 'conv1d':
        return conv1d_model(input_dim)
    if name == 'autoencoder':
        # handled separately in flow
        autoenc, encoder, classifier = autoencoder_models(input_dim)
        return (autoenc, encoder, classifier)
    if name == 'rnn':
        return rnn_model(input_dim)
    if name == 'lstm':
        return lstm_model(input_dim)
    if name == 'gru':
        return gru_model(input_dim)
    raise ValueError("Unknown model name")

def client_data_to_tf_dataset(client_tuple, batch_size, epochs, model_name):
    X, y = client_tuple
    if model_name == 'conv1d':
        # conv expects shape (input_dim,1) per sample
        ds = make_tf_dataset_from_numpy(X, y, batch_size=batch_size, epochs=epochs, conv=True)
    elif model_name in ('rnn','lstm','gru'):
        ds = make_tf_dataset_from_numpy(X, y, batch_size=batch_size, epochs=epochs, seq=True)
    else:
        ds = make_tf_dataset_from_numpy(X, y, batch_size=batch_size, epochs=epochs)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="../processed", help="Processed data dir containing client_*.npz")
    parser.add_argument("--model", default="ffnn", choices=['logistic','ffnn','conv1d','autoencoder','rnn','lstm','gru'])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients_per_round", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()

    processed_dir = os.path.abspath(args.processed)
    client_data = load_client_npz_list(processed_dir)
    if len(client_data) == 0:
        raise RuntimeError(f"No client partitions found in {processed_dir}. Run preprocessing first.")

    # determine input dim from first client
    input_dim = client_data[0][0].shape[1]
    print("Input dim:", input_dim)

    # Build TFF model_fn
    def model_fn():
        # Build a fresh keras model each call - tff expects this
        if args.model == 'autoencoder':
            # For autoencoder workflow we'll pretrain autoencoder centrally later
            # here return classifier architecture (shape depends on encoded dim 32)
            dummy = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(32,)),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            return tff.learning.from_keras_model(
                keras_model=dummy,
                input_spec=(tf.TensorSpec(shape=[None, 32], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32)),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
        else:
            # regular models
            if args.model == 'logistic':
                keras_model = logistic_regression(input_dim)
                input_spec = (tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32))
            elif args.model == 'ffnn':
                keras_model = ffnn_model(input_dim)
                input_spec = (tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32))
            elif args.model == 'conv1d':
                keras_model = conv1d_model(input_dim)
                input_spec = (tf.TensorSpec(shape=[None, input_dim, 1], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32))
            elif args.model == 'rnn':
                keras_model = rnn_model(input_dim)
                input_spec = (tf.TensorSpec(shape=[None, 1, input_dim], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32))
            elif args.model == 'lstm':
                keras_model = lstm_model(input_dim)
                input_spec = (tf.TensorSpec(shape=[None, 1, input_dim], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32))
            elif args.model == 'gru':
                keras_model = gru_model(input_dim)
                input_spec = (tf.TensorSpec(shape=[None, 1, input_dim], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32))
            else:
                raise ValueError("Unknown model")
            return tff.learning.from_keras_model(
                keras_model=keras_model,
                input_spec=input_spec,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(1e-3),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
    )

    state = iterative_process.initialize()

    # optional: autoencoder central pretrain if requested
    if args.model == 'autoencoder':
        # load autoencoder & encoder & classifier and pretrain centrally
        from models.models_tff import autoencoder_models
        autoenc, encoder, classifier = autoencoder_models(input_dim)
        # build aggregated X_train from all client partitions (careful memory)
        Xagg = np.vstack([c[0] for c in client_data])
        print("Pretraining autoencoder centrally on aggregated data:", Xagg.shape)
        autoenc.fit(Xagg, Xagg, epochs=20, batch_size=128, verbose=1)
        # Now generate new client datasets of encoded features
        encoded_clients = []
        for Xc, yc in client_data:
            Z = encoder.predict(Xc, batch_size=128)
            encoded_clients.append((Z, yc))
        # Replace client_data with encoded ones for federated classifier training
        client_data = encoded_clients

    # convert client_data into tf.data.Dataset objects (one per client)
    client_datasets = [client_data_to_tf_dataset(cd, args.batch_size, args.local_epochs, args.model) for cd in client_data]

    # run rounds
    num_clients = len(client_datasets)
    for r in range(1, args.rounds+1):
        selected = np.random.choice(range(num_clients), size=min(args.clients_per_round, num_clients), replace=False)
        federated_train_data = [client_datasets[i] for i in selected]
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f"Round {r} metrics: {metrics}")

        # Evaluate on global test set (single-client) every 5 rounds
        if r % 5 == 0 or r == args.rounds:
            # build evaluation computation and run
            eval_comp = tff.learning.build_federated_evaluation(model_fn)
            # create tf.dataset for test set
            test_npz = np.load(os.path.join(processed_dir if 'processed_dir' in locals() else args.processed, "test_set.npz"))
            X_test = test_npz['X']
            y_test = test_npz['y']
            if args.model == 'conv1d':
                test_ds = make_tf_dataset_from_numpy(X_test, y_test, batch_size=args.batch_size, epochs=1, conv=True)
            elif args.model in ('rnn','lstm','gru'):
                test_ds = make_tf_dataset_from_numpy(X_test, y_test, batch_size=args.batch_size, epochs=1, seq=True)
            elif args.model == 'autoencoder':
                # if autoencoder, encode test set first
                Ztest = encoder.predict(X_test, batch_size=256)
                test_ds = make_tf_dataset_from_numpy(Ztest, y_test, batch_size=args.batch_size, epochs=1)
            else:
                test_ds = make_tf_dataset_from_numpy(X_test, y_test, batch_size=args.batch_size, epochs=1)

            metrics_eval = eval_comp(state.model, [test_ds])
            print(f"Evaluation metrics at round {r}: {metrics_eval}")

    print("Training finished.")
