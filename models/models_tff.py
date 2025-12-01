"""
Keras model definitions used by TFF and Flower clients.
All models return compiled tf.keras.Model objects.
"""

import tensorflow as tf


# ---------- COMMON METRICS ---------- #

def _clf_compile(model):
    """Compile a 4-class classifier with stable, sparse metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


# ---------- BASELINE MODELS ---------- #

def logistic_regression(input_dim: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return _clf_compile(model)


def ffnn_model(input_dim: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return _clf_compile(model)


def conv1d_model(input_dim: int):
    # expects input shape (input_dim, 1)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim, 1)),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.GlobalMaxPool1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return _clf_compile(model)


def autoencoder_models(input_dim: int):
    """
    Returns: autoencoder, encoder, classifier (compiled)
    Autoencoder compiled with MSE; classifier compiled as 4-class sparse.
    """
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    bottleneck = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(bottleneck)
    out = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    autoencoder = tf.keras.Model(inputs=inp, outputs=out)
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
    )

    encoder = tf.keras.Model(inputs=inp, outputs=bottleneck)

    classifier = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    _clf_compile(classifier)

    return autoencoder, encoder, classifier


def rnn_model(input_dim: int, units: int = 32):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1, input_dim)),
            tf.keras.layers.SimpleRNN(units),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return _clf_compile(model)


def lstm_model(input_dim: int, units: int = 64):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1, input_dim)),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return _clf_compile(model)


def gru_model(input_dim: int, units: int = 64):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1, input_dim)),
            tf.keras.layers.GRU(units),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return _clf_compile(model)


# ---------- HYBRID MODEL (CNN + BiLSTM) ---------- #
# Hybrid = CNN (local patterns) + BiLSTM (temporal memory)


def hybrid_cnn_bilstm(input_dim: int) -> tf.keras.Model:
    """
    Hybrid model:
      - Input: (batch, 1, input_dim)  -> same as RNN/LSTM/GRU
      - Block 1: 1D CNN over the feature dimension
      - Block 2: BiLSTM over the same sequence
      - attention: concatenate CNN + BiLSTM features
      - Output: 4-class softmax
    """

    inputs = tf.keras.Input(shape=(1, input_dim))  # (batch, time=1, features=3)

    # CNN path: treat the 3 features as a single 1D "patch"
    cnn = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=1,
        activation="relu",
        name="hybrid_cnn_block",
    )(inputs)
    cnn = tf.keras.layers.GlobalAveragePooling1D()(cnn)  # (batch, 32)

    # BiLSTM path
    rnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False),
        name="hybrid_bilstm_block",
    )(inputs)  # (batch, 64)

    # Fuse
    x = tf.keras.layers.Concatenate()([cnn, rnn])  # (batch, 96)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Hybrid_CNN_BiLSTM",
    )

    _clf_compile(model)
    return model
