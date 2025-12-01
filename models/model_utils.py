import tensorflow as tf

# ---------------- HYBRID MODEL ---------------- #

import tensorflow as tf
from models.models_tff import (
    logistic_regression,
    ffnn_model,
    conv1d_model,
    autoencoder_models,
    rnn_model,
    lstm_model,
    gru_model,
    hybrid_cnn_bilstm,
)
def hybrid_cnn_bilstm_attention(input_dim: int) -> tf.keras.Model:

    inputs = tf.keras.Input(shape=(1, input_dim))  # Proper sequential shape

    # CNN Block
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, activation="relu")(inputs)

    # BiLSTM Block
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True)
    )(x)

    # Multi-head Self Attention
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=16
    )(x, x)
    x = tf.keras.layers.Add()([x, attention])  # residual
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="Hybrid_CNN_BiLSTM_Attn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


# ---------------- SIMPLE MODELS ---------------- #
def logistic_regression(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(4, activation="softmax")
    ])
    compile_model(model)
    return model

def ffnn_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation="softmax")
    ])
    compile_model(model)
    return model

def conv1d_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim, 1)),
        tf.keras.layers.Conv1D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    compile_model(model)
    return model

def rnn_model(input_dim, units=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    compile_model(model)
    return model

def lstm_model(input_dim, units=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    compile_model(model)
    return model

def gru_model(input_dim, units=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.GRU(units),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    compile_model(model)
    return model


# COMMON Compile Wrapper
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# ---------------- SELECTOR ---------------- #

def get_model_by_name(name: str, input_dim: int) -> tf.keras.Model:
    name = name.lower()

    if name == "logistic":
        return logistic_regression(input_dim)
    if name == "ffnn":
        return ffnn_model(input_dim)
    if name == "conv1d":
        return conv1d_model(input_dim)
    if name == "rnn":
        return rnn_model(input_dim)
    if name == "lstm":
        return lstm_model(input_dim)
    if name == "gru":
        return gru_model(input_dim)
    if name == "hybrid":
        return hybrid_cnn_bilstm(input_dim)

    raise ValueError(f"Unknown model name: {name}")