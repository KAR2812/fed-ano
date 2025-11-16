# models/model_utils.py

import tensorflow as tf


def build_logistic(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    outputs = tf.keras.layers.Dense(4, activation="softmax")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="logistic_multiclass")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_ffnn(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ffnn_multiclass")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_conv1d(input_dim: int) -> tf.keras.Model:
    # Input shape: (timesteps=input_dim, channels=1)
    inputs = tf.keras.Input(shape=(input_dim, 1))
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="conv1d_multiclass")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_rnn(input_dim: int) -> tf.keras.Model:
    # Input shape: (timesteps=1, features=input_dim)
    inputs = tf.keras.Input(shape=(1, input_dim))
    x = tf.keras.layers.SimpleRNN(32, return_sequences=False)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rnn_multiclass")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(1, input_dim))
    x = tf.keras.layers.LSTM(32, return_sequences=False)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_multiclass")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(1, input_dim))
    x = tf.keras.layers.GRU(32, return_sequences=False)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_multiclass")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model_by_name(name: str, input_dim: int) -> tf.keras.Model:
    name = name.lower()
    if name == "logistic":
        return build_logistic(input_dim)
    if name == "ffnn":
        return build_ffnn(input_dim)
    if name == "conv1d":
        return build_conv1d(input_dim)
    if name == "rnn":
        return build_rnn(input_dim)
    if name == "lstm":
        return build_lstm(input_dim)
    if name == "gru":
        return build_gru(input_dim)
    raise ValueError(f"Unknown model name: {name}")
