"""
Keras model definitions used by TFF and Flower clients.
All models return compiled tf.keras.Model objects.
"""
import tensorflow as tf

def logistic_regression(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(4, activation="softmax")

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

def ffnn_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation="softmax")

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

def conv1d_model(input_dim):
    # expects input shape (input_dim, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim, 1)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation="softmax")

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

def autoencoder_models(input_dim):
    """
    Returns: autoencoder, encoder, classifier (compiled)
    Autoencoder compiled with MSE; classifier compiled with binary_crossentropy.
    """
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(inp)
    bottleneck = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(bottleneck)
    out = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    autoencoder = tf.keras.Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    encoder = tf.keras.Model(inputs=inp, outputs=bottleneck)

    classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation="softmax")

    ])
    classifier.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy",
                       metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')])
    return autoencoder, encoder, classifier

def rnn_model(input_dim, units=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.Dense(4, activation="softmax")

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

def lstm_model(input_dim, units=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(4, activation="softmax")

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

def gru_model(input_dim, units=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.GRU(units),
        tf.keras.layers.Dense(4, activation="softmax")

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model
