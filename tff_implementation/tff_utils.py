"""
Helper utilities for TFF script.
"""
import tensorflow as tf

def make_tf_dataset_from_numpy(X, y, batch_size=32, epochs=1, seq=False, conv=False):
    ds = tf.data.Dataset.from_tensor_slices((X.astype('float32'), y.astype('int32')))
    if conv:
        ds = ds.map(lambda x, yy: (tf.expand_dims(x, -1), yy))
    if seq:
        ds = ds.map(lambda x, yy: (tf.expand_dims(x, 0), yy))
    ds = ds.shuffle(1000).batch(batch_size).repeat(epochs)
    return ds
