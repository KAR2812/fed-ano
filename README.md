## Federated Anomaly Detection in Smart Grids
Implementation of the IEEE paper using TensorFlow Federated and Flower.
# Federated Anomaly Detection in Smart Grids

Repository implementing "Distributed Anomaly Detection in Smart Grids: A Federated Learning-Based Approach" (IEEE Access, 2023).

This repo contains:
- Preprocessing for the Ausgrid dataset (SMOTE-NC, time features).
- Keras model architectures for Logistic Regression, FFNN, 1D-CNN, Autoencoder+Classifier, RNN, LSTM, GRU.
- TensorFlow Federated (TFF) implementation for simulation experiments.
- Flower-based implementation (server + client) with TLS support for Raspberry Pi experiments.
- Scripts to run experiments and collect results.

**Structure**: See repository tree. Place raw datasets in `data/` and generate processed client partitions using `preprocessing/preprocess_ausgrid.py`.

