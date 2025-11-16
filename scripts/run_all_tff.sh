#!/usr/bin/env bash
# Run TFF training for each model sequentially (simulation mode).
# Adjust paths and args as needed.

MODELS=("logistic" "ffnn" "conv1d" "rnn" "lstm" "gru")
for m in "${MODELS[@]}"; do
  echo "Running TFF train for model $m"
  python3 tff_implementation/tff_train.py --model $m --rounds 20 --clients_per_round 10 --batch_size 64 --local_epochs 1
done

# Autoencoder separately (with pretraining)
echo "Running TFF train for autoencoder workflow"
python3 tff_implementation/tff_train.py --model autoencoder --rounds 20 --clients_per_round 10 --batch_size 128 --local_epochs 1
