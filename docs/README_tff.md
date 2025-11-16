# TFF Simulation Guide

1. Ensure `processed/` contains client_*.npz and test_set.npz (use preprocessing script).
2. Install requirements: `pip install -r requirements.txt`
3. Run TFF training:
   ```bash
   python tff_implementation/tff_train.py --model ffnn --rounds 50 --clients_per_round 10
