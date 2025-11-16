
### `docs/README_flower.md`
```markdown
# Flower (Raspberry Pi) Guide

1. On server (workstation):
   - Generate certs: `bash flower_implementation/generate_certs.sh`
   - Copy `certs/client_crt.pem`, `certs/client_key.pem`, `certs/ca_crt.pem` to each Pi.
   - Start server: `bash scripts/run_flower_server.sh`

2. On each Raspberry Pi:
   - Install dependencies. Pi may need a special wheel for TensorFlow.
   - Place client partition file `processed/client_X.npz` on the Pi.
   - Place `certs/client_crt.pem`, `certs/client_key.pem`, `certs/ca_crt.pem` on the Pi.
   - Run client script (adjust server IP): `python3 flower_implementation/flower_client_ssl.py --server <server-ip>:8080 --data processed/client_X.npz --model ffnn`

Notes:
- Flower client script uses client cert for mutual TLS.
- Autoencoder flow assumes central pretraining (see TFF script).
