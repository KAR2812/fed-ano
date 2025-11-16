#!/usr/bin/env bash
cd flower_implementation
python3 flower_server_ssl.py --host 0.0.0.0 --port 8080 --rounds 10 --min_available 4 --cert ../certs/server_crt.pem --key ../certs/server_key.pem --client_ca ../certs/ca_crt.pem
