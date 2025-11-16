#!/usr/bin/env bash
# run this on each Raspberry Pi. Replace SERVER_IP with your server's IP
SERVER_IP=SERVER_IP_HERE
CLIENT_INDEX=0  # set 0,1,2,3 per Pi

python3 flower_implementation/flower_client_ssl.py --server ${SERVER_IP}:8080 --data processed/client_${CLIENT_INDEX}.npz --model ffnn --ca certs/ca_crt.pem --cert certs/client_crt.pem --key certs/client_key.pem
