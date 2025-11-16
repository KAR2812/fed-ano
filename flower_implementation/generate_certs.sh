#!/usr/bin/env bash
set -e

mkdir -p certs && cd certs

echo "Generating CA key and cert..."
openssl genrsa -out ca_key.pem 4096
openssl req -x509 -new -nodes -key ca_key.pem -sha256 -days 3650 -out ca_crt.pem -subj "/CN=Local-CA"

echo "Generating server key and CSR..."
openssl genrsa -out server_key.pem 4096
openssl req -new -key server_key.pem -out server.csr -subj "/CN=fl-server.local"

echo "Signing server cert with CA..."
openssl x509 -req -in server.csr -CA ca_crt.pem -CAkey ca_key.pem -CAcreateserial -out server_crt.pem -days 365 -sha256

echo "Generating client key and CSR..."
openssl genrsa -out client_key.pem 2048
openssl req -new -key client_key.pem -out client.csr -subj "/CN=pi-client-1"

echo "Signing client cert with CA..."
openssl x509 -req -in client.csr -CA ca_crt.pem -CAkey ca_key.pem -CAcreateserial -out client_crt.pem -days 365 -sha256

echo "Certificates generated in $(pwd). Copy client_crt.pem, client_key.pem, ca_crt.pem to each Raspberry Pi."
