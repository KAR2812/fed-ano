#!/bin/bash
# ========================================
# 🚀 Federated Anomaly Detection (classic)
# Runs: Server → 3 direct Flower clients
# No SuperLink, no SuperNode.
# ========================================

set -e

source venv_arm/bin/activate

echo "🧩 Cleaning up old Flower processes..."
pkill -f flower-superlink || true
pkill -f flower-supernode || true
pkill -f flower-superexec || true
pkill -f flower_implementation.flower_server_ssl || true
sleep 1

echo "🚀 Starting classic Flower server on 0.0.0.0:8080..."
python flower_implementation/flower_server_ssl.py > logs_server.txt 2>&1 &
SERVER_PID=$!
sleep 4

echo "💻 Launching Client 0..."
python flower_implementation/flower_client_ssl.py \
  --server 127.0.0.1:8080 \
  --data processed/client_0.npz \
  --model lstm > logs_client0.txt 2>&1 &
sleep 2

echo "💻 Launching Client 1..."
python flower_implementation/flower_client_ssl.py \
  --server 127.0.0.1:8080 \
  --data processed/client_1.npz \
  --model lstm > logs_client1.txt 2>&1 &
sleep 2

echo "💻 Launching Client 2..."
python flower_implementation/flower_client_ssl.py \
  --server 127.0.0.1:8080 \
  --data processed/client_2.npz \
  --model lstm > logs_client2.txt 2>&1 &
sleep 2

echo ""
echo "✅ All components launched!"
echo "   ⚙️  Server PID:    $SERVER_PID"
echo "   💻 Clients:       3"
echo ""
echo "📜 Logs:"
echo "   logs_server.txt, logs_client0/1/2.txt"
echo ""
echo "🔎 Monitor training with:"
echo "   tail -f logs_server.txt"
