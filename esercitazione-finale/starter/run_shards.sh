#!/usr/bin/env bash
# Avvia i 3 shard in background per l'Esercizio 2.
# Uso: ./run_shards.sh   (poi python client.py)  ·  ./run_shards.sh stop   per fermarli

set -e
cd "$(dirname "$0")"

if [ "$1" == "stop" ]; then
  pkill -f "shard_server.py --port 800" 2>/dev/null || true
  echo "Shard fermati."
  exit 0
fi

python3 shard_server.py --port 8001 --shard-id 0 --latency 0.30 & echo $! > .shard0.pid
python3 shard_server.py --port 8002 --shard-id 1 --latency 0.50 & echo $! > .shard1.pid
python3 shard_server.py --port 8003 --shard-id 2 --latency 0.20 & echo $! > .shard2.pid

sleep 1
echo "3 shard avviati sulle porte 8001, 8002, 8003."
echo "Ferma con: ./run_shards.sh stop"
