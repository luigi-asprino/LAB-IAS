#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Avvia N shard localmente su porte consecutive
# =============================================================================
#
# Uso:
#   ./run_local.sh           # 2 shard su porte 8080, 8081
#   ./run_local.sh 4         # 4 shard su porte 8080, 8081, 8082, 8083
#   N_SHARDS=3 ./run_local.sh
#
# Ogni shard viene avviato in background. Lo script stampa i PID e
# al termine pulisce tutti i processi con CTRL+C.
#
# Prerequisiti:
#   pip install -r requirements.txt
#   (Flask, requests, torch già installati)
#
# Verifica rapida dopo l'avvio (aspetta ~2s per il warmup):
#   curl http://localhost:8080/health | python3 -m json.tool
#   python3 client.py --mode single
# =============================================================================

set -euo pipefail

N_SHARDS=${1:-${N_SHARDS:-2}}
BASE_PORT=${BASE_PORT:-8080}

# Parametri del modello — devono essere uguali per tutti gli shard
TOTAL_LAYERS=${TOTAL_LAYERS:-8}
D_MODEL=${D_MODEL:-256}

echo "=== Avvio pipeline con ${N_SHARDS} shard ==="
echo "    Porte: ${BASE_PORT} – $((BASE_PORT + N_SHARDS - 1))"
echo "    Modello: total_layers=${TOTAL_LAYERS}, d_model=${D_MODEL}"
echo ""

PIDS=()

# ──────────────────────────────────────────────────────────────────────────────
# Cleanup: uccide tutti gli shard quando lo script termina (CTRL+C o exit)
# ──────────────────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "=== Arresto di tutti gli shard ==="
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  Terminato PID $pid"
        fi
    done
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTANTE: gli shard devono essere avviati in ordine INVERSO.
#
# Motivazione: lo shard 0 ha bisogno di conoscere l'URL dello shard 1
# al momento del suo avvio (variabile NEXT_SHARD_URL). Se avviassimo lo
# shard 0 per primo, NEXT_SHARD_URL punterebbe a un server non ancora attivo.
# Le prime richieste fallirebbero con "Connection refused".
#
# Avviando prima l'ultimo shard e procedendo verso lo shard 0, garantiamo
# che ogni shard sia pronto prima che il suo predecessore inizi a inoltrargli
# richieste.
# ──────────────────────────────────────────────────────────────────────────────
NEXT_URL=""

for (( sid=N_SHARDS-1; sid>=0; sid-- )); do
    PORT=$((BASE_PORT + sid))

    # Costruisce il comando di avvio con le variabili d'ambiente corrette
    env_vars=(
        "SHARD_ID=${sid}"
        "N_SHARDS=${N_SHARDS}"
        "PORT=${PORT}"
        "TOTAL_LAYERS=${TOTAL_LAYERS}"
        "D_MODEL=${D_MODEL}"
    )

    if [[ -n "$NEXT_URL" ]]; then
        env_vars+=("NEXT_SHARD_URL=${NEXT_URL}")
    fi

    echo "  Avvio shard ${sid} su porta ${PORT}..."
    if [[ -n "$NEXT_URL" ]]; then
        echo "    → punta a ${NEXT_URL}"
    else
        echo "    → è l'ultimo shard (nessun next)"
    fi

    # Avvia il processo in background, reindirizza stdout/stderr su file di log
    LOG_FILE="/tmp/shard_${sid}.log"
    env "${env_vars[@]}" python3 shard_server.py > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    echo "    PID: $PID  |  Log: $LOG_FILE"

    # Il prossimo shard (indice più basso) punterà a questo shard
    NEXT_URL="http://localhost:${PORT}"

    # Breve attesa per dare tempo a Flask di avviarsi prima di lanciare
    # il processo successivo (evita race condition nei log)
    sleep 1
done

echo ""
echo "=== Pipeline avviata ==="
echo ""
echo "    Punto di ingresso (shard 0): http://localhost:${BASE_PORT}"
echo ""
echo "    Health check:"
for (( sid=0; sid<N_SHARDS; sid++ )); do
    echo "      curl http://localhost:$((BASE_PORT + sid))/health"
done
echo ""
echo "    Test rapido:"
echo "      python3 client.py --mode single"
echo ""
echo "    Benchmark:"
echo "      python3 client.py --mode bench --n 20"
echo ""
echo "    Log in tempo reale:"
echo "      tail -f /tmp/shard_0.log /tmp/shard_1.log"
echo ""
echo "Premi CTRL+C per arrestare tutti gli shard."
echo ""

# Aspetta che tutti i processi in background terminino
# (rimarrà in attesa finché non viene inviato SIGINT)
wait
