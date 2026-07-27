"""
Esercizio 2 — Shard di un servizio di inferenza (INFRASTRUTTURA GIA' PRONTA)
=============================================================================

Questo script simula UNO shard del modello (Lezione 9, sharded service
pattern): espone un endpoint /predict che restituisce un risultato
parziale dopo una latenza simulata (rappresenta il tempo di calcolo su
quella porzione del modello).

In aula i 3 shard sono gia' pacchettizzati come container e avviati; questo
file viene fornito "as-is" per poterli far girare anche in locale durante
l'esercitazione. Gli studenti NON devono modificare questo file: il loro
lavoro e' scrivere il client (client.py) che li interroga.

Avvio di 3 shard in locale (in 3 terminali separati, o con lo script
run_shards.sh):
    python shard_server.py --port 8001 --shard-id 0 --latency 0.30
    python shard_server.py --port 8002 --shard-id 1 --latency 0.50
    python shard_server.py --port 8003 --shard-id 2 --latency 0.20
"""

import argparse
import time

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    x = payload.get("x", [])

    # Simula il tempo di calcolo di questo shard (diverso per ogni shard,
    # come ci si aspetta in un sistema reale con shard non omogenei).
    time.sleep(app.config["LATENCY"])

    shard_id = app.config["SHARD_ID"]
    partial = sum(x) * (shard_id + 1)  # calcolo fittizio, solo a scopo didattico
    return jsonify({"shard_id": shard_id, "partial": partial})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "shard_id": app.config["SHARD_ID"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--latency", type=float, default=0.2,
                         help="Latenza simulata dello shard, in secondi")
    args = parser.parse_args()

    app.config["SHARD_ID"] = args.shard_id
    app.config["LATENCY"] = args.latency
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
