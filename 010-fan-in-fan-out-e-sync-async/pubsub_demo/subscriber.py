"""
Subscriber — equivalente locale di Google Pub/Sub (subscribe side).

Riceve eventi "training completato" pubblicati su publisher.py e triggera
un'azione a valle: in questo caso una finta pipeline di "deployment"
(scrittura di un file che simula la promozione del modello in produzione).

Su GCP: questo sarebbe un Cloud Function/Cloud Run triggerato da una
subscription Pub/Sub push, che avvia una Vertex AI Pipeline di deployment.

Uso (avviare PRIMA del publisher, in un altro terminale):
    python pubsub_demo/subscriber.py
"""

import json
import os
import time
import redis

CHANNEL = "training-events"
DEPLOY_LOG = "data/deployments.log"


def trigger_deployment(evento: str):
    """Simula il trigger di una pipeline di deployment."""
    os.makedirs(os.path.dirname(DEPLOY_LOG), exist_ok=True)
    with open(DEPLOY_LOG, "a") as f:
        f.write(f"[DEPLOY TRIGGERATO] {time.strftime('%H:%M:%S')} — a seguito di: {evento}\n")
    print(f"  -> Deployment pipeline avviata per: {evento}")


def main():
    r = redis.Redis(host="localhost", port=6379, db=2)
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNEL)

    print(f"In ascolto sul canale '{CHANNEL}'... (Ctrl+C per uscire)")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue  # ignora il messaggio di conferma sottoscrizione
        evento = message["data"].decode("utf-8")
        print(f"Evento ricevuto: {evento}")
        trigger_deployment(evento)


if __name__ == "__main__":
    main()
