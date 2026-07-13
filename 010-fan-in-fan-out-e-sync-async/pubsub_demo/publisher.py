"""
Publisher — equivalente locale di Google Pub/Sub (publish side).

Google Pub/Sub                     Redis Pub/Sub (locale)
------------------------------------------------------------------
Topic                               Canale Redis (PUBLISH/SUBSCRIBE)
Publisher.publish(topic, message)   redis.publish(channel, message)
Subscription (>=1 subscriber)       Più client SUBSCRIBE sullo stesso canale
Consegna "at-least-once", durevole  Consegna "fire-and-forget", NON durevole
                                     (se nessun subscriber è in ascolto, il
                                      messaggio va perso — differenza chiave
                                      da sottolineare in aula)

Differenza concettuale con Cloud Tasks (vista in Sez. 5):
  - Cloud Tasks: il mittente sa esattamente chi riceverà il task (push diretto
    a un endpoint specifico) ed è pensato per UN consumer.
  - Pub/Sub: il publisher non sa (né gli interessa) chi sono i subscriber;
    più servizi possono reagire allo stesso evento (fan-out a più consumer).

Uso:
    python pubsub_demo/publisher.py
"""

import time
import redis

CHANNEL = "training-events"


def main():
    r = redis.Redis(host="localhost", port=6379, db=2)

    eventi = [
        "job_seed_1 completato con loss finale 0.3421",
        "job_seed_2 completato con loss finale 0.2987",
        "job_seed_3 completato con loss finale 0.3105",
    ]

    for evento in eventi:
        n_subscribers = r.publish(CHANNEL, evento)
        print(f"Pubblicato su '{CHANNEL}': {evento}  (ricevuto da {n_subscribers} subscriber)")
        time.sleep(2)


if __name__ == "__main__":
    main()
