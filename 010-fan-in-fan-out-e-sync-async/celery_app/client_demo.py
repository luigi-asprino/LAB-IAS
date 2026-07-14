"""
Client dimostrativo — Esercizio Sez. 5

Confronta i due pattern chiamando l'API FastAPI: STESSA operazione
(inferenza), scala diversa.

1. Chiamata SINCRONA a /predict: UNA riga di feature, il client resta
   bloccato finché non arriva la risposta (richiesta/risposta classica).
2. Chiamata ASINCRONA a /batch-predict: un BATCH di migliaia di righe,
   il client riceve subito un task_id, e fa polling su
   /batch-predict/status/{task_id} finché lo stato non è SUCCESS.

Prerequisito: docker compose up (redis + celery-worker + fastapi-app)
Esecuzione:   python celery_app/client_demo.py
"""

import time
import requests

BASE_URL = "http://localhost:8000"
N_FEATURES = 20
BATCH_SIZE = 40  # righe nel batch — abbastanza per rendere visibile il polling


def demo_sync():
    print("\n--- PATTERN SINCRONO (/predict — una singola riga) ---")
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/predict", json={"features": [0.1] * N_FEATURES})
    elapsed = time.time() - t0
    print(f"Risposta ricevuta in {elapsed:.2f}s: {resp.json()}")


def demo_async():
    print(f"\n--- PATTERN ASINCRONO (/batch-predict — batch di {BATCH_SIZE} righe) ---")
    batch = [[0.1] * N_FEATURES for _ in range(BATCH_SIZE)]

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/batch-predict", json={"rows": batch})
    submit_elapsed = time.time() - t0
    task_id = resp.json()["task_id"]
    print(f"Batch sottomesso in {submit_elapsed:.2f}s (il chiamante NON ha aspettato l'elaborazione)")
    print(f"task_id = {task_id}")

    print("Inizio polling stato...")
    while True:
        status_resp = requests.get(f"{BASE_URL}/batch-predict/status/{task_id}")
        status = status_resp.json()
        print(f"  stato: {status['state']}  {status.get('meta', '')}")
        if status["state"] in ("SUCCESS", "FAILURE"):
            print(f"Risultato finale: {status}")
            break
        time.sleep(1)
    t1 = time.time()
    print(f"{t1 - t0:.2f}s {(t1-t0)/BATCH_SIZE:.2f} ")


if __name__ == "__main__":
    demo_sync()
    demo_async()
