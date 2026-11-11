"""
Task asincroni — equivalente locale di Cloud Tasks.

Cloud Tasks (GCP)                  Celery + Redis (locale)
------------------------------------------------------------------
Coda di task                       Broker Redis (db 0)
Task HTTP verso Cloud Run/Function Task Python eseguito da un worker
Stato del task (PENDING/DONE)      AsyncResult (backend Redis, db 1)
Retry automatico configurabile     `autoretry_for` / `retry_backoff`

Il pattern concettuale è identico: il chiamante NON aspetta il
completamento del lavoro, riceve subito un identificativo e può
interrogare lo stato in un secondo momento (polling).

Nota didattica: sync e async qui sono la STESSA operazione (inferenza)
a scale diverse, non due operazioni diverse. Un client che chiede una
predizione singola si aspetta una risposta nella stessa richiesta HTTP
(sincrono, /predict). Un client che sottomette un batch di migliaia di
righe (es. da un file CSV) sa che la risposta richiedera' tempo e non
vuole restare bloccato: sottomette il job e fa polling (asincrono,
/batch-predict). Il training vero e proprio, invece, di norma non parte
da una richiesta client via HTTP: e' schedulato (cron, trigger su nuovi
dati, pipeline CI/CD) o lanciato da Vertex AI Pipelines stessa — per
questo non e' usato come esempio di endpoint asincrono qui.
"""

import os
import time

import torch
from celery import Celery

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

app = Celery("lezione10", broker=BROKER_URL, backend=RESULT_BACKEND)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)


def _build_inference_model():
    """Stesso modello usato sia dall'inferenza singola sia dal batch: cambia solo la scala."""
    torch.manual_seed(0)
    return torch.nn.Linear(20, 2)


@app.task(bind=True, name="batch_predict_async")
def batch_predict_async(self, batch: list, batch_size_log_every: int = 200):
    """
    Simula una batch prediction su un batch grande (es. migliaia di righe
    caricate da un client da un file CSV/Parquet).
    Rappresenta il caso d'uso ASINCRONO: stessa operazione di
    inference_job_sync, ma su una scala per cui il chiamante NON vuole
    restare bloccato nella stessa richiesta HTTP.

    In produzione su GCP: questo corrisponde a un Vertex AI Batch
    Prediction Job, sottomesso da un client e monitorato via polling.
    """
    model = _build_inference_model()
    x = torch.tensor(batch, dtype=torch.float32)

    n = x.shape[0]
    predictions = []
    with torch.no_grad():
        for i in range(n):
            self.update_state(state="PROGRESS", meta={"processed": i + 1, "total": n})
            logits = model(x[i].unsqueeze(0))
            predictions.append(torch.argmax(logits, dim=1).item())
            time.sleep(0.05)  # simula il costo per riga su un batch grande

    # Pubblica un evento "batch prediction completata" su Redis Pub/Sub
    # (vedi pubsub_demo/publisher.py per lo stesso pattern standalone)
    import redis
    r = redis.Redis(host=BROKER_URL.split("//")[1].split(":")[0], port=6379, db=2)
    r.publish("batch-prediction-events", f"batch di {n} righe completato")

    return {"n_rows": n, "predictions": predictions}


@app.task(name="inference_job_sync")
def inference_job_sync(payload: list):
    """
    Simula un'inferenza singola: rappresenta il caso d'uso SINCRONO.
    Nella demo FastAPI questo viene chiamato direttamente (non via .delay())
    perché il chiamante SI ASPETTA una risposta immediata.
    """
    model = _build_inference_model()
    x = torch.tensor(payload, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return {"prediction": pred}

