"""
API FastAPI — dimostra il contrasto SINCRONO vs ASINCRONO.

Stesso dominio (inferenza), scala diversa:

Endpoint sincrono   /predict                  -> predizione singola, risposta
                                                  immediata (blocca il chiamante)
Endpoint asincrono  /batch-predict            -> predizione su un batch grande,
                                                  ritorna subito un task_id
                                                  (non blocca)
Polling stato       /batch-predict/status/{id} -> il client interroga quando vuole

Avvio locale (senza Docker):
    uvicorn celery_app.main:app --reload
Richiede Redis in esecuzione su localhost:6379 (docker compose up redis)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from celery.result import AsyncResult

from celery_app.tasks import app as celery_app, batch_predict_async, inference_job_sync

app = FastAPI(title="Lezione 10 — Sync/Async Demo")


class InferenceRequest(BaseModel):
    features: list[float]  # vettore di 20 feature (una singola riga)


class BatchPredictRequest(BaseModel):
    rows: list[list[float]]  # batch di N vettori di 20 feature ciascuno


@app.post("/predict")
def predict(req: InferenceRequest):
    """
    PATTERN SINCRONO.
    Il client invia UNA riga di feature e aspetta la risposta nella stessa
    richiesta HTTP. Adatto quando la latenza è bassa e prevedibile
    (richiesta/risposta classica, es. inferenza online su un singolo item).
    """
    result = inference_job_sync(req.features)
    return {"pattern": "sync", **result}


@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest):
    """
    PATTERN ASINCRONO.
    Stessa operazione di /predict (inferenza), ma su un batch di N righe:
    il client sottomette il batch e riceve SUBITO un task_id, senza restare
    bloccato per tutta la durata dell'elaborazione.
    Corrisponde a: client -> Cloud Tasks -> Cloud Run, oppure a un Vertex AI
    Batch Prediction Job sottomesso e monitorato via polling.
    """
    task = batch_predict_async.delay(batch=req.rows)
    return {"pattern": "async", "task_id": task.id, "status": "PENDING", "n_rows": len(req.rows)}


@app.get("/batch-predict/status/{task_id}")
def batch_predict_status(task_id: str):
    """
    Il client fa polling su questo endpoint per conoscere lo stato del batch.
    Stati possibili: PENDING, PROGRESS, SUCCESS, FAILURE.
    """
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": result.state}
    if result.state == "PROGRESS":
        response["meta"] = result.info
    elif result.state == "SUCCESS":
        response["result"] = result.result
    elif result.state == "FAILURE":
        response["error"] = str(result.info)
    return response
