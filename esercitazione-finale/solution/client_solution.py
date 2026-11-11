"""
Esercizio 2 — Client di orchestrazione per un servizio sharded (SOLUZIONE)
=============================================================================

Implementazione completa dei punti (a), (b), (d), con la risposta al punto
(c) in fondo al file.

Prima di lanciare, avviare i 3 shard:
    ./run_shards.sh
Poi:
    python client_solution.py
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import requests

SHARD_URLS = [
    "http://127.0.0.1:8001/predict",
    "http://127.0.0.1:8002/predict",
    "http://127.0.0.1:8003/predict",
]

TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# (a) Fan-in: combinare i risultati parziali degli shard — Lezione 9/10
# ---------------------------------------------------------------------------

def combine(partials):
    """Aggrega le risposte parziali in un unico risultato.

    Gestisce il caso in cui uno o più shard non abbiano risposto
    (partial is None, punto d): il risultato finale si basa solo sugli
    shard disponibili, e riportiamo quanti erano attesi vs quanti hanno
    risposto, cosi' il chiamante può decidere se la risposta è comunque
    accettabile (degradazione controllata, non un errore secco).
    """
    valid = [p["partial"] for p in partials if p is not None]
    return {
        "result": sum(valid),
        "shards_ok": len(valid),
        "shards_total": len(partials),
    }


# ---------------------------------------------------------------------------
# (b) Variante 1 — sincrona sequenziale (naive)
# ---------------------------------------------------------------------------
#
# Ogni chiamata attende il completamento della precedente prima di partire:
# la latenza totale e' la SOMMA delle latenze di tutti gli shard. È la
# baseline "cattiva" con cui confrontare le altre due varianti.

def call_shard_sync(url: str, payload: dict):
    """(d) Timeout esplicito + fallback: se lo shard non risponde in tempo
    o restituisce un errore, torniamo None invece di propagare
    l'eccezione, cosi' combine() puo' comunque produrre un risultato
    parziale con gli shard rimasti."""
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def predict_sync_sequential(payload: dict):
    partials = [call_shard_sync(url, payload) for url in SHARD_URLS]
    return combine(partials)


# ---------------------------------------------------------------------------
# (b) Variante 2 — sincrona con thread pool
# ---------------------------------------------------------------------------
#
# Le chiamate bloccanti vengono distribuite su thread OS distinti: partono
# tutte "in parallelo" e la latenza totale e' vicina al MAX delle latenze
# dei singoli shard, non alla somma. Il costo e' un thread OS per shard
# (poco per 3 shard, ma cresce se gli shard sono centinaia).

def predict_sync_threadpool(payload: dict):
    with ThreadPoolExecutor(max_workers=len(SHARD_URLS)) as pool:
        partials = list(pool.map(lambda url: call_shard_sync(url, payload), SHARD_URLS))
    return combine(partials)


# ---------------------------------------------------------------------------
# (b) Variante 3 — asincrona con asyncio
# ---------------------------------------------------------------------------
#
# Stesso risultato "logico" del thread pool (fan-out concorrente, latenza
# ~ MAX), ma le richieste sono multiplexate su un singolo thread tramite
# l'event loop: nessun thread OS aggiuntivo per shard, quindi scala molto
# meglio quando il numero di shard/richieste concorrenti cresce.

async def call_shard_async(session: aiohttp.ClientSession, url: str, payload: dict):
    try:
        async with session.post(url, json=payload, timeout=TIMEOUT_S) as resp:
            resp.raise_for_status()
            return await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def predict_async(payload: dict):
    async with aiohttp.ClientSession() as session:
        partials = await asyncio.gather(
            *[call_shard_async(session, url, payload) for url in SHARD_URLS]
        )
    return combine(partials)


# ---------------------------------------------------------------------------
# Benchmark — confronta le tre varianti
# ---------------------------------------------------------------------------

def benchmark(n_requests: int = 10):
    payload = {"x": [1, 2, 3, 4, 5]}

    print(f"Esempio di risposta: {predict_sync_sequential(payload)}\n")

    for name, fn in [
        ("sync sequenziale", lambda: predict_sync_sequential(payload)),
        ("sync thread pool", lambda: predict_sync_threadpool(payload)),
    ]:
        t0 = time.perf_counter()
        for _ in range(n_requests):
            fn()
        elapsed = time.perf_counter() - t0
        print(f"{name:20s}: {elapsed:6.3f}s totali, {elapsed / n_requests * 1000:6.1f} ms/richiesta")

    async def run_async():
        for _ in range(n_requests):
            await predict_async(payload)

    t0 = time.perf_counter()
    asyncio.run(run_async())
    elapsed = time.perf_counter() - t0
    print(f"{'async asyncio':20s}: {elapsed:6.3f}s totali, {elapsed / n_requests * 1000:6.1f} ms/richiesta")

    print(
        "\nCon shard di latenza 0.30s/0.50s/0.20s: la sequenziale dovrebbe "
        "avvicinarsi a ~1.0s/richiesta (somma), thread pool e async a "
        "~0.50s/richiesta (max)."
    )


if __name__ == "__main__":
    benchmark()

# ---------------------------------------------------------------------------
# (c) Risposta alla domanda di riflessione
# ---------------------------------------------------------------------------
#
# Con soli 3 shard e poche richieste, thread pool e asyncio si comportano
# in modo quasi identico: entrambi ottengono concorrenza nel fan-out e la
# latenza per richiesta e' vicina al max delle latenze di shard.
#
# La differenza emerge quando il numero di shard (o di richieste
# concorrenti da servire) cresce molto:
#   - il thread pool paga un thread OS per chiamata in corso: con centinaia
#     di shard o migliaia di richieste concorrenti, il costo di
#     scheduling/context-switch e la memoria per thread diventano
#     significativi, e la tail latency (p99) peggiora perche' i thread
#     competono per la CPU e per il Global Interpreter Lock nei tratti di
#     codice Python puro fra una I/O e l'altra;
#   - asyncio multiplexa migliaia di richieste I/O-bound su un solo thread
#     e un solo event loop: nessun costo di context-switch tra thread,
#     footprint di memoria molto più basso, e la tail latency resta
#     dominata dalla latenza di rete/shard più lento, non dal
#     sovraccarico del client.
#
# In sintesi: per pochi shard la scelta e' quasi indifferente; quando il
# fan-out cresce (molti shard, o molte richieste client concorrenti), la
# versione asincrona scala meglio in throughput e mantiene una tail
# latency più bassa e più prevedibile.
