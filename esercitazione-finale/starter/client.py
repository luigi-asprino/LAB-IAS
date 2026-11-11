"""
Esercizio 2 — Client di orchestrazione per un servizio sharded (STARTER)
===========================================================================

Il modello e' partizionato in 3 shard, ciascuno esposto da shard_server.py
(gia' pronto, non modificarlo). Il compito e' scrivere il client che:

  (a) Sharded service — invia la stessa richiesta a tutti gli shard
      (fan-out) e ne combina le risposte in un unico risultato (fan-in).
      Lezione 9 e 10.
  (b) Implementare tre varianti della chiamata: sequenziale (naive),
      con thread pool, e asincrona con asyncio — e misurarne la latenza.
      Lezione 10.
  (c) Rispondere alla domanda di riflessione in fondo al file.
  (d) Bonus: timeout + fallback per shard lenti/non disponibili.

Prima di lanciare questo script, avviare i 3 shard (in terminali separati
o con run_shards.sh):
    python shard_server.py --port 8001 --shard-id 0 --latency 0.30
    python shard_server.py --port 8002 --shard-id 1 --latency 0.50
    python shard_server.py --port 8003 --shard-id 2 --latency 0.20

Poi lanciare:
    python client.py
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

TIMEOUT_S = 2.0  # timeout per singola chiamata a uno shard (punto d)


# ---------------------------------------------------------------------------
# (a) Fan-in: combinare i risultati parziali degli shard
# ---------------------------------------------------------------------------

def combine(partials):
    """TODO (a): aggregare le risposte parziali degli shard in un unico
    risultato. Gestire anche il caso in cui uno o piu' shard non abbiano
    risposto (partial=None), riportando quanti shard hanno risposto
    correttamente (punto d, fallback)."""
    raise NotImplementedError("TODO (a): implementare combine()")


# ---------------------------------------------------------------------------
# (b) Variante 1 — sincrona sequenziale (naive)
# ---------------------------------------------------------------------------

def call_shard_sync(url: str, payload: dict):
    """TODO (d): gestire timeout/errori restituendo None invece di
    propagare l'eccezione, per permettere il fallback in combine()."""
    r = requests.post(url, json=payload, timeout=TIMEOUT_S)
    return r.json()


def predict_sync_sequential(payload: dict):
    """TODO (b): fan-out chiamando gli shard UNO ALLA VOLTA, in sequenza."""
    raise NotImplementedError("TODO (b): implementare predict_sync_sequential")


# ---------------------------------------------------------------------------
# (b) Variante 2 — sincrona con thread pool
# ---------------------------------------------------------------------------

def predict_sync_threadpool(payload: dict):
    """TODO (b): fan-out lanciando le chiamate agli shard in parallelo
    usando un ThreadPoolExecutor (una richiesta bloccante per thread)."""
    raise NotImplementedError("TODO (b): implementare predict_sync_threadpool")


# ---------------------------------------------------------------------------
# (b) Variante 3 — asincrona con asyncio
# ---------------------------------------------------------------------------

async def call_shard_async(session: aiohttp.ClientSession, url: str, payload: dict):
    """TODO (d): gestire timeout/errori restituendo None."""
    async with session.post(url, json=payload, timeout=TIMEOUT_S) as resp:
        return await resp.json()


async def predict_async(payload: dict):
    """TODO (b): fan-out concorrente con asyncio.gather su tutti gli shard."""
    raise NotImplementedError("TODO (b): implementare predict_async")


# ---------------------------------------------------------------------------
# Benchmark — confronta le tre varianti
# ---------------------------------------------------------------------------

def benchmark(n_requests: int = 10):
    payload = {"x": [1, 2, 3, 4, 5]}

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


if __name__ == "__main__":
    benchmark()

# ---------------------------------------------------------------------------
# (c) Domanda di riflessione
# ---------------------------------------------------------------------------
# In quali condizioni (numero di shard, latenza per shard, variabilita' fra
# shard) la versione asincrona conviene rispetto a quella sincrona (sia
# sequenziale che a thread pool)? Collegare la risposta a tail latency e
# utilizzo delle risorse (thread OS vs event loop a singolo thread).
#
# Risposta:
# ...
