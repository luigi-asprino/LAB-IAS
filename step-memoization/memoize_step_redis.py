"""
LAB-IAS - Lezione 11: Step Memoization
========================================

Variante con backend Redis, pensata per riutilizzare lo stack Celery+Redis
gia' introdotto in Lezione 10 (Fan-in/Fan-out e pattern sincroni/asincroni).

Perche' Redis invece di diskcache?
----------------------------------
diskcache (vedi memoize_step.py) e' perfetto in locale, su singolo processo.
Ma appena la pipeline gira su piu' worker Celery (come nella Lezione 10),
serve una cache CONDIVISA e accessibile da tutti i processi: Redis, gia'
presente nello stack come message broker, e' la scelta naturale.

Questo mostra concretamente il salto concettuale dal pattern "cache locale
per singolo processo" al pattern "cache condivisa per sistema distribuito",
lo stesso salto che poi ritroviamo nel caching nativo di Vertex AI Pipelines
(vedi kfp_caching_demo.py), dove la cache e' condivisa a livello di progetto
GCP tramite Vertex ML Metadata.

Dipendenze:
    pip install redis torch
    Un'istanza Redis raggiungibile (locale: `docker run -p 6379:6379 redis`)
"""

from __future__ import annotations

import functools
import pickle
import time
from typing import Any, Callable, Optional

import redis

from memoize_step import compute_cache_key


def memoize_step_redis(
    redis_url: str = "redis://localhost:6379/0",
    namespace: str = "stepcache",
    ttl_seconds: Optional[int] = None,
    extra_context: Optional[dict] = None,
    verbose: bool = True,
):
    """Decoratore per memoization di step su cache Redis condivisa.

    Adatto a: pipeline distribuite su piu' worker Celery (Lezione 10),
    dove piu' processi devono condividere la stessa cache di risultati.

    Parameters
    ----------
    redis_url : str
        URL di connessione a Redis (stesso broker usato da Celery).
    namespace : str
        Prefisso delle chiavi in Redis, per evitare collisioni con altre
        strutture dati (es. le code dei task Celery) sulla stessa istanza.
    ttl_seconds : int, opzionale
        Tempo di vita della entry in cache. None = nessuna scadenza
        (invalidazione affidata solo al cambio di fingerprint).
    extra_context : dict, opzionale
        Metadati extra da includere nella fingerprint (vedi memoize_step.py).
    """
    client = redis.Redis.from_url(redis_url)

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = compute_cache_key(func, args, kwargs, extra_context)
            redis_key = f"{namespace}:{func.__name__}:{key}"

            cached = client.get(redis_key)
            if cached is not None:
                if verbose:
                    print(f"[memoize_step_redis] CACHE HIT  step={func.__name__} key={key[:10]}...")
                return pickle.loads(cached)

            if verbose:
                print(f"[memoize_step_redis] CACHE MISS step={func.__name__} key={key[:10]}... eseguo lo step")
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0

            payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            if ttl_seconds is not None:
                client.setex(redis_key, ttl_seconds, payload)
            else:
                client.set(redis_key, payload)

            if verbose:
                print(f"[memoize_step_redis] step={func.__name__} eseguito in {elapsed:.2f}s, salvato in Redis (ttl={ttl_seconds})")
            return result

        wrapper.invalidate = lambda *a, **kw: client.delete(
            f"{namespace}:{func.__name__}:{compute_cache_key(func, a, kw, extra_context)}"
        )
        return wrapper

    return decorator


if __name__ == "__main__":
    import torch

    @memoize_step_redis(namespace="lab11_demo", ttl_seconds=3600)
    def slow_normalize(x: torch.Tensor) -> torch.Tensor:
        time.sleep(2)
        return (x - x.mean()) / x.std()

    data = torch.randn(1000)

    print("--- Worker A (o run 1): esegue lo step ---")
    out1 = slow_normalize(data)

    print("--- Worker B (o run 2), stesso input: legge da Redis ---")
    out2 = slow_normalize(data)

    assert torch.allclose(out1, out2)
    print("OK: risultati identici, secondo worker non ha ricalcolato nulla.")
