"""
LAB-IAS - Lezione 11: Step Memoization
========================================

Modulo core: decoratore @memoize_step per pipeline di Machine Learning in PyTorch.

Idea di fondo
-------------
Una pipeline ML e' un DAG di step (preprocessing, feature extraction, training, ...).
Molti step sono COSTOSI (GPU, I/O, tempo) ma DETERMINISTICI rispetto al loro input:
se input, codice e configurazione non cambiano, il risultato non cambia.

La "step memoization" applica il concetto classico di memoization (cache su
argomenti di una funzione pura) a livello di STEP di pipeline, calcolando una
cache key (fingerprint) che combina:

    1. il codice sorgente dello step (hash della funzione)
    2. gli argomenti di input (inclusi tensori PyTorch, array, file, config)
    3. eventuali metadati di ambiente rilevanti (es. versione del modello base)

Se la fingerprint e' gia' presente in cache, lo step non viene rieseguito:
il risultato viene deserializzato e restituito immediatamente (cache hit).
Altrimenti lo step viene eseguito e il risultato salvato (cache miss).

Backend di cache supportati in questo modulo:
    - diskcache (locale, filesystem) -> vedi memoize_step_diskcache
    - Redis (locale o remoto, riusa lo stack Celery+Redis della Lezione 10)
      -> vedi memoize_step_redis.py

Dipendenze:
    pip install diskcache torch
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import io
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch


# ---------------------------------------------------------------------------
# 1. Costruzione della cache key (fingerprint)
# ---------------------------------------------------------------------------

def _hash_tensor(t: torch.Tensor) -> bytes:
    """Serializza un tensore PyTorch in modo deterministico per l'hashing.

    Nota: usiamo t.cpu().contiguous() per garantire che due tensori con lo
    stesso contenuto ma layout di memoria diverso producano lo stesso hash.
    """
    buf = io.BytesIO()
    torch.save(t.detach().cpu().contiguous(), buf)
    return buf.getvalue()


def _hash_value(value: Any) -> bytes:
    """Serializza un valore qualunque (tensore, dict, list, scalare, ...) in
    bytes deterministici da passare all'hash SHA-256.
    """
    if isinstance(value, torch.Tensor):
        return _hash_tensor(value)
    if isinstance(value, (list, tuple)):
        return b"|".join(_hash_value(v) for v in value)
    if isinstance(value, dict):
        # Ordiniamo le chiavi per garantire determinismo indipendentemente
        # dall'ordine di inserimento nel dict.
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return b"|".join(_hash_value(k).__add__(b"=").__add__(_hash_value(v)) for k, v in items)
    # Fallback: pickle per tipi generici (str, int, float, oggetti custom, ...)
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _hash_source(func: Callable) -> str:
    """Hash del codice sorgente dello step.

    Questo e' cio' che rende la cache "code-aware": se modifichiamo la logica
    dello step (anche solo una riga), la fingerprint cambia e la cache
    precedente viene automaticamente invalidata, anche se gli input restano
    identici.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        # Non sempre possibile ottenere il sorgente (es. funzioni definite
        # dinamicamente); fallback sul nome qualificato.
        source = func.__qualname__
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def compute_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    extra_context: Optional[dict] = None,
) -> str:
    """Calcola la fingerprint (cache key) per una singola invocazione di step.

    La key combina:
      - hash del codice sorgente della funzione
      - hash di tutti gli argomenti posizionali e nominali
      - hash di un contesto extra opzionale (es. versione dataset, versione
        modello base, seed) fornito esplicitamente dal chiamante

    Returns
    -------
    str
        Digest esadecimale SHA-256, usato come chiave di cache.
    """
    hasher = hashlib.sha256()
    hasher.update(_hash_source(func).encode("utf-8"))
    for a in args:
        hasher.update(_hash_value(a))
    for k in sorted(kwargs.keys()):
        hasher.update(k.encode("utf-8"))
        hasher.update(_hash_value(kwargs[k]))
    if extra_context:
        hasher.update(_hash_value(extra_context))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# 2. Decoratore basato su diskcache (backend locale, filesystem)
# ---------------------------------------------------------------------------

def memoize_step_diskcache(
    cache_dir: str = ".step_cache",
    extra_context: Optional[dict] = None,
    verbose: bool = True,
):
    """Decoratore per memoization di step su filesystem locale (diskcache).

    Adatto a: esecuzione locale, singola macchina, debugging/iterazione rapida.

    Parameters
    ----------
    cache_dir : str
        Directory dove diskcache persiste i risultati.
    extra_context : dict, opzionale
        Metadati aggiuntivi da includere nella fingerprint (es. versione
        dataset). Utile per invalidare manualmente la cache incrementando
        un valore di versione.
    verbose : bool
        Se True, stampa a schermo hit/miss e tempo risparmiato/impiegato.

    Example
    -------
    >>> @memoize_step_diskcache(cache_dir=".cache/preprocess")
    ... def preprocess(dataset_path: str, tokenizer_name: str) -> torch.Tensor:
    ...     ...  # operazione costosa
    """
    import diskcache  # import locale: dipendenza opzionale del modulo

    cache = diskcache.Cache(cache_dir)

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = compute_cache_key(func, args, kwargs, extra_context)
            if key in cache:
                if verbose:
                    print(f"[memoize_step] CACHE HIT  step={func.__name__} key={key[:10]}...")
                return cache[key]

            if verbose:
                print(f"[memoize_step] CACHE MISS step={func.__name__} key={key[:10]}... eseguo lo step")
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            cache[key] = result
            if verbose:
                print(f"[memoize_step] step={func.__name__} eseguito in {elapsed:.2f}s e salvato in cache")
            return result

        wrapper.cache = cache  # esposto per ispezione/clear manuale nei test
        wrapper.cache_key_for = lambda *a, **kw: compute_cache_key(func, a, kw, extra_context)
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 3. Esempio di step tipici di una pipeline ML (per la demo in laboratorio)
# ---------------------------------------------------------------------------

@memoize_step_diskcache(cache_dir=".step_cache/preprocess")
def preprocess_dataset(raw_texts: list[str], max_len: int = 64) -> torch.Tensor:
    """Step di preprocessing: simula una tokenizzazione costosa.

    In una pipeline reale qui ci sarebbe una chiamata a un tokenizer
    HuggingFace o a una pipeline di normalizzazione testo su larga scala.
    """
    time.sleep(2)  # simuliamo un'operazione costosa (I/O + CPU bound)
    # "Tokenizzazione" fittizia: lunghezza della stringa troncata a max_len
    encoded = [
        [min(ord(c), 255) for c in text[:max_len]] + [0] * max(0, max_len - len(text))
        for text in raw_texts
    ]
    return torch.tensor(encoded, dtype=torch.long)


@memoize_step_diskcache(cache_dir=".step_cache/features")
def extract_features(tokens: torch.Tensor, embedding_dim: int = 32) -> torch.Tensor:
    """Step di feature extraction con un modello congelato.

    In produzione questo sarebbe un forward pass attraverso un encoder
    pre-addestrato (es. BERT congelato); qui usiamo un embedding layer
    inizializzato con seed fisso per rendere la demo deterministica e
    riproducibile in laboratorio, senza dipendere da pesi scaricati da rete.
    """
    time.sleep(3)  # simuliamo un forward pass costoso su GPU
    torch.manual_seed(42)
    vocab_size = 256
    embedding = torch.nn.Embedding(vocab_size, embedding_dim)
    with torch.no_grad():
        features = embedding(tokens).mean(dim=1)  # mean pooling
    return features


if __name__ == "__main__":
    # Demo eseguibile: due run consecutivi mostrano cache miss -> cache hit
    texts = ["distributed training", "gpu memory management", "checkpointing"]

    print("\n--- Prima esecuzione (ci si aspetta MISS) ---")
    tokens = preprocess_dataset(texts, max_len=32)
    features = extract_features(tokens, embedding_dim=16)
    print("Shape feature:", features.shape)

    print("\n--- Seconda esecuzione, stessi input (ci si aspetta HIT) ---")
    tokens2 = preprocess_dataset(texts, max_len=32)
    features2 = extract_features(tokens2, embedding_dim=16)
    print("Shape feature:", features2.shape)
    assert torch.equal(features, features2)

    print("\n--- Terza esecuzione, parametro diverso (ci si aspetta MISS) ---")
    tokens3 = preprocess_dataset(texts, max_len=64)  # max_len cambiato!
    print("Nuova fingerprint calcolata perche' l'input di configurazione e' cambiato.")
