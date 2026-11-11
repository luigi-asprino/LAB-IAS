# =============================================================================
# client.py — Client di test e benchmark per lo Sharded Service Pattern
# =============================================================================
#
# Questo script serve a due scopi:
#
#   1. VERIFICA FUNZIONALE: invia una singola richiesta e mostra la risposta,
#      utile per verificare che la pipeline di shard funzioni correttamente.
#
#   2. BENCHMARK: misura latenza e throughput al variare del carico,
#      raccogliendo statistiche (media, mediana, p95, p99) su N richieste.
#
# Uso:
#   # Singola richiesta di test
#   python client.py --mode single
#
#   # Benchmark sequenziale (20 richieste una alla volta)
#   python client.py --mode bench --n 20
#
#   # Benchmark con concorrenza (20 richieste, 4 in parallelo)
#   python client.py --mode bench --n 20 --concurrency 4
#
#   # Confronto 1 vs 4 shard (richiede run_local.sh attivo su entrambe le porte)
#   python client.py --mode compare --url-1 http://localhost:8080 --url-2 http://localhost:9080
#
# Variabili d'ambiente:
#   ROUTER_URL  : URL del primo shard (default http://localhost:8080)
#   SEQ_LEN     : lunghezza delle sequenze sintetiche (default 16)
#   BATCH_SIZE  : campioni per richiesta (default 1)
# =============================================================================

import argparse
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# =============================================================================
# SEZIONE 1 — Configurazione
# =============================================================================

ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:8080")
SEQ_LEN    = int(os.environ.get("SEQ_LEN", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
VOCAB_SIZE = 30522  # deve corrispondere a quello del modello


# =============================================================================
# SEZIONE 2 — Generazione di input sintetici
# =============================================================================

def make_token_ids(batch: int = 1, seq_len: int = 16) -> list[list[int]]:
    """
    Genera token ID casuali nel range [101, VOCAB_SIZE-1].
    101 è il token [CLS] in BERT: lo mettiamo sempre in posizione 0
    per rispettare la convenzione (l'ultimo shard lo usa per la classificazione).

    In un sistema reale questi token_ids verrebbero prodotti da un tokenizer
    (es. BertTokenizer, tiktoken) a partire da testo in linguaggio naturale.
    """
    return [
        [101] + [random.randint(102, VOCAB_SIZE - 1) for _ in range(seq_len - 1)]
        for _ in range(batch)
    ]


# =============================================================================
# SEZIONE 3 — Funzioni di inferenza
# =============================================================================

def infer(
    token_ids: list[list[int]],
    url:       str = ROUTER_URL,
    timeout:   int = 30,
) -> tuple[dict, float]:
    """
    Invia una richiesta di inferenza al primo shard e restituisce
    (risposta, latenza_lato_client).

    La latenza lato client include:
      - Serializzazione JSON del payload
      - Trasmissione HTTP al primo shard
      - Elaborazione dell'intera pipeline di shard
      - Trasmissione HTTP della risposta
      - Deserializzazione JSON della risposta

    La latenza misurata dal server (response["latency_ms"]) esclude
    solo il trasporto tra client e primo shard.

    Parametri
    ----------
    token_ids : lista di liste di interi (batch × seq_len)
    url       : URL del primo shard
    timeout   : timeout in secondi per la richiesta HTTP

    Returns
    -------
    (risposta JSON come dict, latenza lato client in ms)
    """
    payload = {"token_ids": token_ids}

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/forward", json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Impossibile connettersi a {url}. "
            "Verifica che i shard siano in esecuzione con run_local.sh"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Timeout dopo {timeout}s in attesa di {url}")

    latency_ms = (time.perf_counter() - t0) * 1000
    return resp.json(), latency_ms


def check_health(url: str = ROUTER_URL) -> bool:
    """
    Verifica che il primo shard sia raggiungibile e risponda al health check.
    Utile per debugging prima di avviare il benchmark.
    """
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            print(f"[health] Shard {info['shard_id']} OK — {info['model_info']}")
            return True
    except requests.exceptions.RequestException:
        pass
    print(f"[health] ERRORE: {url} non raggiungibile")
    return False


# =============================================================================
# SEZIONE 4 — Modalità di esecuzione
# =============================================================================

def run_single(url: str = ROUTER_URL):
    """
    Invia una singola richiesta e mostra il risultato dettagliato.
    Utile per verificare il funzionamento della pipeline.
    """
    print(f"\n=== Test singola richiesta → {url} ===")
    print(f"Input: batch={BATCH_SIZE}, seq_len={SEQ_LEN}")

    ids = make_token_ids(BATCH_SIZE, SEQ_LEN)
    print(f"Token IDs (prima sequenza, primi 5 token): {ids[0][:5]}...")

    result, client_lat = infer(ids, url)

    if "error" in result:
        print(f"\n[ERRORE] {result['error']}")
        return

    print(f"\nOutput (logit): {result['output']}")
    print(f"\nLatenza lato client:  {client_lat:.1f} ms")
    print(f"Latenza pipeline:     {result.get('latency_ms', 'n/a')} ms")

    shard_times = result.get("shard_times_ms", [])
    if shard_times:
        print(f"Tempo per shard:      {shard_times} ms")
        print(f"  Shard più lento:    shard {shard_times.index(max(shard_times))} ({max(shard_times):.1f} ms)")


def run_benchmark(
    url:         str = ROUTER_URL,
    n:           int = 20,
    concurrency: int = 1,
):
    """
    Benchmark sequenziale o concorrente su N richieste.

    Con concurrency=1: le richieste vengono inviate una alla volta (sequenziale).
    Con concurrency>1: vengono inviate  C (concurrency) richieste in parallelo usando un
    ThreadPoolExecutor. Il throughput scala (idealmente) linearmente con C
    fino a saturare il collo di bottiglia (rete, compute, o GIL Python).

    Statistiche riportate:
      - Latenza: media, mediana, p95, p99, min, max
      - Throughput: richieste/secondo (wall-clock time totale / N)
    """
    print(f"\n=== Benchmark: {n} richieste, concorrenza={concurrency}, → {url} ===")

    latencies = []
    errors    = 0
    t_wall_start = time.perf_counter()

    if concurrency == 1:
        # Sequenziale: più semplice, misura la latenza pura senza effetti di coda
        for i in range(n):
            ids = make_token_ids(BATCH_SIZE, SEQ_LEN)
            try:
                _, lat = infer(ids, url)
                latencies.append(lat)
            except Exception as e:
                print(f"  [req {i}] Errore: {e}")
                errors += 1

            if (i + 1) % 5 == 0:
                print(f"  Completate {i+1}/{n} richieste...")

    else:
        # Concorrente: misura il throughput con più richieste in parallelo.
        # Nota: Python ha il GIL, ma le richieste HTTP sono I/O-bound
        # (bloccano in attesa della rete), quindi ThreadPoolExecutor è
        # efficiente senza bisogno di asyncio.
        def send_one(_):
            ids = make_token_ids(BATCH_SIZE, SEQ_LEN)
            return infer(ids, url)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_one, i) for i in range(n)]
            for i, future in enumerate(as_completed(futures)):
                try:
                    _, lat = future.result()
                    latencies.append(lat)
                except Exception as e:
                    print(f"  [req {i}] Errore: {e}")
                    errors += 1

                if (i + 1) % 5 == 0:
                    print(f"  Completate {i+1}/{n} richieste...")

    t_wall = time.perf_counter() - t_wall_start

    # ── Report statistiche ───────────────────────────────────────────────────
    if not latencies:
        print(f"\nNessuna richiesta riuscita ({errors} errori)")
        return

    latencies.sort()
    n_ok = len(latencies)
    p95  = latencies[int(n_ok * 0.95)]
    p99  = latencies[int(n_ok * 0.99)] if n_ok >= 100 else latencies[-1]

    print(f"\n{'─'*45}")
    print(f"  Richieste:   {n_ok}/{n} OK  ({errors} errori)")
    print(f"  Latenza media:   {statistics.mean(latencies):.1f} ms")
    print(f"  Latenza mediana: {statistics.median(latencies):.1f} ms")
    print(f"  p95:             {p95:.1f} ms")
    print(f"  p99:             {p99:.1f} ms")
    print(f"  Min / Max:       {min(latencies):.1f} / {max(latencies):.1f} ms")
    print(f"  Throughput:      {n_ok / t_wall:.2f} req/s  (wall-clock {t_wall:.1f}s)")
    print(f"{'─'*45}")

    return {
        "n_ok": n_ok, "mean": statistics.mean(latencies),
        "median": statistics.median(latencies), "p95": p95,
        "throughput": n_ok / t_wall,
    }


def run_compare(url1: str, url2: str, n: int = 20):
    """
    Confronta le performance di due configurazioni (es. 2 shard vs 4 shard).

    Utile per rispondere alla domanda: "aggiungere shard aumenta o riduce
    le performance?" (spoiler: riduce la latenza di compute per shard, ma
    aggiunge latenza di rete tra shard — il trade-off dipende dal modello).
    """
    print(f"\n=== Confronto performance ===")
    print(f"  A: {url1}")
    print(f"  B: {url2}")

    print(f"\nBenchmark A ({url1}):")
    stats_a = run_benchmark(url1, n=n, concurrency=1)

    print(f"\nBenchmark B ({url2}):")
    stats_b = run_benchmark(url2, n=n, concurrency=1)

    if stats_a and stats_b:
        ratio = stats_b["mean"] / stats_a["mean"]
        print(f"\n{'═'*45}")
        print(f"  B/A latenza media: {ratio:.2f}x")
        if ratio < 1:
            print(f"  → B è {1/ratio:.1f}x PIÙ VELOCE di A")
        else:
            print(f"  → B è {ratio:.1f}x PIÙ LENTO di A")
        print(f"{'═'*45}")


# =============================================================================
# Entry point — parsing degli argomenti
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client di test e benchmark per lo Sharded Service Pattern"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "bench", "compare", "health"],
        default="single",
        help="Modalità di esecuzione (default: single)",
    )
    parser.add_argument(
        "--url",
        default=ROUTER_URL,
        help=f"URL del primo shard (default: {ROUTER_URL})",
    )
    parser.add_argument(
        "--url-1",
        default="http://localhost:8080",
        help="URL della configurazione A (per --mode compare)",
    )
    parser.add_argument(
        "--url-2",
        default="http://localhost:9080",
        help="URL della configurazione B (per --mode compare)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Numero di richieste per il benchmark (default: 20)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Richieste parallele nel benchmark (default: 1 = sequenziale)",
    )

    args = parser.parse_args()

    if args.mode == "health":
        check_health(args.url)
    elif args.mode == "single":
        check_health(args.url)
        run_single(args.url)
    elif args.mode == "bench":
        check_health(args.url)
        run_benchmark(args.url, n=args.n, concurrency=args.concurrency)
    elif args.mode == "compare":
        run_compare(args.url_1, args.url_2, n=args.n)
