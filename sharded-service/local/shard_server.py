# =============================================================================
# shard_server.py — Server HTTP per un singolo shard del modello
# =============================================================================
#
# Ogni shard del modello gira come un processo Flask indipendente.
# Questo script avvia il server per lo shard identificato da SHARD_ID.
#
# Architettura della pipeline di serving:
#
#   Client
#     │  POST /forward  {"token_ids": [[101, 2054, ...]]}
#     ▼
#   Shard 0  (porta 8080)
#     │  forward locale → tensore (batch, seq, d_model)
#     │  POST /forward  {"tensor": [...], "dtype": "float32"}
#     ▼
#   Shard 1  (porta 8081)
#     │  forward locale → logit (batch, n_classes)
#     │  (nessun next shard → risposta al caller)
#     ▼
#   Shard 0  (riceve la risposta di Shard 1)
#     │
#     ▼
#   Client  {"output": [[0.12, -0.34]], "latency_ms": 45.2}
#
# La catena è configurata tramite variabili d'ambiente:
#   SHARD_ID      : indice di questo shard (0-indexed, default 0)
#   N_SHARDS      : numero totale di shard (default 2)
#   NEXT_SHARD_URL: URL base del prossimo shard (es. http://localhost:8081)
#                   Vuoto o assente → questo è l'ultimo shard
#   PORT          : porta su cui ascoltare (default 8080)
#
# Parametri del modello (devono essere uguali per tutti gli shard!):
#   TOTAL_LAYERS  : layer Transformer complessivi (default 8)
#   D_MODEL       : dimensione embedding (default 256)
#   NHEAD         : teste dell'attenzione (default 4)
#   DIM_FF        : feed-forward interno (default 512)
#   VOCAB_SIZE    : dimensione vocabolario (default 30522)
#   N_CLASSES     : classi di output (default 2)
#
# Avvio locale (due terminali separati):
#   SHARD_ID=1 N_SHARDS=2 PORT=8081 python shard_server.py   # shard 1 prima!
#   SHARD_ID=0 N_SHARDS=2 PORT=8080 NEXT_SHARD_URL=http://localhost:8081 python shard_server.py
# =============================================================================

import json
import os
import time

import requests
import torch
from flask import Flask, Response, jsonify, request

from model import ShardedTransformer

# =============================================================================
# SEZIONE 1 — Inizializzazione del modello (a livello di modulo)
# =============================================================================
#
# Il modello viene caricato UNA SOLA VOLTA all'avvio del processo, non ad
# ogni richiesta. Questo è fondamentale per la performance: caricare i pesi
# (~MB) per ogni richiesta sarebbe ordini di grandezza più lento.
#
# In un sistema di produzione, il modello verrebbe caricato da GCS o
# Vertex AI Model Registry. Qui usiamo pesi randomici (nessun training)
# perché l'obiettivo è osservare il pattern di routing, non l'accuracy.

SHARD_ID      = int(os.environ.get("SHARD_ID", "0"))
N_SHARDS      = int(os.environ.get("N_SHARDS", "2"))
NEXT_SHARD    = os.environ.get("NEXT_SHARD_URL", "").strip()  # "" → ultimo shard
PORT          = int(os.environ.get("PORT", "8080"))
TOTAL_LAYERS  = int(os.environ.get("TOTAL_LAYERS", "8"))
D_MODEL       = int(os.environ.get("D_MODEL", "256"))
NHEAD         = int(os.environ.get("NHEAD", "4"))
DIM_FF        = int(os.environ.get("DIM_FF", "512"))
VOCAB_SIZE    = int(os.environ.get("VOCAB_SIZE", "30522"))
N_CLASSES     = int(os.environ.get("N_CLASSES", "2"))

print(f"[init] Caricamento shard {SHARD_ID}/{N_SHARDS}...")

model = ShardedTransformer(
    total_layers = TOTAL_LAYERS,
    shard_id     = SHARD_ID,
    n_shards     = N_SHARDS,
    d_model      = D_MODEL,
    nhead        = NHEAD,
    dim_ff       = DIM_FF,
    vocab_size   = VOCAB_SIZE,
    n_classes    = N_CLASSES,
)
model.eval()  # disabilita dropout: siamo in inferenza, non in training

print(f"[init] {model.info()}")
print(f"[init] Next shard: {NEXT_SHARD if NEXT_SHARD else '(nessuno, sono l ultimo)'}")
print(f"[init] Pronto su porta {PORT}")


# =============================================================================
# SEZIONE 2 — Serializzazione del tensore
# =============================================================================
#
# Il tensore di attivazione deve viaggiare in rete tra shard. Scegliamo JSON
# per semplicità e debuggabilità: è leggibile, non richiede librerie extra,
# e va bene per le dimensioni di questa lezione (pochi KB per richiesta).
#
# In produzione si usano formati binari più efficienti:
#   - Protocol Buffers (gRPC) → ~3x più compatto, usato in TensorFlow Serving
#   - Apache Arrow (flight protocol) → zero-copy, usato in Triton Inference Server
#   - Pickled tensors → semplice ma non cross-language
#
# Per un tensore (1, 16, 256) in float32:
#   JSON:   ~200 KB (lista anniddata di float, molto overhead)
#   Binary: ~16 KB  (1 × 16 × 256 × 4 byte)
# Il trade-off JSON vs binary è esattamente uno degli esperimenti della lezione.

def tensor_to_json(t: torch.Tensor) -> dict:
    """
    Converte un tensore PyTorch in un dizionario serializzabile JSON.

    Preserva shape e dtype per consentire la ricostruzione fedele.
    I valori float vengono arrotondati a 6 cifre significative per ridurre
    la dimensione del payload senza perdita significativa di precisione.
    """
    return {
        "data":  t.tolist(),   # lista Python annidata (può essere profonda)
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),  # es. "float32", "int64"
    }


def json_to_tensor(payload: dict) -> torch.Tensor:
    """
    Ricostruisce un tensore PyTorch dal dizionario JSON prodotto da tensor_to_json.

    Gestisce i dtype più comuni: float32 (attivazioni), int64 (token ID).
    """
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int64":   torch.int64,
        "long":    torch.int64,
        "int32":   torch.int32,
    }
    dtype = dtype_map.get(payload.get("dtype", "float32"), torch.float32)
    return torch.tensor(payload["data"], dtype=dtype)


# =============================================================================
# SEZIONE 3 — Endpoint Flask
# =============================================================================

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint — usato da Cloud Run e dai load balancer per
    verificare che il container sia pronto a ricevere richieste.

    Restituisce 200 OK con informazioni sul shard. Durante la fase di
    warm-up (avvio del container) questo endpoint non è ancora raggiungibile,
    il che segnala al sistema di non inviare traffico.
    """
    return jsonify({
        "status":      "ok",
        "shard_id":    SHARD_ID,
        "n_shards":    N_SHARDS,
        "next_shard":  NEXT_SHARD or None,
        "model_info":  model.info(),
    })


@app.route("/forward", methods=["POST"])
def forward():
    """
    Endpoint principale: esegue il forward pass del proprio shard e
    propaga il risultato al shard successivo (se esiste).

    Protocollo per il PRIMO shard (shard_id == 0):
      Input:  {"token_ids": [[101, 2023, 2003, ...]]}
              token_ids: lista di liste di interi (batch × seq_len)

    Protocollo per gli shard intermedi e l'ULTIMO:
      Input:  {"tensor": {"data": [...], "shape": [...], "dtype": "float32"}}
              tensor: dizionario prodotto da tensor_to_json

    Risposta (solo dall'ultimo shard, risale la catena):
      {"output": [[0.12, -0.34]], "latency_ms": 45.2, "shard_times_ms": [...]}

    La latenza totale è la somma delle latenze di tutti gli shard più
    il tempo di serializzazione/deserializzazione JSON e trasmissione HTTP.
    """
    t_start = time.perf_counter()

    # ── Deserializza l'input ─────────────────────────────────────────────────
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Body JSON mancante o non valido"}), 400

    if SHARD_ID == 0:
        # Lo shard 0 riceve token ID dal client esterno (es. testo tokenizzato).
        # token_ids è una lista di liste: [[token1, token2, ...], [...], ...]
        if "token_ids" not in data:
            return jsonify({"error": "Campo 'token_ids' mancante"}), 400
        x = torch.tensor(data["token_ids"], dtype=torch.long)
    else:
        # Gli shard intermedi e l'ultimo ricevono il tensore di attivazione
        # prodotto dallo shard precedente.
        if "tensor" not in data:
            return jsonify({"error": "Campo 'tensor' mancante"}), 400
        x = json_to_tensor(data["tensor"])

    # ── Forward pass locale ──────────────────────────────────────────────────
    t_compute_start = time.perf_counter()
    with torch.no_grad():
        # torch.no_grad() disabilita il calcolo del grafo computazionale
        # (necessario per la backpropagation): in inferenza non serve e
        # risparmia memoria e tempo.
        y = model(x)
    t_compute_ms = (time.perf_counter() - t_compute_start) * 1000

    # ── Propaga al shard successivo o restituisce il risultato ──────────────
    if NEXT_SHARD:
        # Serializza il tensore di output e lo invia al prossimo shard.
        payload = {"tensor": tensor_to_json(y)}

        try:
            resp = requests.post(
                f"{NEXT_SHARD}/forward",
                json    = payload,
                timeout = 30,  # timeout per evitare blocchi indefiniti
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Se il prossimo shard non è raggiungibile, restituiamo un errore
            # esplicito invece di lasciare la richiesta in timeout.
            return jsonify({
                "error":       f"Shard {SHARD_ID} non riesce a contattare {NEXT_SHARD}: {e}",
                "shard_id":    SHARD_ID,
            }), 503

        # La risposta del prossimo shard contiene già il risultato finale.
        # Aggiungiamo il tempo di questo shard alla lista shard_times_ms.
        t_total_ms   = (time.perf_counter() - t_start) * 1000
        result       = resp.json()
        shard_times  = result.get("shard_times_ms", [])
        shard_times.insert(0, round(t_compute_ms, 2))  # prependi il nostro tempo
        result["shard_times_ms"] = shard_times
        return jsonify(result)

    else:
        # Siamo sull'ultimo shard: y contiene i logit finali.
        t_total_ms = (time.perf_counter() - t_start) * 1000
        return jsonify({
            "output":         y.tolist(),
            "latency_ms":     round(t_total_ms, 2),
            "shard_times_ms": [round(t_compute_ms, 2)],
        })


@app.route("/info", methods=["GET"])
def info():
    """
    Restituisce informazioni dettagliate sul modello di questo shard.
    Utile per debug e per verificare che tutti gli shard abbiano
    la stessa configurazione del modello.
    """
    return jsonify({
        "shard_id":       SHARD_ID,
        "n_shards":       N_SHARDS,
        "total_layers":   TOTAL_LAYERS,
        "d_model":        D_MODEL,
        "param_count":    model.param_count(),
        "param_mb":       round(model.param_bytes() / 1024 / 1024, 2),
        "activation_kb":  round(model.activation_bytes() / 1024, 2),
        "next_shard":     NEXT_SHARD or None,
    })


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # debug=False in produzione: il server di sviluppo Flask non è adatto
    # a carichi elevati (single-threaded, nessun worker pool).
    # In GCP usiamo gunicorn (vedi Dockerfile).
    app.run(host="0.0.0.0", port=PORT, debug=False)
