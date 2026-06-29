# Lezione 9 — Sharded Service Pattern

## Panoramica

Questa lezione affronta il problema del **serving di modelli di grandi dimensioni**:
come eseguire l'inferenza quando i pesi di un modello non entrano in una singola GPU
(o in una singola macchina).

La soluzione è il **layer sharding**: il modello viene suddiviso in partizioni
(*shard*) di layer contigui, ognuna ospitata da un processo indipendente.
Le richieste attraversano gli shard in sequenza, come in una pipeline.

**Collegamento con le lezioni precedenti**

| Lezione | Pattern | Riuso in L9 |
|---------|---------|-------------|
| L5 | Containerizzazione + Custom Job | Dockerfile e Artifact Registry |
| L6 | Parameter Server | Confronto architetturale (star vs catena) |
| L7 | Collective Communication | Confronto (all-reduce vs pipeline) |
| L8 | Fault Tolerance + Checkpoint | Domanda di approfondimento sulla resilienza |

---

## Struttura del repository

```
009-sharded-serving/
├── local/
│   ├── model.py              # ShardedTransformer: Transformer partizionabile
│   ├── shard_server.py       # Server Flask per ogni shard
│   ├── client.py             # Client di test e benchmark
│   ├── run_local.sh          # Avvia N shard localmente sulle porte 8080+
│   └── requirements.txt
└── gcp/
    ├── Dockerfile            # Immagine unica, parametrizzata via env
    └── deploy.sh             # Build, push e deploy su Cloud Run
```

---

## In Locale

### Prerequisiti

```bash
cd local/
pip install -r requirements.txt
```

Verifica che il modello funzioni correttamente:

```bash
python model.py
# Atteso:
# Shard 0: ShardedTransformer(shard=0/2, layer=0-3, embedding=sì, head=no, ...)
# Shard 1: ShardedTransformer(shard=1/2, layer=4-7, embedding=no, head=sì, ...)
# Smoke test: OK
```

---

### Avvio della pipeline con 2 shard

Avvia entrambi gli shard con lo script automatico:

```bash
./run_local.sh 2
```

Lo script avvia i processi in ordine inverso (prima shard 1, poi shard 0) e
configura automaticamente `NEXT_SHARD_URL`.

Log in tempo reale in un secondo terminale:

```bash
tail -f /tmp/shard_0.log /tmp/shard_1.log
```

Verifica che entrambi gli shard siano pronti:

```bash
curl http://localhost:8080/health | python3 -m json.tool
curl http://localhost:8081/health | python3 -m json.tool
```

---

### Prima richiesta di inferenza

In un altro terminale (mentre `run_local.sh` è ancora in esecuzione):

```bash
python client.py --mode single
```

Output atteso:

```
[health] Shard 0 OK — ShardedTransformer(shard=0/2, layer=0-3, ...)

=== Test singola richiesta → http://localhost:8080 ===
Input: batch=1, seq_len=16
Token IDs (prima sequenza, primi 5 token): [101, 15234, 8876, ...]

Output (logit): [[0.123, -0.456]]

Latenza lato client:  38.4 ms
Latenza pipeline:     35.1 ms
Tempo per shard:      [12.3, 22.8] ms
  Shard più lento:    shard 1 (22.8 ms)
```

Osservazioni da discutere:
- Perché shard 1 è più lento di shard 0?  
  *(Shard 1 non ha NEXT_SHARD da chiamare, ma ha la head di classificazione.
  La differenza principale è la latenza di serializzazione JSON del tensore
  di attivazione, che paga solo shard 0.)*
- Quanta è la differenza tra latenza lato client e latenza pipeline?  
  *(La differenza è il round-trip HTTP client→shard 0)*

---

### Benchmark sequenziale

```bash
python client.py --mode bench --n 30
```

Output atteso (valori indicativi su CPU):

```
=== Benchmark: 30 richieste, concorrenza=1, → http://localhost:8080 ===
  Completate 5/30 richieste...
  ...
─────────────────────────────────────────────
  Richieste:   30/30 OK  (0 errori)
  Latenza media:   35.2 ms
  Latenza mediana: 34.8 ms
  p95:             42.1 ms
  p99:             45.3 ms
  Min / Max:       31.2 / 48.7 ms
  Throughput:      27.4 req/s  (wall-clock 1.09s)
─────────────────────────────────────────────
```

---

### Benchmark concorrente

Senza chiudere la pipeline, misura l'effetto della concorrenza:

```bash
# Sequenziale (baseline)
python client.py --mode bench --n 40 --concurrency 1

# 2 richieste in parallelo
python client.py --mode bench --n 40 --concurrency 2

# 4 richieste in parallelo
python client.py --mode bench --n 40 --concurrency 4
```

Tabella di risultati attesa (i valori dipendono dalla macchina):

| Concorrenza | Latenza media | Throughput |
|-------------|---------------|------------|
| 1           | ~35 ms        | ~27 req/s  |
| 2           | ~38 ms        | ~50 req/s  |
| 4           | ~45 ms        | ~80 req/s  |

Domanda: il throughput scala linearmente? Se no, qual è il collo di bottiglia?

---

### Confronto 2 shard vs 4 shard

Avvia una seconda pipeline con 4 shard su porte diverse:

```bash
# In un nuovo terminale
BASE_PORT=9080 ./run_local.sh 4
```

Poi confronta:

```bash
python client.py --mode compare \
    --url-1 http://localhost:8080 \
    --url-2 http://localhost:9080 \
    --n 30
```

Domanda: la latenza aumenta o diminuisce passando da 2 a 4 shard?
Scomponi la latenza in:
- **compute**: meno layer per shard → riduce
- **rete**: più hop HTTP → aumenta

---

### Ispezione del tensore di attivazione

Capire quanto pesa il tensore che viaggia in rete tra shard:

```python
# Da Python interattivo
import requests
resp = requests.get("http://localhost:8080/info").json()
print(f"Attivazione per richiesta: {resp['activation_kb']} KB")
print(f"Parametri totali shard 0: {resp['param_count']:,}")
print(f"Dimensione modello shard 0: {resp['param_mb']} MB")
```

Poi modifica `SEQ_LEN` e `BATCH_SIZE` e osserva come cambia il costo:

```bash
SEQ_LEN=64 BATCH_SIZE=8 python client.py --mode bench --n 10
```

---

## Architettura della pipeline

```
Client HTTP
    │
    │  POST /forward {"token_ids": [[101, 2054, ...]]}
    ▼
┌──────────────────────────────────────┐
│  Shard 0                             │
│  - Embedding: token → vettore 256D   │
│  - TransformerBlock ×4               │
│  Porta 8080 / Cloud Run service      │
└──────────────────────┬───────────────┘
                       │
                       │  POST /forward {"tensor": {"data": [...], "dtype": "float32"}}
                       │  (~16 KB per richiesta con batch=1, seq=16, d_model=256)
                       ▼
┌──────────────────────────────────────┐
│  Shard 1                             │
│  - TransformerBlock ×4               │
│  - Head: vettore CLS → logit         │
│  Porta 8081 / Cloud Run service      │
└──────────────────────┬───────────────┘
                       │
                       │  {"output": [[0.12, -0.34]], "latency_ms": 35.1}
                       ▼
                    Client
```

---

## Domande di discussione

- **Pipeline bubble**: in questa implementazione ogni richiesta attraversa gli shard
   in sequenza. Come si potrebbe sfruttare la pipeline bubble sovrapponendo
   l'elaborazione di più richieste? (Suggerimento: micro-batching.)

- **Serializzazione**: usiamo JSON per trasportare i tensori tra shard.
   Con batch=1, seq=16, d\_model=256, il tensore è ~16 KB come dati binari
   ma ~200 KB come JSON. Quale formato binario useresti in produzione?
   (Protocol Buffers, Apache Arrow, pickled tensors — pro e contro di ciascuno.)
