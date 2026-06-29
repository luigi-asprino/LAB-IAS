# =============================================================================
# Dockerfile — Lezione 9: Sharded Service Pattern su Cloud Run
# =============================================================================
#
# Questo Dockerfile produce UN'UNICA immagine parametrizzabile via variabili
# d'ambiente: la stessa immagine viene usata per tutti gli shard.
# Lo shard specifico viene configurato al momento del deploy (vedi deploy.sh).
#
# Scelta di Cloud Run:
#   - Nessuna gestione di VM: il container è la sola unità di deployment
#   - Scaling automatico: Cloud Run scala ogni shard indipendentemente
#   - Pay-per-request: ideale per un sistema di serving che ha traffico variabile
#   - HTTPS out of the box: non serve configurare certificati
#
# Differenza rispetto alle lezioni precedenti (Custom Training Job):
#   - Le lezioni 5–8 usavano Vertex AI Custom Jobs (batch, terminano quando finiscono)
#   - Questa lezione usa Cloud Run (serving, rimane attivo finché non viene eliminato)
#   - Il modello di pricing è diverso: Custom Jobs pagano per il tempo di VM,
#     Cloud Run paga per il numero di richieste e il tempo CPU.
#
# Gunicorn come server WSGI:
#   Flask da solo (app.run()) è single-threaded e non adatto alla produzione.
#   Gunicorn crea un pool di worker che gestiscono richieste concorrenti.
#   Con --workers=1 --threads=8, un singolo processo gestisce 8 thread I/O-bound.
#   Per richieste che fanno forward HTTP (I/O-bound), i thread sono efficienti.
# =============================================================================

FROM python:3.11-slim

# ── Dipendenze di sistema ────────────────────────────────────────────────────
# curl: utile per debug dall'interno del container (health check manuale)
# Nessun driver GPU necessario: Cloud Run usa CPU per questa lezione
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dipendenze Python ────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .

# torch CPU-only: ~200 MB invece di ~2 GB per la versione CUDA.
# In questa lezione l'inferenza è sul modello locale (leggero), non su GPU.
# L'indice PyPI speciale serve per installare torch senza CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir flask requests gunicorn

# ── Codice applicativo ───────────────────────────────────────────────────────
COPY model.py shard_server.py ./

# ── Variabili d'ambiente default ────────────────────────────────────────────
# Queste vengono sovrascritte da --set-env-vars nel comando di deploy Cloud Run
ENV SHARD_ID=0 \
    N_SHARDS=2 \
    TOTAL_LAYERS=8 \
    D_MODEL=256 \
    NHEAD=4 \
    DIM_FF=512 \
    VOCAB_SIZE=30522 \
    N_CLASSES=2 \
    PORT=8080

# ── Expose della porta ───────────────────────────────────────────────────────
# Cloud Run usa la porta specificata da $PORT (default 8080).
# Il EXPOSE è documentativo: Cloud Run mappa automaticamente la porta.
EXPOSE 8080

# ── Entrypoint ───────────────────────────────────────────────────────────────
# Gunicorn avvia shard_server:app con 1 worker e 8 thread.
# --timeout 60: tempo massimo di attesa per una richiesta (la pipeline
#               di shard può impiegare qualche secondo end-to-end).
# --bind 0.0.0.0:$PORT: ascolta su tutte le interfacce alla porta configurata.
CMD gunicorn \
    --workers=1 \
    --threads=8 \
    --timeout=60 \
    --bind="0.0.0.0:${PORT}" \
    shard_server:app
