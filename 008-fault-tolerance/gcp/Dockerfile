# =============================================================================
# Dockerfile — Lezione 8: Elasticity e Fault-Tolerance con Spot VM
# =============================================================================
#
# Immagine base identica alla Lezione 7 per continuità.
#
# Nota sul SIGTERM su Vertex AI:
#   Vertex AI invia SIGTERM al container 30 secondi prima di terminare la VM
#   Spot. Il processo Python riceve il segnale e l'handler in train_spot.py
#   salva il checkpoint su GCS entro quel limite di tempo.
#
#   Con modelli molto grandi (>1 GB) il salvataggio potrebbe non completarsi
#   in 30 secondi. In quel caso è necessario impostare checkpoint_interval
#   più basso (ogni N step invece che ogni epoch) o usare checkpoint asincroni.
#
# Build e push:
#   PROJECT=test-project-493915
#   REGION=europe-west4
#   REPO=ias-repo
#
#   docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab08-spot:v1 .
#   docker push ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab08-spot:v1
#
# Test locale (senza GPU, senza GCS):
#   docker run --rm \
#     -e USE_SYNTHETIC=1 \
#     -e N_EPOCHS=10 \
#     ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab08-spot:v1
# =============================================================================

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# ── Dipendenze Python ─────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Codice applicativo ────────────────────────────────────────────────────────
COPY . .

# ── Variabili d'ambiente con valori di default ────────────────────────────────
ENV USE_SYNTHETIC="0"
ENV N_SAMPLES="4000"
ENV N_FEATURES="128"
ENV BATCH_SIZE="128"
ENV N_EPOCHS="20"
ENV LR="0.05"
ENV CHECKPOINT_INTERVAL="1"
ENV GCS_BUCKET="ias-luigi-asprino-bucket-eu"
ENV GCS_MODEL_PATH="models/lezione8/"

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "train_spot.py"]
