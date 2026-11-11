# =============================================================================
# Dockerfile — Lezione 7: Collective Communication Pattern
# =============================================================================
#
# Immagine base: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
#   - PyTorch 2.0.1 con supporto CUDA 11.7 e cuDNN 8
#   - Include NCCL precompilato per comunicazione collettiva su GPU
#   - Compatibile con GPU T4 (Turing, Compute Capability 7.5) di Vertex AI
#
# Su CPU locale (esercizi): il container funziona identicamente
# con backend "gloo" invece di "nccl" — nessuna modifica al codice.
#
# Build e push su Artifact Registry:
#   PROJECT=test-project-493915
#   REGION=europe-west4
#   REPO=lab-repo
#
#   docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab07-ddp:latest .
#   docker push ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab07-ddp:latest
#
# Test locale (senza GPU):
#   docker run --rm \
#     -e USE_SYNTHETIC=1 \
#     -e RANK=0 -e WORLD_SIZE=1 -e MASTER_ADDR=localhost -e MASTER_PORT=29500 \
#     ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab07-ddp:latest
# =============================================================================

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# ── Dipendenze Python ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Codice applicativo ────────────────────────────────────────────────────────
# Copiare tutti i file del lab (dataset.py, model.py, train_ddp.py, ...)
COPY . .

# ── Variabili d'ambiente con valori di default ────────────────────────────────
# Tutte sovrascrivibili al lancio del job via job_config.yaml (env section)
# o via docker run -e KEY=VALUE.
ENV USE_SYNTHETIC="1"
ENV N_SAMPLES="4000"
ENV N_FEATURES="128"
ENV BATCH_SIZE="128"
ENV N_EPOCHS="5"
ENV LR="0.05"
ENV GCS_BUCKET="ias-luigi-asprino-bucket-eu"
ENV GCS_MODEL_PATH="models/lezione7/"

# ── Entry point ───────────────────────────────────────────────────────────────
# Vertex AI inietta RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT in ogni replica.
# Il container li legge tramite dist.init_process_group(init_method="env://").
CMD ["python", "train_ddp.py"]
