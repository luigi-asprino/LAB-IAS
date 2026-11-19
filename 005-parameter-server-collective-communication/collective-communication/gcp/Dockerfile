# =============================================================================
# Dockerfile — Lezione 5 (parte 2): Collective Communication Pattern
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
# Build e push su Artifact Registry (variabili del passo 0 del README):
#   docker build --platform linux/amd64 -t $IMAGE_DDP .
#   docker push $IMAGE_DDP
#
# Test locale (senza GPU):
#   docker run --rm \
#     -e USE_SYNTHETIC=1 \
#     -e RANK=0 -e WORLD_SIZE=1 -e MASTER_ADDR=localhost -e MASTER_PORT=29500 \
#     $IMAGE_DDP
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
# Tutte sovrascrivibili al lancio del job (sezione env di job.yaml)
# o via docker run -e KEY=VALUE. Nessun valore di progetto o bucket
# nell'immagine: BUCKET si passa solo se USE_SYNTHETIC=0.
ENV USE_SYNTHETIC="1"
ENV N_SAMPLES="4000"
ENV N_FEATURES="128"
ENV BATCH_SIZE="128"
ENV N_EPOCHS="5"
ENV LR="0.05"
ENV MODEL_PATH="lab5/ddp/"

# ── Entry point ───────────────────────────────────────────────────────────────
# Su Vertex AI rank e indirizzo del rank 0 si ricavano da CLUSTER_SPEC
# (_parse_cluster_spec in train_ddp.py); con torchrun arrivano già nell'ambiente.
CMD ["python", "train_ddp.py"]
