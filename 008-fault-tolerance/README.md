# Lezione 8 — Fault-Tolerance

## Struttura del repository

```
008-spot-checkpoint/
├── local/
│   ├── train_spot.py       # script principale con SIGTERM handler
│   ├── checkpoint.py       # logica di checkpoint (save/load su GCS e locale)
│   ├── model.py            # MLP (identico a L7)
│   ├── dataset.py          # dataset sintetico + GCS (identico a L7)
│   └── requirements.txt
└── gcp/
    ├── Dockerfile
    ├── job_config.yaml     # Spot VM + restartJobOnWorkerRestart
    └── requirements.txt
```

## Esercizi locali (senza GCP)

### Prerequisiti

```bash
cd local/
pip install -r requirements.txt
```

### Training da zero

```bash
USE_SYNTHETIC=1 N_EPOCHS=20 python train_spot.py
```

Osserva nei log:
- `[ckpt] Nessun checkpoint trovato — training da zero`
- `[ckpt] Checkpoint salvato: epoch=0, step=...` (ogni epoch)

### Simulazione preemption con SIGTERM

**Terminale 1** — avvia il training:
```bash
USE_SYNTHETIC=1 N_EPOCHS=200 python train_spot.py
```

**Terminale 2** — dopo qualche epoch, invia SIGTERM:
```bash
kill -SIGTERM $(pgrep -f train_spot.py)
```

Osserva nel Terminale 1:
```
[SIGTERM] Segnale ricevuto! Salvataggio checkpoint di emergenza...
[ckpt] Checkpoint salvato: epoch=3, step=...
[SIGTERM] Checkpoint salvato all'epoch 3. Il job verrà riavviato da Vertex AI.
```

### Ripresa dal checkpoint

```bash
USE_SYNTHETIC=1 N_EPOCHS=20 python train_spot.py
```

Osserva:
- `[ckpt] Ripresa da epoch=4, step=...`  ← riparte dall'epoch successiva


###  Verifica del checkpoint

```python
import torch
state = torch.load("/tmp/checkpoints/lezione8/checkpoint_latest.pt",
                   map_location="cpu", weights_only=False)
print(f"Epoch: {state['epoch']}")
print(f"Loss:  {state['loss']:.4f}")
print(f"Chiavi model: {list(state['model'].keys())[:3]}")
```

### Impatto di CHECKPOINT_INTERVAL

```bash
# Checkpoint ogni 5 epoch (meno overhead I/O, più training perso)
USE_SYNTHETIC=1 N_EPOCHS=20 CHECKPOINT_INTERVAL=5 python train_spot.py
```

## Deploy su Vertex AI (versione GCP)

### 1. Build e push del container

```bash
PROJECT=test-project-493915
REGION=europe-west4
REPO=ias-repo

cd gcp/

# Copia i file Python dalla directory local/
cp ../local/train_spot.py .
cp ../local/checkpoint.py .
cp ../local/model.py .
cp ../local/dataset.py .

docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab08-spot:v1 .
docker push ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab08-spot:v1
```

### 2. Lancio del job

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name=lab08-spot-checkpoint \
  --config=job_config.yaml \
  --project=test-project-493915
```

### 3. Monitoraggio

```bash
# Lista job
gcloud ai custom-jobs list \
  --region=europe-west4 \
  --project=test-project-493915

# Dettagli job (sostituire JOB_ID)
gcloud ai custom-jobs describe JOB_ID \
  --region=europe-west4 \
  --project=test-project-493915
```

### 4. Simulazione preemption dalla console

Dalla Console GCP → Vertex AI → Training → Custom Jobs → seleziona il job → **Cancel**.

Poi riavvia:
```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name=lab08-spot-checkpoint-ripresa \
  --config=job_config.yaml \
  --project=test-project-493915
```

### 5. Verifica checkpoint su GCS

```bash
# Lista i checkpoint salvati
gsutil ls gs://ias-luigi-asprino-bucket-eu/checkpoints/lezione8/

# Metadati del checkpoint più recente
gsutil stat gs://ias-luigi-asprino-bucket-eu/checkpoints/lezione8/checkpoint_latest.pt

# Download e ispezione
gsutil cp gs://ias-luigi-asprino-bucket-eu/checkpoints/lezione8/checkpoint_latest.pt /tmp/
python3 -c "
import torch
s = torch.load('/tmp/checkpoint_latest.pt', map_location='cpu', weights_only=False)
print(f'Epoch: {s[\"epoch\"]}')
print(f'Step:  {s[\"step\"]}')
print(f'Loss:  {s[\"loss\"]:.4f}')
"
```

## Domande di discussione

1. Cosa succederebbe se l'handler SIGTERM non fosse registrato?
2. Perché solo `epoch + 1` nel `load_checkpoint` e non `epoch`?
3. Con N worker DDP, chi deve salvare il checkpoint? Perché solo rank 0?
4. Qual è il trade-off tra `CHECKPOINT_INTERVAL=1` e `CHECKPOINT_INTERVAL=10`?
5. Spot vs Standard: a parità di prezzo totale, quante interruzioni si possono
   tollerare prima che il risparmio svanisca?
