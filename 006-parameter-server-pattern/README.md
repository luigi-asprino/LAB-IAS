# Lezione 06 — Parameter Server Pattern su GCP

Implementazione del **Parameter Server (PS) pattern** con PyTorch `torch.distributed.rpc` su Vertex AI Custom Training Jobs.

Lo stesso container Docker viene eseguito su tre VM simultaneamente: una funge da Parameter Server (detiene i pesi e l'optimizer), le altre due da worker (eseguono il training e comunicano con il PS via RPC).

---

## Struttura del progetto

```
006-parameter-server-pattern/
├── train_ps.py       # Script di training principale (PS + worker)
├── model.py          # Definizione di SimpleNet (235K parametri)
├── dataset.py        # Caricamento dataset da GCS via gcsfs
├── job_run.py        # Lancio del Custom Job su Vertex AI
├── upload_mnist.py   # Upload dataset MNIST su GCS (una tantum)
├── Dockerfile        # Immagine container basata su pytorch/pytorch
└── requirements.txt  # Dipendenze Python
```

---

## Prerequisiti

### Ambiente locale

```bash
# Attiva l'ambiente conda del corso
conda activate gcp

# Installa/aggiorna il client Vertex AI
pip install --upgrade google-cloud-aiplatform

# Autentica il CLI gcloud (per comandi gcloud)
gcloud auth login

# Autentica l'SDK Python (per job_run.py e upload_mnist.py)
# Da rinnovare ogni ~12 ore
gcloud auth application-default login

# Imposta il progetto di default 
gcloud config set project test-project-493915 # Modificare con il proprio project-id
gcloud config set ai/region europe-west4
```

### Risorse GCP necessarie (da modificare con le proprie)

| Risorsa | Nome | Note |
|---------|------|------|
| Progetto | `test-project-493915` | |
| Artifact Registry | `ias-repo` | Regione `europe-west4` |
| Bucket dataset | `ias-luigi-asprino-bucket` | Regione `us` — solo per dataset |
| Bucket staging | `ias-luigi-asprino-bucket-eu` | Regione `europe-west4` — obbligatorio per Vertex AI |

**Perché due bucket?** Il bucket `ias-luigi-asprino-bucket` era già esistente dalla Lezione 3 ed era stato creato in regione `us`. Vertex AI richiede che lo staging bucket sia nella **stessa regione** del job (`europe-west4`), quindi è stato necessario crearne uno nuovo. In un progetto creato da zero si userebbe un unico bucket in `europe-west4` per tutto (dataset, modelli e staging).

---

## Esecuzione passo per passo

### Passo 1 — Crea il repository Artifact Registry (una tantum)

Artifact Registry se non esiste va creato per mantenere l'immagine che viene usata

```bash
gcloud artifacts repositories create ias-repo \
  --project=test-project-493915 \
  --repository-format=docker \
  --location=europe-west4 \
  --description="Immagini container per il corso IAS"

# Verifica
gcloud artifacts repositories list \
  --project=test-project-493915 \
  --location=europe-west4
```

### Passo 2 — Crea il bucket di staging in europe-west4 (una tantum)

Il bucket di staging **deve essere nella stessa regione** del job Vertex AI. Il bucket `ias-luigi-asprino-bucket` è in `us` e non è utilizzabile come staging.

```bash
gcloud storage buckets create gs://ias-luigi-asprino-bucket-eu \
  --project=test-project-493915 \
  --location=europe-west4 \
  --uniform-bucket-level-access

# Verifica la regione
gcloud storage ls -L -b gs://ias-luigi-asprino-bucket-eu
# Output atteso: EUROPE-WEST4
```

### Passo 3 — Carica il dataset MNIST su GCS (una tantum)

```bash
python upload_mnist.py
```

Output atteso:
```
=== train ===
  Download https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz ...
  Caricato gs://ias-luigi-asprino-bucket/datasets/mnist/train/X.npy  shape=(60000, 784)  (187.5 MB)
  Caricato gs://ias-luigi-asprino-bucket/datasets/mnist/train/y.npy  shape=(60000,)  (0.5 MB)
...
Dataset MNIST disponibile su GCS.
```

Verifica:
```bash
gsutil ls -lh gs://ias-luigi-asprino-bucket/datasets/mnist/train/
```

### Passo 4 — Build e push del container

```bash
IMAGE="europe-west4-docker.pkg.dev/test-project-493915/ias-repo/train-ps:latest"

# Autorizza Docker ad accedere ad Artifact Registry
gcloud auth configure-docker europe-west4-docker.pkg.dev

# Build dell'immagine (richiede ~3-4 minuti al primo run)
docker build -t $IMAGE .

# Push su Artifact Registry
docker push $IMAGE

# Verifica che l'immagine sia presente
gcloud artifacts docker images list \
  europe-west4-docker.pkg.dev/test-project-493915/ias-repo \
  --include-tags
```

### Passo 5 — Lancia il job

```bash
python job_run.py
```

Output atteso:
```
Job creato con successo.
Job name: projects/714331511867/locations/europe-west4/customJobs/XXXXXXXXXXXXXXXXX
JOB_ID:   XXXXXXXXXXXXXXXXX
Stato:    3

Per monitorare i log:
  gcloud ai custom-jobs stream-logs XXXXXXXXXXXXXXXXX --region=europe-west4

Per controllare lo stato:
  gcloud ai custom-jobs describe XXXXXXXXXXXXXXXXX --region=europe-west4 --format='value(state)'
```

---

## Monitoraggio

### Stati del job

| Valore numerico | Nome | Significato |
|-----------------|------|-------------|
| 2 | `JOB_STATE_PENDING` | VM in provisioning (~3-5 min) |
| 3 | `JOB_STATE_RUNNING` | Training in corso |
| 4 | `JOB_STATE_SUCCEEDED` | Completato con successo |
| 5 | `JOB_STATE_FAILED` | Errore — leggere i log |

### Controlla lo stato

```bash
# Sostituire JOB_ID con il valore stampato da job_run.py
gcloud ai custom-jobs describe JOB_ID \
  --region=europe-west4 \
  --format='value(state)'
```

### Polling automatico ogni 30 secondi

```bash
watch -n 30 "gcloud ai custom-jobs describe JOB_ID \
  --region=europe-west4 \
  --format='value(state)'"
```

### Log in tempo reale (tutte le repliche)

```bash
gcloud ai custom-jobs stream-logs JOB_ID --region=europe-west4
```

### Log di una singola replica

```bash
# Parameter Server (workerpool0-0)
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="JOB_ID" AND resource.labels.task_name="workerpool0-0"' \
  --project=test-project-493915 \
  --format="table(timestamp, textPayload)" \
  --limit=50 --order=asc

# Worker 0 (workerpool1-0)
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="JOB_ID" AND resource.labels.task_name="workerpool1-0"' \
  --project=test-project-493915 \
  --format="table(timestamp, textPayload)" \
  --limit=50 --order=asc

# Worker 1 (workerpool1-1)
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="JOB_ID" AND resource.labels.task_name="workerpool1-1"' \
  --project=test-project-493915 \
  --format="table(timestamp, textPayload)" \
  --limit=50 --order=asc
```

### Recupera il JOB_ID dell'ultimo job lanciato

```bash
gcloud ai custom-jobs list \
  --project=test-project-493915 \
  --region=europe-west4 \
  --filter="displayName:lezione6" \
  --sort-by="~createTime" \
  --limit=3 \
  --format="table(name.segment(-1), displayName, state, createTime)"
```

---

## Verifica del completamento

### 1. Controlla che lo stato sia SUCCEEDED

```bash
gcloud ai custom-jobs describe JOB_ID \
  --region=europe-west4 \
  --format='value(state)'
# Output atteso: JOB_STATE_SUCCEEDED
```

### 2. Verifica il modello salvato su GCS

```bash
gsutil ls -lh gs://ias-luigi-asprino-bucket/models/lezione6/
# Output atteso:
#   920 KiB  gs://ias-luigi-asprino-bucket/models/lezione6/model.pth
```

### 3. Cosa cercare nei log per confermare il successo

```
# PS: avvio corretto
[ps] ParameterServer in attesa di connessioni RPC...

# Worker: training attivo con loss in discesa
[worker-0] epoch=0 loss=2.3120
[worker-0] epoch=0 loss=2.2719
[worker-0] epoch=0 loss=2.0634
...

# Worker: shutdown ordinato
[worker-0] Shutdown completato.
[worker-1] Shutdown completato.
[ps] Shutdown completato.
```

L'asincronicità del training è visibile dai **numeri di step diversi** tra worker-0 e worker-1 nei log del PS — è il comportamento atteso del pattern PS con `rpc_async`.

---

## Tempistiche attese

| Fase | Durata |
|------|--------|
| Provisioning VM | 3–5 min |
| Handshake RPC | 10–30 sec |
| Training (5 epoch, CPU) | ~5 min |
| Salvataggio modello | ~5 sec |
| **Totale** | **~10–12 min** |

---

## Errori comuni e soluzioni

### `Repository "ias-repo" not found`
Artifact Registry non crea il repository automaticamente. Eseguire il Passo 1.

### `Bucket in location 'us' != 'europe-west4'`
Il bucket di staging deve essere in `europe-west4`. Usare `ias-luigi-asprino-bucket-eu`.

### `Reauthentication is needed`
Le Application Default Credentials sono scadute (~12h). Rieseguire:
```bash
gcloud auth application-default login
```

### `KeyError: 'ps'` o `KeyError: 'worker'`
Vertex AI usa `"workerpool0"` e `"workerpool1"`, non `"ps"` e `"worker"`.
Il codice in `train_ps.py` gestisce già correttamente questo caso.

### `invalid literal for int('')` su TASK_INDEX
Vertex AI inietta `TASK_TYPE` e `TASK_INDEX` come stringhe vuote.
Il codice usa `CLUSTER_SPEC["task"]` per leggere ruolo e indice — già corretto.

### `PyRRef has no attribute 'model'`
I metodi di classe non funzionano come target RPC in PyTorch.
Il codice usa funzioni standalone con `ps_rref.local_value()` — già corretto.

### `Parent directory /gcs/... does not exist`
Il filesystem `/gcs/` non è montato nei container Vertex AI.
Il codice usa `google.cloud.storage` SDK per salvare su GCS — già corretto.

### Job in `JOB_STATE_FAILED` senza messaggio chiaro
Leggere prima l'errore di alto livello, poi i log della replica che ha fallito:
```bash
# Errore di alto livello
gcloud ai custom-jobs describe JOB_ID \
  --region=europe-west4 \
  --format="yaml(error)"

# Log dettagliati del PS
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="JOB_ID" AND resource.labels.task_name="workerpool0-0"' \
  --project=test-project-493915 \
  --format="table(timestamp, textPayload)" \
  --limit=30 --order=asc
```

---

## Architettura del sistema

```
┌─────────────────────────────────────────────────────────┐
│                    Vertex AI Cluster                     │
│                                                         │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │   workerpool0-0   │         │   workerpool1-0   │     │
│  │  Parameter Server │◄──push──│    Worker 0       │     │
│  │                   │──pull──►│                   │     │
│  │  • SimpleNet      │         │  • Forward pass   │     │
│  │  • SGD optimizer  │◄──push──│  • Backward pass  │     │
│  │  • Lock           │──pull──►│                   │     │
│  └──────────────────┘         └──────────────────┘     │
│           ▲  │                                           │
│     push  │  │ pull                                      │
│           │  ▼                                           │
│         ┌──────────────────┐                            │
│         │   workerpool1-1   │                            │
│         │    Worker 1       │                            │
│         │  • Forward pass   │                            │
│         │  • Backward pass  │                            │
│         └──────────────────┘                            │
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
  gs://ias-luigi-asprino-bucket-eu   gs://ias-luigi-asprino-bucket
  (staging Vertex AI)                (dataset + modello salvato)
```

**push** = `rpc_async` — non bloccante → genera l'asincronicità
**pull** = `rpc_sync` — bloccante → il worker aspetta i pesi prima del forward
