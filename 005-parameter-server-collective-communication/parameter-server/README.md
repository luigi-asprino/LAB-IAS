# Lezione 05 (parte 1) — Parameter Server Pattern su GCP

> Prima metà della lezione 5. La seconda metà (collective communication con DDP) è in [`../collective-communication`](../collective-communication).

Implementazione del **Parameter Server (PS) pattern** con PyTorch `torch.distributed.rpc` su Vertex AI Custom Training Jobs.

Lo stesso container Docker viene eseguito su tre VM simultaneamente: una funge da Parameter Server (detiene i pesi e l'optimizer), le altre due da worker (eseguono il training e comunicano con il PS via RPC).

---

## Struttura del progetto

```
005-parameter-server-collective-communication/parameter-server/
├── train_ps.py       # Script di training principale (PS + worker)
├── model.py          # Definizione di SimpleNet (235K parametri)
├── dataset.py        # Dataset da GCS (gcsfs) o sintetico, con sharding per worker
├── job_run.py        # Lancio del Custom Job su Vertex AI
├── upload_mnist.py   # Upload dataset MNIST su GCS (una tantum)
├── Dockerfile        # Immagine container basata su pytorch/pytorch
└── requirements.txt  # Dipendenze Python
```

---

## Prerequisiti

- Un progetto GCP con fatturazione attiva
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) installata
- Docker installato e in esecuzione
- Python con `google-cloud-aiplatform`, `google-cloud-storage` e `numpy` (`pip install -r requirements.txt google-cloud-aiplatform`)

## 0. Variabili

I propri valori si scrivono in **un solo punto**: questo blocco. Sostituire `<ID_DEL_PROGETTO_GCP>`
e `<NOME_DEL_BUCKET>` con i propri valori (tra le virgolette), poi eseguire nel terminale.
Tutti i comandi successivi, `upload_mnist.py` e `job_run.py` prendono i valori da qui.
Se si apre un nuovo terminale, rieseguire questo blocco.

```bash
export PROJECT_ID="<ID_DEL_PROGETTO_GCP>"    # es. my-project-123456
export REGION="europe-west4"
export BUCKET="<NOME_DEL_BUCKET>"            # senza gs://, es. ias-mario-rossi-bucket
export REPO="lab5-repo"
export IMAGE_PS="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/lab5-ps:v1"

echo $PROJECT_ID $BUCKET $IMAGE_PS     # controllo: nessun valore deve contenere < >

gcloud auth login
gcloud auth application-default login   # per job_run.py e upload_mnist.py (da rinnovare ogni ~12 ore)
gcloud config set project $PROJECT_ID
```

Comodo: salvare il blocco in un file `env.sh` (escluso dal `.gitignore`) e caricarlo con `source env.sh`.

### Parametri

Nessun file contiene valori di progetto o bucket: tutto si legge da variabili d'ambiente.

| Variabile | Usata da | Obbligatoria | Default | Significato |
|---|---|---|---|---|
| `PROJECT_ID` | `job_run.py`, `upload_mnist.py` | sì | — | Progetto GCP |
| `REGION` | `job_run.py` | no | `europe-west4` | Regione del job (bucket e repository devono stare qui) |
| `BUCKET` | tutti | sì (tranne con `USE_SYNTHETIC=1`) | — | Bucket per dataset, staging e modello, senza `gs://` |
| `IMAGE_PS` | `job_run.py` | sì | — | Immagine su Artifact Registry |
| `N_WORKERS` | `job_run.py` | no | `2` | Repliche del pool dei worker |
| `N_EPOCHS` | `train_ps.py` | no | `5` | Epoch per worker (passata ai container da `job_run.py`) |
| `MODEL_PATH` | `train_ps.py` | no | `lab5/ps/model.pth` | Percorso del modello nel bucket |
| `USE_SYNTHETIC` | `train_ps.py` | no | `0` | `1` = dati sintetici, nessun bucket necessario |
| `LOG_EVERY` | `train_ps.py` | no | `50` | Ogni quanti step stampare i log |

Un unico bucket nella regione del job serve per tutto: dataset, file di staging di Vertex AI e modello.
Se il bucket della lezione 3 è in un'altra regione (es. `us`), crearne uno nuovo in `$REGION` (passo 2).

---

## Esecuzione passo per passo

### Passo 1 — API e repository Artifact Registry (una tantum)

```bash
gcloud services enable artifactregistry.googleapis.com aiplatform.googleapis.com storage.googleapis.com

gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="IAS Lab 5"

gcloud artifacts repositories list --location=$REGION
```

### Passo 2 — Bucket nella regione del job (una tantum)

Vertex AI richiede che lo staging bucket sia **nella stessa regione** del job.
Saltare questo passo se `$BUCKET` esiste già in `$REGION`.

```bash
gcloud storage buckets create gs://$BUCKET --location=$REGION --uniform-bucket-level-access

# Verifica della regione: deve stampare la regione di $REGION (es. EUROPE-WEST4)
gcloud storage buckets describe gs://$BUCKET --format="value(location)"
```

### Passo 3 — Carica il dataset MNIST su GCS (una tantum)

```bash
python upload_mnist.py
```

Output atteso:
```
=== train ===
  Download https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz ...
  Caricato gs://<BUCKET>/datasets/mnist/train/X.npy  shape=(60000, 784)  (179.4 MB)
  Caricato gs://<BUCKET>/datasets/mnist/train/y.npy  shape=(60000,)  (0.5 MB)
...
Dataset MNIST disponibile su GCS.
```

Verifica:
```bash
gcloud storage ls gs://$BUCKET/datasets/mnist/train/
```

### Passo 4 — Build e push del container

```bash
# Autorizza Docker ad accedere ad Artifact Registry (una volta per macchina)
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build (~3-4 minuti al primo run). --platform serve sui Mac con chip Apple
docker build --platform linux/amd64 -t $IMAGE_PS .
docker push $IMAGE_PS

gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/$REPO --include-tags
```

### Passo 5 — Lancia il job

```bash
python job_run.py
```

`job_run.py` stampa la configurazione letta dall'ambiente e passa `BUCKET` e `N_EPOCHS` ai container.
Se manca una variabile obbligatoria si ferma con un messaggio `ERRORE: ...`.

Output atteso:
```
Job creato con successo.
Job name: projects/<PROJECT_NUMBER>/locations/europe-west4/customJobs/XXXXXXXXXXXXXXXXX
JOB_ID:   XXXXXXXXXXXXXXXXX
Stato:    3

Per monitorare i log:
  gcloud ai custom-jobs stream-logs XXXXXXXXXXXXXXXXX --region=europe-west4

Per controllare lo stato:
  gcloud ai custom-jobs describe XXXXXXXXXXXXXXXXX --region=europe-west4 --format='value(state)'
```

---

```
PS                                  Worker 0              Worker 1
│                                   │                     │
├─ ParameterServer() (prima di init_rpc)                  │
├─ init_rpc() ──────────────────────┤ init_rpc()          │ init_rpc()
│  (attende tutti)                  │  (attende tutti)    │  (attende tutti)
│◄──────────────── handshake completato ──────────────────┤
│                                   │                     │
├─ shutdown(graceful=True) ◄────────┤ get_ps_rref()       │
│  │                                │                     │
│  ├◄── get_params() ───────────────┤ rpc_sync            │
│  ├──► restituisce pesi ──────────►│                     │
│  │                                │ forward+backward    │
│  ├◄── update_params(grads) ───────┤ rpc_async           │
│  ├─── optimizer.step()            │                     │
│  │                                │  ...                │
│  ├◄── get_params() ──────────────────────────────────── │ rpc_sync
│  ├──► restituisce pesi ─────────────────────────────── ►│
│  │                                │                     │ forward+backward
│  ├◄── update_params(grads) ──────────────────────────── │ rpc_async
│  ├─── optimizer.step()            │                     │
│  │                                │  ...                │  ...
│  ├◄── shutdown segnalato ─────────┤                     │
│  ├◄── shutdown segnalato ────────────────────────────── │
│
└─ print("Shutdown completato.")
```

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
# ID dell'ultimo job lanciato (oppure il valore JOB_ID stampato da job_run.py)
export JOB_ID=$(gcloud ai custom-jobs list --region=$REGION \
  --filter="displayName=lab5-parameter-server" \
  --sort-by=~createTime --limit=1 --format="value(name.segment(-1))")
echo $JOB_ID

gcloud ai custom-jobs describe $JOB_ID \
  --region=$REGION \
  --format='value(state)'
```

### Polling automatico ogni 30 secondi

```bash
watch -n 30 "gcloud ai custom-jobs describe $JOB_ID \
  --region=$REGION \
  --format='value(state)'"
```

### Log in tempo reale (tutte le repliche)

```bash
gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION
```

### Log di una singola replica

```bash
# Parameter Server (workerpool0-0)
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="'$JOB_ID'" AND resource.labels.task_name="workerpool0-0"' \
  --project=$PROJECT_ID \
  --format="table(timestamp, textPayload)" \
  --limit=50 --order=asc

# Worker 0 (workerpool1-0)
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="'$JOB_ID'" AND resource.labels.task_name="workerpool1-0"' \
  --project=$PROJECT_ID \
  --format="table(timestamp, textPayload)" \
  --limit=50 --order=asc

# Worker 1 (workerpool1-1)
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="'$JOB_ID'" AND resource.labels.task_name="workerpool1-1"' \
  --project=$PROJECT_ID \
  --format="table(timestamp, textPayload)" \
  --limit=50 --order=asc
```

### Elenco degli ultimi job della lezione

```bash
gcloud ai custom-jobs list \
  --project=$PROJECT_ID \
  --region=$REGION \
  --filter="displayName:lab5" \
  --sort-by="~createTime" \
  --limit=3 \
  --format="table(name.segment(-1), displayName, state, createTime)"
```

---

## Verifica del completamento

### 1. Controlla che lo stato sia SUCCEEDED

```bash
gcloud ai custom-jobs describe $JOB_ID \
  --region=$REGION \
  --format='value(state)'
# Output atteso: JOB_STATE_SUCCEEDED
```

### 2. Verifica il modello salvato su GCS

```bash
gcloud storage ls -l gs://$BUCKET/lab5/ps/
# Output atteso: model.pth (circa 920 KiB)
```

### 3. Cosa cercare nei log per confermare il successo

```
# PS: avvio corretto
[ps] ParameterServer in attesa di connessioni RPC...

# PS: aggiornamenti dai due worker, con la staleness di ciascuno
[ps] version=100 da worker-1 (update n.51 di questo worker) staleness=1
[ps] version=150 da worker-0 (update n.75 di questo worker) staleness=2

# Worker: training attivo con loss in discesa
[worker-0] epoch=0 step=50 version_letta=97 loss=2.1034
[worker-0] epoch=1 step=500 version_letta=998 loss=0.6120

# Shutdown ordinato e riepilogo sul PS
[worker-0] Shutdown completato.
[worker-1] Shutdown completato.
[ps] Shutdown completato.
[ps] Aggiornamenti totali: 4690 | per worker: {'worker-0': 2345, 'worker-1': 2345}
[ps] Staleness media=1.00 massima=9
```

L'asincronicità è visibile in due modi: la `version_letta` di un worker avanza di circa 2 a ogni suo step (l'altro worker aggiorna i pesi nel frattempo) e la **staleness** registrata dal PS è quasi sempre ≥ 1: il gradiente viene applicato a pesi più recenti di quelli su cui è stato calcolato. I valori esatti cambiano a ogni esecuzione.

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

## Prova in locale (senza GCP)

Il pattern si può provare sul proprio computer con tre processi che simulano i tre container. `USE_SYNTHETIC=1` sostituisce MNIST con dati casuali della stessa forma, e il modello viene salvato in `/tmp/model_ps.pth`.

```bash
pip install torch numpy
CL='"cluster":{"workerpool0":["localhost:29600"],"workerpool1":["localhost:2222","localhost:2223"]}'
export USE_SYNTHETIC=1 N_EPOCHS=2 LOG_EVERY=40
CLUSTER_SPEC="{$CL,\"task\":{\"type\":\"workerpool0\",\"index\":0}}" python train_ps.py &
CLUSTER_SPEC="{$CL,\"task\":{\"type\":\"workerpool1\",\"index\":0}}" python train_ps.py &
CLUSTER_SPEC="{$CL,\"task\":{\"type\":\"workerpool1\",\"index\":1}}" python train_ps.py
wait
```

---

## Errori comuni e soluzioni

### `ERRORE: variabile d'ambiente ... non impostata`
Il passo 0 non è stato eseguito nel terminale corrente (o i segnaposto `<...>` non sono stati sostituiti). Rieseguirlo, poi rilanciare il comando.

### `Repository "lab5-repo" not found`
Artifact Registry non crea il repository automaticamente. Eseguire il Passo 1.

### `Bucket in location 'us' != 'europe-west4'`
Lo staging bucket deve essere nella regione del job. Creare un bucket in `$REGION` (Passo 2), aggiornare `BUCKET` nel passo 0 e ricaricare il dataset (Passo 3).

### `exec format error` nei log del job
Immagine costruita per arm64 (Mac Apple Silicon). Ricostruire con `--platform linux/amd64` e rifare il push.

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

### Il modello salvato non è addestrato / nessuno stale gradient nei log
Succede se ogni worker crea il PS con `rpc.remote("ps", ParameterServer)`: sul PS nascono tante istanze quanti sono i worker, ognuno addestra la propria copia privata e il PS salva un'istanza mai aggiornata. Il codice attuale crea **una sola** istanza sul PS e i worker ne ottengono il riferimento con `rpc.rpc_sync("ps", get_ps_rref)`.

### `Parent directory /gcs/... does not exist`
Il filesystem `/gcs/` non è montato nei container Vertex AI.
Il codice usa `google.cloud.storage` SDK per salvare su GCS — già corretto.

### Job in `JOB_STATE_FAILED` senza messaggio chiaro
Leggere prima l'errore di alto livello, poi i log della replica che ha fallito:
```bash
# Errore di alto livello
gcloud ai custom-jobs describe $JOB_ID \
  --region=$REGION \
  --format="yaml(error)"

# Log dettagliati del PS
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="'$JOB_ID'" AND resource.labels.task_name="workerpool0-0"' \
  --project=$PROJECT_ID \
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
                          │
                          ▼
                  gs://$BUCKET
     (staging Vertex AI, dataset, modello salvato)
```

**push** = `rpc_async` — non bloccante → genera l'asincronicità
**pull** = `rpc_sync` — bloccante → il worker aspetta i pesi prima del forward
