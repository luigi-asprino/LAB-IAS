# Lezione 05 (parte 2) — Collective Communication Pattern su GCP

## Obiettivi

Al termine di questa lezione lo studente sarà in grado di:

1. Descrivere il funzionamento dell'all-reduce ring-based e perché è bandwidth-optimal
2. Osservare le primitive collettive (`broadcast`, `all_reduce`, `all_gather`, `scatter`) con `torch.distributed`
3. Scrivere uno script PyTorch DDP che usa NCCL/gloo come backend collettivo
4. Lanciare un Custom Training Job multi-worker su Vertex AI
5. Confrontare empiricamente la scaling efficiency tra 1, 2 e 4 worker

---

## Prerequisiti

- Python 3.9+, PyTorch ≥ 2.0
- Lezione 4 completata (Custom Training Job su Vertex AI funzionante) e parte 1 della lezione 5 (parameter server)
- Per la versione GCP: `gcloud` installata, Docker in esecuzione, un progetto GCP con fatturazione attiva

```bash
pip install torch torchvision
python -c "import torch; import torch.distributed as dist; print('gloo:', dist.is_gloo_available())"
```

---

## Struttura

```
005-parameter-server-collective-communication/collective-communication/gcp/
├── dataset.py          # SyntheticDataset + GCSDataset + build_dataloader()
├── model.py            # MLP a tre strati con param_bytes() per il benchmark
├── train_ddp.py        # ⭐ Script principale: training DDP completo
├── Dockerfile          # Container per Vertex AI Custom Training Job
├── job.yaml            # Generato al passo 3 dalle variabili del passo 0 (non va committato)
└── requirements.txt
```

## Versione GCP — Vertex AI Custom Training Job

### 0. Variabili

I propri valori si scrivono in **un solo punto**: questo blocco (lo stesso della parte 1, con in più `IMAGE_DDP`).
Sostituire `<ID_DEL_PROGETTO_GCP>` e `<NOME_DEL_BUCKET>`, poi eseguire nel terminale.
Se si apre un nuovo terminale, rieseguire questo blocco.

```bash
export PROJECT_ID="<ID_DEL_PROGETTO_GCP>"    # es. my-project-123456
export REGION="europe-west4"
export BUCKET="<NOME_DEL_BUCKET>"            # senza gs://, nella regione $REGION
export REPO="lab5-repo"
export IMAGE_DDP="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/lab5-ddp:v1"

echo $PROJECT_ID $BUCKET $IMAGE_DDP    # controllo: nessun valore deve contenere < >

gcloud auth login
gcloud config set project $PROJECT_ID
```

Repository Artifact Registry e bucket sono gli stessi della parte 1 (passi 1 e 2 di [`../../parameter-server/README.md`](../../parameter-server/README.md)).

| Variabile del container | Default | Significato |
|---|---|---|
| `USE_SYNTHETIC` | `1` | `1` = dati sintetici; `0` = MNIST da `gs://$BUCKET/datasets/mnist/train` |
| `BUCKET` | — | Bucket, obbligatorio solo con `USE_SYNTHETIC=0` |
| `N_SAMPLES`, `N_FEATURES` | `4000`, `128` | Dimensioni del dataset sintetico |
| `BATCH_SIZE`, `N_EPOCHS`, `LR` | `128`, `5`, `0.05` | Iperparametri del training |
| `MODEL_PATH` | `lab5/ddp/` | Prefisso nel bucket per il modello (solo con `USE_SYNTHETIC=0`) |

### 1. Build e push del container

```bash
cd gcp
gcloud auth configure-docker $REGION-docker.pkg.dev    # una volta per macchina
docker build --platform linux/amd64 -t $IMAGE_DDP .
docker push $IMAGE_DDP
```

### 2. Generare `job.yaml`

Il file si genera dalle variabili del passo 0, così immagine e bucket coincidono per costruzione
con quelli usati negli altri passi. Vertex AI ammette una sola replica nel primo pool:
per due rank si usano due pool da una replica ciascuno.

```bash
cat > job.yaml <<EOF
workerPoolSpecs:
- machineSpec:
    machineType: n1-standard-4
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: $IMAGE_DDP
    env:
    - {name: USE_SYNTHETIC, value: "1"}
    - {name: N_SAMPLES, value: "8000"}
    - {name: BATCH_SIZE, value: "256"}
    - {name: N_EPOCHS, value: "5"}
    - {name: BUCKET, value: "$BUCKET"}
- machineSpec:
    machineType: n1-standard-4
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: $IMAGE_DDP
    env:
    - {name: USE_SYNTHETIC, value: "1"}
    - {name: N_SAMPLES, value: "8000"}
    - {name: BATCH_SIZE, value: "256"}
    - {name: N_EPOCHS, value: "5"}
    - {name: BUCKET, value: "$BUCKET"}
EOF
cat job.yaml
```

### 3. Lancio del job

```bash
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=lab5-ddp \
  --config=job.yaml
```

### 4. Monitoring

```bash
export JOB_ID=$(gcloud ai custom-jobs list --region=$REGION \
  --filter="displayName=lab5-ddp" \
  --sort-by=~createTime --limit=1 --format="value(name.segment(-1))")

gcloud ai custom-jobs describe $JOB_ID --region=$REGION --format="value(state)"
gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION
```

### Quota GPU esaurita

Se il job restituisce `429 RESOURCE_EXHAUSTED` per la quota T4, togliere da `job.yaml` le righe
`acceleratorType` e `acceleratorCount` e rilanciare: il codice passa da solo a `backend='gloo'` su CPU.

### Pulizia

```bash
gcloud artifacts docker images delete $IMAGE_DDP --quiet
```

---

## Corrispondenza locale ↔ GCP

| Componente locale | Equivalente su Vertex AI |
|---|---|
| `torchrun --nproc_per_node=2` | 2 `workerPoolSpecs` da 1 replica (il primo pool ammette una sola replica) |
| `backend='gloo'` | `backend='nccl'` (GPU) |
| `device=cpu` | `device=cuda` |
| `RANK`, `WORLD_SIZE`, `MASTER_ADDR` iniettate da `torchrun` | Ricavate da `CLUSTER_SPEC` in `train_ddp.py` |
| Socket loopback `127.0.0.1` | Rete VPC ad alta banda tra VM |

Il training loop di `train_ddp.py` è identico nei due ambienti: la versione GCP aggiunge solo `_parse_cluster_spec()`.
Cambiano il launcher e il backend: `torchrun` → Vertex AI, `gloo` → `nccl`.

---

## Concetti chiave

- **All-reduce** = aggregazione globale senza nodo centrale; `⊕` deve essere associativa
- **Ring-based all-reduce**: Reduce-Scatter + All-Gather → volume `2(N-1)/N·|g|` per processo → bandwidth-optimal
- **DDP**: `loss.backward()` → NCCL esegue all-reduce → `optimizer.step()` aggiorna parametri coerenti su tutti i rank
- **DistributedSampler**: garantisce partizioni disgiunte del dataset; `set_epoch(epoch)` obbligatorio ad ogni epoch
- **Scaling efficiency**: `speedup / N × 100%`; < 100% a causa di overhead di comunicazione e sincronizzazione
