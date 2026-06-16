#  Collective Communication Pattern su GCP

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
- Lezione 6 completata (Custom Training Job su Vertex AI funzionante)
- Per la versione GCP: `gcloud` autenticato, progetto `test-project-493915`

```bash
pip install torch torchvision
python -c "import torch; import torch.distributed as dist; print('gloo:', dist.is_gloo_available())"
```

---

## Struttura

```
007-collective-communication/
├── dataset.py          # SyntheticDataset + GCSDataset + build_dataloader()
├── model.py            # MLP a tre strati con param_bytes() per il benchmark
├── train_ddp.py        # ⭐ Script principale: training DDP completo
├── Dockerfile          # Container per Vertex AI Custom Training Job
├── job_config.yaml     # Configurazione del job multi-worker
└── requirements.txt
```

## Versione GCP — Vertex AI Custom Training Job

### 1. Build e push del container

```bash
PROJECT=test-project-493915
REGION=europe-west4
REPO=lab-repo
IMAGE=${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab07-ddp:latest

docker build -t ${IMAGE} .
docker push ${IMAGE}
```

### 2. Lancio del job

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name=lab07-ddp-collective \
  --config=job_config.yaml
```

### 3. Monitoring

```bash
# Lista job
gcloud ai custom-jobs list --region=europe-west4

# Log in tempo reale
gcloud ai custom-jobs stream-logs <JOB_ID> --region=europe-west4
```

---

## Corrispondenza locale ↔ GCP

| Componente locale | Equivalente su Vertex AI |
|---|---|
| `torchrun --nproc_per_node=2` | `replicaCount: 2` in `workerPoolSpecs` |
| `backend='gloo'` | `backend='nccl'` (GPU) |
| `device=cpu` | `device=cuda` |
| Variabili iniettate da `torchrun` | Stesse variabili iniettate da Vertex AI |
| Socket loopback `127.0.0.1` | Rete VPC ad alta banda tra VM |

Il codice di `train_ddp.py` è **identico** in entrambi gli ambienti.
Cambia solo il launcher e il backend: `torchrun` → Vertex AI, `gloo` → `nccl`.

---

## Concetti chiave

- **All-reduce** = aggregazione globale senza nodo centrale; `⊕` deve essere associativa
- **Ring-based all-reduce**: Reduce-Scatter + All-Gather → volume `2(N-1)/N·|g|` per processo → bandwidth-optimal
- **DDP**: `loss.backward()` → NCCL esegue all-reduce → `optimizer.step()` aggiorna parametri coerenti su tutti i rank
- **DistributedSampler**: garantisce partizioni disgiunte del dataset; `set_epoch(epoch)` obbligatorio ad ogni epoch
- **Scaling efficiency**: `speedup / N × 100%`; < 100% a causa di overhead di comunicazione e sincronizzazione
