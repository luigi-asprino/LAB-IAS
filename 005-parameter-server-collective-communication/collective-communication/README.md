# Lezione 05 (parte 2) — Collective Communication Pattern

> Seconda metà della lezione 5, dopo il [parameter server](../parameter-server).

## Obiettivi

Al termine di questa lezione lo studente sarà in grado di:

1. Descrivere il funzionamento dell'all-reduce ring-based e perché è bandwidth-optimal
2. Osservare le primitive collettive (`broadcast`, `reduce`, `all_reduce`, `all_gather`, `scatter`) con `torch.distributed`
3. Scrivere uno script PyTorch DDP che usa NCCL/gloo come backend collettivo
4. Lanciare un Custom Training Job multi-worker su Vertex AI
5. Confrontare empiricamente la scaling efficiency tra 1 e 2 worker

---

## Struttura della directory

```
005-parameter-server-collective-communication/collective-communication/
├── local/                  # Esercizi eseguibili in locale con torchrun (nessuna GPU richiesta)
│   ├── primitives.py       # broadcast, reduce, all_reduce, all_gather, scatter + ring all-reduce a mano
│   ├── dataset.py          # SyntheticDataset + GCSDataset + build_dataloader()
│   ├── model.py            # MLP a tre strati con param_bytes() per il benchmark
│   └── train_ddp.py        # Script principale: training DDP completo
│
└── gcp/                    # Versione per Vertex AI Custom Training Job
    ├── dataset.py          # Stesso dataset, include lettura da GCS
    ├── model.py            # Stesso modello
    ├── train_ddp.py        # train_ddp con parsing di CLUSTER_SPEC per Vertex AI
    ├── Dockerfile          # Container per il job multi-worker
    ├── (job.yaml)          # Generato dal README di gcp/ con le proprie variabili
    └── requirements.txt
```

Il codice nelle due cartelle è quasi identico. L'unica differenza rilevante è in `gcp/train_ddp.py`, che include la funzione `_parse_cluster_spec()` per leggere le variabili di topologia del cluster da `CLUSTER_SPEC`, la variabile d'ambiente iniettata da Vertex AI al posto delle variabili `RANK`/`WORLD_SIZE`/`MASTER_ADDR` iniettate da `torchrun`.

---

## Versione locale

### Prerequisiti

```bash
pip install torch
python -c "import torch.distributed as dist; print('gloo:', dist.is_gloo_available())"
```

### Lancio

```bash
cd local

# Primitive collettive, una per volta (4 processi)
torchrun --nproc_per_node=4 primitives.py

# 1 processo — baseline senza comunicazione collettiva
torchrun --nproc_per_node=1 train_ddp.py

# 2 processi — DDP esegue all-reduce su ogni backward
torchrun --nproc_per_node=2 train_ddp.py

# 4 processi
torchrun --nproc_per_node=4 train_ddp.py
```

### Variabili d'ambiente configurabili

| Variabile | Default | Descrizione |
|---|---|---|
| `USE_SYNTHETIC` | `1` | `1` = dataset sintetico, `0` = GCS |
| `N_SAMPLES` | `4000` | Campioni nel dataset sintetico |
| `N_FEATURES` | `128` | Feature per campione |
| `BATCH_SIZE` | `128` | Campioni per batch per rank |
| `N_EPOCHS` | `5` | Numero di epoch |
| `LR` | `0.05` | Learning rate SGD |

Esempio con parametri personalizzati:

```bash
N_EPOCHS=3 N_SAMPLES=8000 BATCH_SIZE=256 torchrun --nproc_per_node=2 train_ddp.py
```

### Warning attesi (non sono errori)

```
W ... Setting OMP_NUM_THREADS environment variable for each process to be 1
W ... Redirects are currently not supported in Windows or MacOs.
```

Il primo è torchrun che limita i thread OpenMP per evitare sovraccarico. Il secondo indica che l'output dei processi non viene separato per file su macOS — entrambi benigni.

---

## Versione GCP — Vertex AI Custom Training Job

Le istruzioni complete sono in [`gcp/README.md`](gcp/README.md): come nella lezione 4, i propri valori
(progetto, regione, bucket, immagine) si impostano una volta nel passo 0 e `job.yaml` si genera da quelle variabili.
Nessun file del repository contiene valori di un progetto specifico.

---

## Corrispondenza locale ↔ GCP

| Locale (torchrun) | GCP (Vertex AI) |
|---|---|
| `torchrun --nproc_per_node=2` | 2 `workerPoolSpecs` da 1 replica ciascuno |
| `backend='gloo'` | `backend='nccl'` (con GPU) |
| `device=cpu` | `device=cuda` |
| `RANK`, `WORLD_SIZE`, `MASTER_ADDR` iniettate da torchrun | Ricavate da `CLUSTER_SPEC` iniettata da Vertex AI |
| Comunicazione via socket loopback | Comunicazione via rete VPC |

---

## Concetti chiave

- **Collective communication pattern**: operazione atomica in cui tutti i processi del gruppo partecipano; nessun mittente o destinatario esplicito
- **All-reduce**: ogni processo ottiene la somma (o media) globale dei contributi di tutti; primitiva centrale di DDP
- **Ring-based all-reduce**: Reduce-Scatter + All-Gather; volume per processo = `2·(N-1)/N·|g|`; bandwidth-optimal; implementato da NCCL
- **DDP**: `loss.backward()` intercetta il backward e chiama all-reduce sui gradienti prima di restituire il controllo; `optimizer.step()` aggiorna parametri già sincronizzati
- **DistributedSampler**: assegna partizioni disgiunte del dataset a ogni rank; `set_epoch(epoch)` obbligatorio ad ogni epoch per il corretto shuffle
- **Scaling efficiency**: `speedup / N × 100%`; inferiore a 100% a causa di overhead di comunicazione e sincronizzazione
