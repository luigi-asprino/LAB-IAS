# Lezione 05 — Parameter Server e Collective Communication

Due pattern per sincronizzare il training data-parallel:

1. **Parameter Server** — un nodo centrale tiene i pesi; i worker fanno pull dei parametri e push dei gradienti, anche in modo asincrono.
2. **Collective Communication** — tutti i processi sono simmetrici e si scambiano i gradienti con un all-reduce (DistributedDataParallel).

Slide: [005-ParameterServer-Collective-GCP](https://docs.google.com/presentation/d/1rgdHeoP0ISOmM_teruvDDxEqRS4xXKwomtdV0FIjP28/edit?usp=sharing)

## Struttura

```
005-parameter-server-collective-communication/
├── parameter-server/            # Parte 1 — PS con torch.distributed.rpc su Vertex AI
│   ├── train_ps.py              # PS e worker (pull sincrono, push asincrono, staleness)
│   ├── job_run.py               # Lancio del Custom Job (1 PS + 2 worker)
│   ├── dataset.py               # MNIST da GCS o sintetico, uno shard per worker
│   ├── model.py, Dockerfile, requirements.txt, upload_mnist.py
│   └── README.md                # Deploy, monitoraggio, errori comuni, prova in locale
└── collective-communication/    # Parte 2 — primitive collettive e DDP
    ├── local/                   # torchrun su CPU (gloo), nessun costo
    │   ├── primitives.py        # broadcast, reduce, all_reduce, all_gather, scatter + ring all-reduce a mano
    │   └── train_ddp.py         # DDP + DistributedSampler, misura di throughput e scaling
    ├── gcp/                     # Stesso training su Vertex AI (GPU, NCCL)
    └── README.md
```

## Configurazione

Come nella lezione 4, nessun file contiene valori di un progetto specifico: progetto, regione, bucket
e immagini si impostano **una volta** con il blocco di variabili del passo 0 e tutti i comandi e gli
script li leggono dall'ambiente.

```bash
export PROJECT_ID="<ID_DEL_PROGETTO_GCP>"
export REGION="europe-west4"
export BUCKET="<NOME_DEL_BUCKET>"            # senza gs://, nella regione $REGION
export REPO="lab5-repo"
export IMAGE_PS="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/lab5-ps:v1"
export IMAGE_DDP="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/lab5-ddp:v1"
```

Conviene salvarlo in `env.sh` (escluso dal `.gitignore`) e caricarlo con `source env.sh`.
Le prove in locale (`USE_SYNTHETIC=1`, `torchrun`) non richiedono alcuna variabile.

## Percorso consigliato

| Parte | Cosa fare | Dove |
|---|---|---|
| 1 | Prova del PS in locale con `USE_SYNTHETIC=1`, poi su Vertex AI | [`parameter-server/README.md`](parameter-server/README.md) |
| 1 | Nei log: versione letta da ogni worker e staleness degli update | idem |
| 2 | `torchrun --nproc_per_node=4 primitives.py` | [`collective-communication/README.md`](collective-communication/README.md) |
| 2 | `train_ddp.py` con 1, 2 e 4 processi: speedup e scaling efficiency | idem |
| 2 | (opzionale) DDP su Vertex AI con GPU | [`collective-communication/gcp/README.md`](collective-communication/gcp/README.md) |
