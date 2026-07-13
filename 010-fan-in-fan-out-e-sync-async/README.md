# Fan-in/Fan-out & Sync/Async Pattern 

## Struttura

```
lezione10/
├── docker-compose.yml          # Redis + Celery worker + FastAPI app
├── docker/
│   ├── Dockerfile.celery
│   └── Dockerfile.fastapi
├── kfp_pipeline/
│   └── pipeline.py             # Sez. 2-3: Fan-out/Fan-in con KFP local runner
├── celery_app/
│   ├── tasks.py                # Sez. 5: task Celery (sync/async)
│   ├── main.py                 # Sez. 5: API FastAPI
│   └── client_demo.py          # Sez. 5: client di esempio
├── pubsub_demo/
│   ├── publisher.py            # Sez. 6: publisher Redis Pub/Sub
│   └── subscriber.py           # Sez. 6: subscriber Redis Pub/Sub
├── data/                       # shard locali di dati (simulano GCS)
└── requirements.txt
```

## Setup ambiente (conda)

```bash
conda create -n lab-ias-l10 python=3.11 -y
conda activate lab-ias-l10
pip install -r requirements.txt
```

L'ambiente serve per tutte le parti eseguite fuori da Docker (Parte 1: KFP
in locale, Parte 1bis: script Vertex AI, client di demo delle Parti 2-3).
I servizi in `docker-compose.yml` (Redis, Celery worker, FastAPI) girano
comunque in container e non richiedono l'ambiente conda per essere avviati
— serve pero' per eseguire `celery_app/client_demo.py` e gli script in
`pubsub_demo/` dalla propria macchina.

## Parte 1 — Fan-out / Fan-in con KFP (Sez. 2-3)

Con l'ambiente conda gia' attivo (vedi sopra):

```bash
python kfp_pipeline/pipeline.py
```

Il componente `preprocess_shard` prepara i dati, `dsl.ParallelFor` esegue
`train_on_shard` in parallelo per 4 seed diversi (**fan-out**), e
`collect_and_select_best` aggrega i risultati e sceglie il modello migliore
(**fan-in**) — tutto eseguito realmente in locale da `kfp.local`.

> **Richiede `kfp>=2.17`** (vedi `requirements.txt`): con versioni piu'
> datate dell'SDK, `dsl.ParallelFor` + `dsl.Collected` nello stesso DAG
> possono fallire in modo intermittente su `kfp.local` con
> `KeyError: 'for-loop-1'` — una race condition nella costruzione del
> grafo di dipendenze, risolta nelle versioni recenti dell'SDK. Se si
> installa l'ambiente da un `requirements.txt` che non fissa il minimo a
> 2.17, il resolver di pip/conda potrebbe scegliere una versione piu'
> vecchia e soggetta al problema; in tal caso il rimedio e'
> `pip install -U kfp` per allinearsi almeno alla 2.17.

> **Parallelismo effettivo in locale vs Vertex AI**: la pipeline dichiara
> `parallelism=4` (4 branch di training realmente simultanei), ed e' il
> valore corretto che viene rispettato interamente su Vertex AI Pipelines.
> `kfp.local.SubprocessRunner`, pero', limita SEMPRE la concorrenza
> effettiva a un massimo di **2 task paralleli**, indipendentemente dal
> valore di `parallelism` specificato — e' un limite fisso dell'SDK
> (`conservative_max = 2`), pensato per evitare un'esplosione di thread
> con `ParallelFor` annidati. Osservando i log si vedranno quindi al
> massimo 2 branch in esecuzione contemporanea (mai tutti e 4 insieme):
> non e' un errore del codice ne' dell'ambiente, ma un limite noto e
> intenzionale del runner locale.

> Nota: lo stesso identico codice (`dsl.pipeline`, `dsl.ParallelFor`,
> `dsl.Collected`) viene sottomesso a Vertex AI Pipelines in produzione,
> cambiando solo il backend di esecuzione (`aiplatform.PipelineJob` invece
> di `kfp.local.SubprocessRunner`).

## Parte 1bis — Stessa pipeline su GCP (Vertex AI Pipelines)

`kfp_pipeline/run_on_vertex_ai.py` riusa la stessa definizione `training_pipeline`
di `kfp_pipeline/pipeline.py` (nessuna riga di logica del DAG viene duplicata),
cambiando solo il backend di esecuzione da locale a Vertex AI Pipelines.

Prerequisiti:
```bash
pip install "google-cloud-aiplatform>=1.60"
gcloud auth application-default login
```

Serve un bucket GCS esistente per `pipeline_root` (es. `gs://ias-luigi-asprino-bucket-eu/pipeline-root`)
e le API `aiplatform.googleapis.com` e `storage.googleapis.com` abilitate sul progetto.

Esecuzione:
```bash
cd kfp_pipeline
python run_on_vertex_ai.py \
    --project test-project-493915 \
    --location europe-west4 \
    --pipeline-root gs://ias-luigi-asprino-bucket-eu/pipeline-root \
    --seeds 1 2 3 4 \
    --epochs 3
```

Lo script: compila `training_pipeline` in IR YAML, poi sottomette il job con
`aiplatform.PipelineJob` e stampa il link alla console Vertex AI per seguire
l'esecuzione (parallelismo dei branch, artefatti, log). Aggiungi `--no-wait`
per non bloccare il terminale in attesa del completamento.

## Parte 2 — Sync vs Async con Celery + Redis (Sez. 5)

```bash
docker compose up --build
```

In un altro terminale (con l'ambiente conda attivo):

```bash
python celery_app/client_demo.py
```

Osservare la differenza — STESSA operazione (inferenza), scala diversa:
- `/predict` (sync): una singola riga di feature, il client aspetta la
  risposta nella stessa richiesta.
- `/batch-predict` (async): un batch di righe, il client riceve subito un
  `task_id` e fa polling su `/batch-predict/status/{task_id}`.

Nota: il training vero e proprio di solito non parte da una richiesta
client via HTTP (viene schedulato o lanciato da una pipeline), per questo
qui il contrasto sync/async resta nello stesso dominio — inferenza singola
vs inferenza batch — invece di confrontare inferenza e training.

## Parte 3 — Event-driven con Pub/Sub (Sez. 6)

Con `docker compose up` già attivo (serve solo Redis) e l'ambiente conda attivo:

Terminale A:
```bash
python pubsub_demo/subscriber.py
```

Terminale B:
```bash
python pubsub_demo/publisher.py
```

Il subscriber riceve gli eventi "training completato" e triggera una finta
pipeline di deployment (scrive in `data/deployments.log`).

## Mapping concettuale locale → GCP

| Locale (oggi)              | GCP (produzione)                    |
|-----------------------------|--------------------------------------|
| `kfp.local.SubprocessRunner` | Vertex AI Pipelines                 |
| Celery + Redis broker        | Cloud Tasks                         |
| Redis Pub/Sub                | Google Cloud Pub/Sub                |
| Filesystem locale (`data/`)  | Google Cloud Storage                |
| Container Docker locale      | Cloud Run / Cloud Run Jobs          |
