# Laboratorio di Intelligenza Artificiale Scalabile

Materiali delle lezioni sincrone del corso di **Intelligenza Artificiale Scalabile** (IAS),
Università Telematica San Raffaele Roma, anno accademico 2026/27.

- Pagina del corso: <https://www.uniroma5.it/insegnamenti/0322509INGINF05II>
- Docente: Luigi Asprino — <luigi.asprino@uniroma5.it>

Il laboratorio implementa su **Google Cloud Platform** i pattern architetturali per il
machine learning distribuito visti nelle videolezioni: ingestion di dataset distribuiti,
training distribuito, fault-tolerance, serving di modelli di grandi dimensioni e workflow.

---

## Calendario

Le lezioni sincrone si svolgono dalle 14:00 alle 16:00 e saranno registrate.

| Lezione | Data | Argomento | Cartella |
|---|---|---|---|
| 1 | 8 ottobre | Introduzione a GCP e setup dell'ambiente ML | [`001-intro-gcp`](001-intro-gcp) |
| 2 | 15 ottobre | Acceleratori in GCP (GPU e TPU) | in preparazione |
| 3 | 22 ottobre | Google Cloud Storage: gestione di dataset distribuiti | [`003-sharding-batching`](003-sharding-batching) |
| 4 | 5 novembre | Vertex AI Workbench e containerizzazione dei notebook | [`004-training`](004-training) |
| 5 | 12 novembre | Containerizzazione e Custom Training Job su Vertex AI | [`005-training`](005-training) |
| 6 | 19 novembre | Parameter server pattern e collective communication pattern | [`006-parameter-server-pattern`](006-parameter-server-pattern), [`007-collective-communication`](007-collective-communication) |
| 7 | 26 novembre | Elasticity e fault-tolerance: Spot VM e checkpoint automatici | [`008-fault-tolerance`](008-fault-tolerance) |
| 8 | 10 dicembre | Sharded service pattern: serving di modelli di grandi dimensioni | [`009-sharded-service`](009-sharded-service) |
| 9 | 17 dicembre | Fan-in/fan-out, synchronous/asynchronous e step memoization | [`010-fan-in-fan-out-e-sync-async`](010-fan-in-fan-out-e-sync-async), [`011-step-memoization`](011-step-memoization) |


I materiali di ogni lezione vengono aggiornati prima della lezione stessa: eseguite
`git pull` all'inizio di ogni incontro.


---

## Prerequisiti

- Python 3.10 o superiore, Jupyter
- Un account Google Cloud con crediti didattici e un progetto dedicato al corso
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk?hl=it) (`gcloud`)
- [Docker](https://docs.docker.com/get-docker/), consigliato dalla lezione 4
- Conoscenze di base di PyTorch o TensorFlow

## Setup iniziale

```bash
git clone https://github.com/luigi-asprino/LAB-IAS.git
cd LAB-IAS

# Login e progetto di default
gcloud init
gcloud auth application-default login
```

I comandi completi della prima lezione sono in [`001-intro-gcp/comandi.txt`](001-intro-gcp/comandi.txt).

---

## Costi e sicurezza

- **Impostate un budget alert** sul progetto GCP durante la prima lezione.
- **Spegnete le risorse** a fine attività: VM, istanze Workbench, endpoint e job in esecuzione
  continuano a consumare crediti.
- Quando possibile usate **Spot VM**, che riducono il costo del training.
- **Non caricate mai su Git le chiavi dei service account** (`key.json`): il file è già
  escluso nel `.gitignore`. Preferite `gcloud auth application-default login`.

---

## Esame e progetto

L'esame prevede una prova teorica obbligatoria e un **progetto opzionale** che dà fino a
5 punti di bonus: un sistema di ML distribuito che addestra un nuovo modello e implementa
uno o più pattern architetturali visti a lezione. Il progetto va proposto e approvato dal
docente prima di iniziare, e consegnato almeno due settimane prima della prova intermedia.

Le regole complete sono nelle slide di introduzione al laboratorio e sulla pagina del corso.

---

## Domande e supporto

Segnalate dubbi e difficoltà durante le lezioni oppure scrivete a
<luigi.asprino@uniroma5.it>.
