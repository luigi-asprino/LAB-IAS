# Laboratorio di Intelligenza Artificiale Scalabile

Materiali delle lezioni sincrone del corso di **Intelligenza Artificiale Scalabile** (IAS),
Università Telematica San Raffaele Roma, anno accademico 2026/27.

- Pagina del corso: <https://www.uniroma5.it/insegnamenti/0322509INGINF05II>
- Docente: Luigi Asprino — <luigi.asprino@uniroma5.it>
- Slide introduttive del laboratorio (calendario, bonus di frequenza, regole del progetto): [Introduzione al laboratorio](https://docs.google.com/presentation/d/1W3lTubEjEsBAc_SG2fAksUT1lfr2D6f6bnXEAqt9j9A/edit?usp=sharing)

Il laboratorio implementa su **Google Cloud Platform** i pattern architetturali per il
machine learning distribuito visti nelle videolezioni: ingestion di dataset distribuiti,
training distribuito, fault-tolerance, serving di modelli di grandi dimensioni e workflow.

---

## Calendario

Le lezioni sincrone si svolgono dalle 14:00 alle 16:00 e saranno registrate.

| Lezione | Data | Argomento | Materiali |
|---|---|---|---|
| 1 | 8 ottobre | Introduzione a GCP e setup dell'ambiente ML | [`001-intro-gcp`](001-intro-gcp), [slide](https://docs.google.com/presentation/d/1VqRw35Wx7zyRx-SMacAycdQf6D-mmHPHHlw__3Za5qw/edit?usp=sharing) |
| 2 | 15 ottobre | Acceleratori in GCP (GPU e TPU) | [`002-acceleratori`](002-acceleratori), [slide](https://docs.google.com/presentation/d/1lgPl4FqEK8QbgR_6Km86P-irWEHhK2PU5EzdPpDFzQM/edit?usp=sharing) |
| 3 | 22 ottobre | Google Cloud Storage: gestione di dataset distribuiti | in preparazione |
| 4 | 5 novembre | Vertex AI Workbench e containerizzazione dei notebook | in preparazione |
| 5 | 11 novembre | Containerizzazione e Custom Training Job su Vertex AI | in preparazione |
| 6 | 19 novembre | Parameter server pattern e collective communication pattern | in preparazione |
| 7 | 26 novembre | Elasticity e fault-tolerance: Spot VM e checkpoint automatici | in preparazione |
| 8 | 10 dicembre | Sharded service pattern: serving di modelli di grandi dimensioni | in preparazione |
| 9 | 17 dicembre | Fan-in/fan-out, synchronous/asynchronous e step memoization | in preparazione |


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

Le regole complete sono nelle [slide di introduzione al laboratorio](https://docs.google.com/presentation/d/1W3lTubEjEsBAc_SG2fAksUT1lfr2D6f6bnXEAqt9j9A/edit?usp=sharing) e sulla pagina del corso.

---

## Domande e supporto

Segnalate dubbi e difficoltà durante le lezioni oppure scrivete a
<luigi.asprino@uniroma5.it>.
