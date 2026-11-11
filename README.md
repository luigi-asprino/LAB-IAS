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
| 3 | 22 ottobre | Google Cloud Storage: gestione di dataset distribuiti | [`003-sharding-batching`](003-sharding-batching), [slide](https://docs.google.com/presentation/d/1GTJJQ-dr10vBwBSEtHoTs73BFJtC12WgZF_cgkxLs9s/edit?usp=sharing) |
| 4 | 11 novembre | Containerizzazione e Custom Training Job su Vertex AI | [`004-training`](004-training), [slide](https://docs.google.com/presentation/d/17SkF8MjYK8BokkOtgOdStQdyRE7TN6WOuRMmQwtRXjE/edit?usp=sharing) |
| 5 | 19 novembre | Parameter server pattern e collective communication pattern | in preparazione |
| 6 | 26 novembre | Elasticity e fault-tolerance: Spot VM e checkpoint automatici | in preparazione |
| 7 | 10 dicembre | Sharded service pattern: serving di modelli di grandi dimensioni | in preparazione |
| 8 | 17 dicembre | Fan-in/fan-out, synchronous/asynchronous e step memoization | in preparazione |
| 9 | TBD | TBD | in preparazione |


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

### Eliminare le risorse a pagamento

Alla fine di ogni lezione, e comunque a fine corso, verificate che nel progetto non restino
risorse attive. Impostate prima le variabili (sostituire il segnaposto):

```bash
export PROJECT_ID="<ID_DEL_PROGETTO_GCP>"
export REGION="europe-west4"
gcloud config set project $PROJECT_ID
gcloud config get-value project     # controllo: deve stampare il proprio progetto
```

**1. Inventario.** Questi comandi elencano le risorse che possono generare costi
(un elenco vuoto significa che non c'è nulla da eliminare):

```bash
gcloud compute instances list                      # VM (anche con GPU), incluse le Spot VM
gcloud compute disks list --filter="-users:*"      # dischi non collegati ad alcuna VM
gcloud compute addresses list                      # indirizzi IP statici riservati
gcloud workbench instances list --location=${REGION}-a   # istanze Vertex AI Workbench
gcloud ai custom-jobs list --region=$REGION  --filter="state:(JOB_STATE_QUEUED JOB_STATE_PENDING JOB_STATE_RUNNING)"   # job ancora attivi
gcloud ai endpoints list --region=$REGION          # endpoint di serving (costano finché esistono)
gcloud run services list --region=$REGION          # servizi Cloud Run
gcloud artifacts repositories list --location=$REGION    # immagini Docker
gcloud storage buckets list --format="value(name)"  # bucket Cloud Storage
```

**2. Eliminazione.** Per ogni risorsa trovata, sostituire `NOME`, `ID` e `ZONA` con i valori
mostrati dall'inventario:

```bash
# VM (il disco di avvio viene eliminato con la VM)
gcloud compute instances delete NOME --zone=ZONA

# Dischi non collegati e IP statici
gcloud compute disks delete NOME --zone=ZONA
gcloud compute addresses delete NOME --region=$REGION

# Istanze Workbench
gcloud workbench instances delete NOME --location=ZONA

# Custom Training Job ancora in esecuzione (i job terminati non hanno costi)
gcloud ai custom-jobs cancel ID --region=$REGION

# Endpoint: prima si rimuovono i modelli deployati, poi l'endpoint
gcloud ai endpoints describe ID --region=$REGION --format="value(deployedModels.id)"
gcloud ai endpoints undeploy-model ID --region=$REGION --deployed-model-id=ID_MODELLO_DEPLOYATO
gcloud ai endpoints delete ID --region=$REGION

# Servizi Cloud Run
gcloud run services delete NOME --region=$REGION

# Repository di Artifact Registry (con tutte le immagini contenute)
gcloud artifacts repositories delete NOME --location=$REGION

# Bucket con tutto il contenuto (irreversibile)
gcloud storage rm -r gs://NOME
```

Rieseguite l'inventario del punto 1 per controllare che tutti gli elenchi siano vuoti.

**3. Soluzione drastica a fine corso.** Eliminare l'intero progetto arresta tutte le risorse
e tutti i costi in un colpo solo. Il progetto resta recuperabile per 30 giorni con
`gcloud projects undelete $PROJECT_ID`, poi viene cancellato definitivamente.

```bash
gcloud projects delete $PROJECT_ID
```

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
