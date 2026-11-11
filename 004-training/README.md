# Lezione 4 — Containerizzazione e Custom Training Job su Vertex AI

Esercitazione: addestrare un `RandomForestClassifier` sul dataset Iris eseguendo
`lab4.py` dentro un container Docker, lanciato come **Custom Training Job** su Vertex AI.

## File della cartella

| File | Contenuto |
|---|---|
| `lab4.py` | Script di training: legge `iris.csv` da GCS, addestra il modello, salva `model.joblib` |
| `Dockerfile` | Immagine del container (base: container Vertex AI per scikit-learn) |
| `job.yaml` | Specifica del Custom Training Job (macchina, immagine, directory di output) |
| `key.json` | *Generato al passo 4*: chiave del service account per il test locale — **non va committata** |

`key.json`, `iris.csv` e `model.joblib` sono file generati durante l'esercitazione e devono
restare fuori dal repository: sono esclusi dal `.gitignore` di questa cartella.

## Prerequisiti

- Un progetto GCP con fatturazione attiva
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) installata e autenticata
- Docker installato e in esecuzione

## 0. Variabili

Adattare i valori al proprio progetto e al proprio bucket, poi eseguire nel terminale
(tutti i comandi successivi le usano):

```bash
export PROJECT_ID=test-project-493915
export REGION=europe-west4
export BUCKET=ias-luigi-asprino-bucket
export REPO=lab4-repo
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/lab4-trainer:v1

gcloud auth login
gcloud config set project $PROJECT_ID
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
```

Se si cambiano progetto o bucket, aggiornare anche:
- `BUCKET` in `lab4.py`;
- `imageUri` e `outputUriPrefix` in `job.yaml`.

## 1. Abilitare le API

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  --project=$PROJECT_ID
```

## 2. Preparare i dati su Cloud Storage

Creare il bucket (solo se non esiste già):

```bash
gcloud storage buckets create gs://$BUCKET --location=$REGION
```

Scaricare il dataset Iris (colonne `sepal.length`, `sepal.width`, `petal.length`,
`petal.width`, `variety`) e caricarlo nel bucket:

```bash
curl -L -o iris.csv https://gist.githubusercontent.com/netj/8836201/raw/iris.csv
gcloud storage cp iris.csv gs://$BUCKET/
gcloud storage ls gs://$BUCKET/
```

## 3. Build dell'immagine

```bash
docker build --platform linux/amd64 -t lab4-trainer:latest .
```

`--platform linux/amd64` serve sui Mac con chip Apple: Vertex AI esegue solo immagini amd64.

## 4. Test locale del container (opzionale ma consigliato)

Creare una chiave per il service account (una sola volta):

```bash
gcloud iam service-accounts keys create key.json \
  --iam-account=$PROJECT_NUMBER-compute@developer.gserviceaccount.com
```

Eseguire il container passando credenziali e variabili d'ambiente:

```bash
docker run --rm \
  -e AIP_MODEL_DIR=/modello-lab4 \
  -e GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/key.json \
  -v "$PWD/key.json":/secrets/key.json \
  lab4-trainer:latest
```

L'output deve mostrare le prime righe del dataset, l'accuracy e il percorso del modello salvato.
Con `AIP_MODEL_DIR` locale il modello resta nel container; omettendo la variabile viene
caricato su `gs://$BUCKET/models/model.joblib`.

> In alternativa alla chiave JSON: `gcloud auth application-default login` e, al posto delle
> ultime due opzioni, `-v ~/.config/gcloud:/root/.config/gcloud`.

## 5. Artifact Registry: creare il repository e caricare l'immagine

```bash
# Repository Docker nella stessa regione del job
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="IAS Lab 4" \
  --project=$PROJECT_ID

# Autenticare Docker verso il registry (una volta per macchina)
gcloud auth configure-docker $REGION-docker.pkg.dev

# Tag e push
docker tag lab4-trainer:latest $IMAGE
docker push $IMAGE

# Verifica
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/$REPO
```

## 6. Lanciare il Custom Training Job

```bash
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=lab4-training-job \
  --config=job.yaml
```

Il comando stampa il nome del job (`projects/.../locations/.../customJobs/JOB_ID`).
Salvare l'ID in una variabile:

```bash
export JOB_ID=$(gcloud ai custom-jobs list --region=$REGION \
  --filter="displayName=lab4-training-job" \
  --sort-by=~createTime --limit=1 --format="value(name)")
echo $JOB_ID
```

## 7. Monitoraggio

```bash
# Stato del job (JOB_STATE_PENDING → RUNNING → SUCCEEDED / FAILED)
gcloud ai custom-jobs describe $JOB_ID --region=$REGION --format="value(state)"

# Log del container in tempo reale
gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION
```

Il job è visibile anche nella console: **Vertex AI → Training → Custom jobs**.

## 8. Verifica del modello

`job.yaml` imposta `baseOutputDirectory` a `gs://$BUCKET/lab4`, quindi Vertex AI passa allo
script `AIP_MODEL_DIR=gs://$BUCKET/lab4/model/`:

```bash
gcloud storage ls gs://$BUCKET/lab4/model/
```

Deve comparire `model.joblib`. Per scaricarlo:

```bash
gcloud storage cp gs://$BUCKET/lab4/model/model.joblib .
```

## 9. Pulizia (per evitare costi)

```bash
# Immagine e repository
gcloud artifacts docker images delete $IMAGE --quiet
gcloud artifacts repositories delete $REPO --location=$REGION --quiet

# Output del training
gcloud storage rm -r gs://$BUCKET/lab4/

# Chiave locale del service account: revocarla (l'ID si legge da key.json) e cancellare il file
export KEY_ID=$(python3 -c "import json; print(json.load(open('key.json'))['private_key_id'])")
gcloud iam service-accounts keys delete $KEY_ID \
  --iam-account=$PROJECT_NUMBER-compute@developer.gserviceaccount.com --quiet \
  && rm key.json
```

Se `key.json` è già stato cancellato, elencare le chiavi create dall'utente e cancellare
quella con la data del passo 4:

```bash
gcloud iam service-accounts keys list --managed-by=user \
  --iam-account=$PROJECT_NUMBER-compute@developer.gserviceaccount.com
```

La VM del training viene deallocata automaticamente da Vertex AI al termine del job.

## Problemi frequenti

| Sintomo | Causa probabile | Soluzione |
|---|---|---|
| `exec format error` nei log del job | Immagine costruita per arm64 (Mac Apple Silicon) | Ricostruire con `--platform linux/amd64` e rifare il push |
| `403` / `Permission denied` su GCS nel test locale | Chiave mancante o service account senza accesso al bucket | Verificare il montaggio di `key.json` e i ruoli del service account |
| `denied: Permission ... artifactregistry` al push | Docker non autenticato verso il registry | `gcloud auth configure-docker $REGION-docker.pkg.dev` |
| `Image not found` all'avvio del job | `imageUri` in `job.yaml` diverso da `$IMAGE` | Allineare `job.yaml` al tag effettivamente caricato |
| `model.joblib` assente in `gs://$BUCKET/lab4/model/` | Job fallito o `baseOutputDirectory` mancante | Controllare lo stato e i log del job (passo 7) |
