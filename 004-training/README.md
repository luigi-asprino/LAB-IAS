# Lezione 4 — Containerizzazione e Custom Training Job su Vertex AI

Esercitazione: addestrare un `RandomForestClassifier` sul dataset Iris eseguendo
`lab4.py` dentro un container Docker, lanciato come **Custom Training Job** su Vertex AI.

## File della cartella

| File | Contenuto |
|---|---|
| `lab4.py` | Script di training: legge `iris.csv` da GCS, addestra il modello, salva `model.joblib` |
| `Dockerfile` | Immagine del container (base: container Vertex AI per scikit-learn) |
| `job.yaml` | *Generato al passo 6* dalle variabili del passo 0: specifica del Custom Training Job |
| `key.json` | *Generato al passo 4*: chiave del service account per il test locale — **non va committata** |

`key.json`, `iris.csv`, `model.joblib` e `job.yaml` sono file generati durante l'esercitazione e devono
restare fuori dal repository: sono esclusi dal `.gitignore` di questa cartella.

## Prerequisiti

- Un progetto GCP con fatturazione attiva
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) installata e autenticata
- Docker installato e in esecuzione

## 0. Variabili

I propri valori si scrivono in **un solo punto**: questo blocco. Sostituire `<ID_DEL_PROGETTO_GCP>`
e `<NOME_DEL_BUCKET>` con i propri valori (tra le virgolette), poi eseguire nel terminale.
Tutti i comandi successivi, lo script e `job.yaml` prendono i valori da qui.
Se si apre un nuovo terminale, rieseguire questo blocco.

```bash
export PROJECT_ID="<ID_DEL_PROGETTO_GCP>"    # es. my-project-123456
export REGION="europe-west4"
export BUCKET="<NOME_DEL_BUCKET>"            # senza gs://, es. ias-mario-rossi-bucket
export REPO="lab4-repo"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/lab4-trainer:v1"

echo $PROJECT_ID $BUCKET $IMAGE     # controllo: nessun valore deve contenere < >

gcloud auth login
gcloud config set project $PROJECT_ID
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
```

### Parametri dello script

`lab4.py` non contiene valori scritti nel codice: legge tutto da variabili d'ambiente.

| Variabile | Obbligatoria | Default | Significato |
|---|---|---|---|
| `BUCKET` | sì | — | Bucket GCS con il dataset (senza `gs://`) |
| `GOOGLE_CLOUD_PROJECT` | in locale | — | Progetto GCP (su Vertex AI si usa `CLOUD_ML_PROJECT_ID`, impostata automaticamente) |
| `DATA_FILE` | no | `iris.csv` | Nome del dataset nel bucket |
| `LABEL_COLUMN` | no | `variety` | Colonna da predire |
| `N_ESTIMATORS` | no | `50` | Numero di alberi del RandomForest |
| `TEST_SIZE` | no | `0.2` | Frazione dei dati usata come test set |
| `SEED` | no | `42` | Seme per split e modello (risultati ripetibili) |
| `AIP_MODEL_DIR` | no | `gs://$BUCKET/models/` | Dove salvare il modello (impostata da Vertex AI, vedi passo 6) |

All'avvio lo script stampa la configurazione usata; se manca una variabile obbligatoria
si ferma con un messaggio `ERRORE: ...`.

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
  -e BUCKET=$BUCKET \
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

Generare `job.yaml` dalle variabili del passo 0 (immagine, bucket e directory di output
coincidono così per costruzione con quelli usati negli altri passi):

```bash
cat > job.yaml <<EOF
workerPoolSpecs:
- machineSpec:
    machineType: n1-standard-4
  replicaCount: 1
  containerSpec:
    imageUri: $IMAGE
    env:
    - name: BUCKET
      value: $BUCKET
baseOutputDirectory:
  outputUriPrefix: gs://$BUCKET/lab4
EOF
cat job.yaml
```

Altri parametri dello script (es. `N_ESTIMATORS`) si aggiungono come ulteriori voci sotto `env:`.
`baseOutputDirectory` fa sì che Vertex AI imposti `AIP_MODEL_DIR=gs://$BUCKET/lab4/model/`.

Lanciare il job:

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

Il modello si trova in `AIP_MODEL_DIR`, cioè `gs://$BUCKET/lab4/model/`:

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

# (Opzionale) Intero bucket: cancella il bucket e TUTTO il suo contenuto, dataset compreso.
# Operazione irreversibile: non eseguirla se il bucket serve ad altre lezioni.
gcloud storage rm -r gs://$BUCKET

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
| `Image not found` all'avvio del job | Variabili del passo 0 non impostate quando è stato generato `job.yaml` | Rieseguire il passo 0, poi rigenerare `job.yaml` (passo 6) |
| Comandi con valori vuoti o con `<...>` | Passo 0 non eseguito nel terminale corrente o segnaposto non sostituiti | Sostituire i segnaposto e rieseguire il passo 0 |
| `ERRORE: variabile d'ambiente BUCKET non impostata` | Manca `-e BUCKET=...` (locale) o la voce `env` in `job.yaml` | Aggiungere la variabile come nei passi 4 e 6 |
| `model.joblib` assente in `gs://$BUCKET/lab4/model/` | Job fallito o `baseOutputDirectory` mancante | Controllare lo stato e i log del job (passo 7) |
