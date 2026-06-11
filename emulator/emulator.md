# Guida Pratica: Configurazione Ambiente GCP Locale per IA Scalabile

Questa guida descrive i passi necessari per configurare un ambiente di emulazione locale completo per **Google Cloud Platform (GCP)** sul proprio laptop. L'obiettivo è sviluppare, testare e validare pipeline di dati e modelli di Machine Learning in modo completamente disaccoppiato e a costo zero, simulando l'interazione con **Cloud Storage (GCS)** e **Cloud Pub/Sub**.

---

## 1. Prerequisiti di Sistema

Prima di iniziare, assicurati di avere installato sul tuo sistema:
* **Docker / Docker Desktop** (fondamentale per containerizzare i servizi non nativi)
* **Python 3.8+** (con il gestore di pacchetti `pip`)

---

## 2. Installazione della Google Cloud CLI e degli Emulatori

La Google Cloud CLI (Command Line Interface) contiene gli strumenti fondamentali per interagire con GCP ed include componenti nativi per l'emulazione.

### 2.1. Installazione di gcloud CLI
Seguire le istruzioni ufficiali in base al proprio sistema operativo:
* **macOS (Homebrew):** `brew install --cask google-cloud-sdk`
* **Linux (Debian/Ubuntu):**

```bash
  sudo apt-get update && sudo apt-get install google-cloud-cli
```

### 2.2. Installazione dell'emulatore Pub/Sub

Una volta completata l'installazione della CLI, aprire il terminale e installare il componente per l'emulazione di Pub/Sub:

```bash
gcloud components install pubsub-emulator
```

*Nota: Se l'installazione richiede i privilegi di amministratore, eseguire il comando con `sudo` (su Linux/macOS).*

---

## 3. Avvio degli Ambienti di Simulazione

Per rendere persistente la simulazione, è consigliabile aprire **due finestre di terminale separate** (o due tab) dedicate all'esecuzione dei servizi in background.

### Terminale 1: Avvio dell'Emulatore Cloud Pub/Sub

Eseguire il comando specificando l'host e la porta locale (standard: `8085`):

```bash
gcloud beta emulators pubsub start --host-port=localhost:8085

```

Mantenere questo terminale aperto. L'emulatore accetterà connessioni gRPC ed HTTP trasparenti.

### Terminale 2: Avvio di Fake-GCS-Server (Cloud Storage)

Poiché la CLI non include un emulatore locale integrato per lo Storage di oggetti, utilizziamo l'immagine Docker standard della community `fake-gcs-server`. Questo container replica fedelmente le API REST di Google Cloud Storage.

Eseguire il container sulla porta `4443` disabilitando temporaneamente il protocollo HTTPS (usando lo schema HTTP semplice per i test locali):

```bash
docker run -d --name fake-gcs-server -p 4443:4443 fsouza/fake-gcs-server -scheme http
```

---

## 4. Configurazione delle Librerie Client (Python)

Le librerie client ufficiali di Google per Python sono progettate per intercettare specifiche variabili d'ambiente. Se presenti, la libreria devia le richieste API dall'endpoint di produzione Cloud verso l'indirizzo locale specificato, saltando i controlli di autenticazione OAuth2.

Creare un ambiente virtuale ed installare i pacchetti necessari:

```bash
# Creazione e attivazione dell'ambiente virtuale
python3 -m venv gcp_env
source gcp_env/bin/activate  # Su Windows usa: gcp_env\\Scripts\\activate

# Installazione delle librerie client ufficiali
pip install google-cloud-storage google-cloud-pubsub

```

---

## 5. Script di Prova Integrato (`pipeline_test.py`)

Crea un file chiamato `pipeline_test.py` sul tuo editor di testo e incolla il seguente script. Questo codice configura le variabili d'ambiente dinamicamente tramite Python, crea un bucket, effettua l'upload di un dataset sintetico (struttura MNIST) e pubblica una notifica di evento su Pub/Sub.

```python
import os
import sys

# 1. Configurazione Rigida delle Variabili d'Ambiente per l'Emulazione
os.environ["GOOGLE_CLOUD_PROJECT"] = "scalable-ai-local-lab"
os.environ["PUBSUB_EMULATOR_HOST"] = "localhost:8085"
os.environ["STORAGE_EMULATOR_HOST"] = "http://localhost:4443"

# Disabilita i controlli SSL stretti per le connessioni HTTP locali dell'emulatore storage
os.environ["STORAGE_EMULATOR_HOST_W_SCHEME"] = "http://localhost:4443"

from google.cloud import storage
from google.cloud import pubsub_v1

def run_local_pipeline_test():
    print("=== AVVIO TEST AMBIENTE SIMULATO GCP ===")
    print(f"Project ID impostato: {os.environ['GOOGLE_CLOUD_PROJECT']}")
    print(f"Endpoint GCS: {os.environ['STORAGE_EMULATOR_HOST']}")
    print(f"Endpoint Pub/Sub: {os.environ['PUBSUB_EMULATOR_HOST']}\\n")

    # -------------------------------------------------------------------------
    # PARTE 1: Simulazione Google Cloud Storage
    # -------------------------------------------------------------------------
    print("[GCS] Connessione al client di archiviazione locale...")
    storage_client = storage.Client()
    
    bucket_name = "mnist-local-dataset"
    blob_name = "raw_features.csv"
    
    # Creazione del Bucket
    try:
        bucket = storage_client.create_bucket(bucket_name)
        print(f"[GCS] Bucket '{bucket_name}' creato con successo.")
    except Exception as e:
        # Se il bucket esiste già localmente, lo recuperiamo
        bucket = storage_client.get_bucket(bucket_name)
        print(f"[GCS] Bucket '{bucket_name}' già esistente. Utilizzo corrente.")

    # Upload di una stringa che simula un dataset CSV di test
    blob = bucket.blob(blob_name)
    mock_csv_data = "label,pixel0,pixel1,pixel2\\n7,0,255,128\\n2,0,0,64"
    blob.upload_from_string(mock_csv_data, content_type="text/csv")
    print(f"[GCS] Caricato file '{blob_name}' nel bucket '{bucket_name}'.")

    # -------------------------------------------------------------------------
    # PARTE 2: Simulazione Google Cloud Pub/Sub
    # -------------------------------------------------------------------------
    print("\\n[PUB/SUB] Connessione al Publisher Client locale...")
    publisher_client = pubsub_v1.PublisherClient()
    
    topic_id = "ml-pipeline-trigger"
    topic_path = publisher_client.topic_path(os.environ["GOOGLE_CLOUD_PROJECT"], topic_id)
    
    # Creazione del Topic
    try:
        publisher_client.create_topic(request={"name": topic_path})
        print(f"[PUB/SUB] Topic '{topic_id}' creato con successo.")
    except Exception:
        print(f"[PUB/SUB] Topic '{topic_id}' già esistente.")

    # Pubblicazione di un evento di notifica
    message_payload = b"EVENT: NEW_DATASET_AVAILABLE | BUCKET: mnist-local-dataset"
    print(f"[PUB/SUB] Pubblicazione messaggio sul topic '{topic_id}'...")
    
    future = publisher_client.publish(topic_path, message_payload, file_origin="local_script")
    message_id = future.result()
    
    print(f"[PUB/SUB] Messaggio pubblicato correttamente. Message ID generato: {message_id}")
    print("\\n=== TEST COMPLETATO CON SUCCESSO ===")
    print("L'ambiente ha risposto correttamente senza alcuna chiamata verso i server remoti di Google.")

if __name__ == "__main__":
    try:
        run_local_pipeline_test()
    except Exception as error:
        print(f"\\n[ERRORE] Il test è fallito: {error}", file=sys.stderr)
        print("Verifica che gli emulatori nei Terminali 1 e 2 siano attivi e configurati sulle porte corrette.", file=sys.stderr)

```

### Esecuzione della Prova

Per eseguire lo script di test, lancia il comando dal terminale dove hai attivato l'ambiente virtuale:

```bash
python pipeline_test.py

```

### Output Atteso nel Terminale

Se la configurazione è corretta, sul terminale apparirà la seguente sequenza di log confermati:

```text
=== AVVIO TEST AMBIENTE SIMULATO GCP ===
Project ID impostato: scalable-ai-local-lab
Endpoint GCS: http://localhost:4443
Endpoint Pub/Sub: localhost:8085

[GCS] Connessione al client di archiviazione locale...
[GCS] Bucket 'mnist-local-dataset' creato con successo.
[GCS] Caricato file 'raw_features.csv' nel bucket 'mnist-local-dataset'.

[PUB/SUB] Connessione al Publisher Client locale...
[PUB/SUB] Topic 'ml-pipeline-trigger' creato con successo.
[PUB/SUB] Pubblicazione messaggio sul topic 'ml-pipeline-trigger'...
[PUB/SUB] Messaggio pubblicato correttamente. Message ID generato: 1

=== TEST COMPLETATO CON SUCCESSO ===
L'ambiente ha risposto correttamente senza alcuna chiamata verso i server remoti di Google.

```

---

## 6. Pulizia e Reset dell'Ambiente Locale

Al termine della sessione di laboratorio o di sviluppo, è possibile interrompere gli emulatori per liberare le porte di sistema:

1. **Interrompere Pub/Sub:** Posizionarsi sul *Terminale 1* e premere `Ctrl + C`.
2. **Interrompere e rimuovere il container dello Storage:**
```bash
docker stop fake-gcs-server
docker rm fake-gcs-server

```

