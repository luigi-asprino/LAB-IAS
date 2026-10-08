# IAS Laboratorio – Lezione 1: Introduzione alla Google Cloud Platform

Comandi da eseguire, nell'ordine delle slide. Sostituite i valori tra `< >` con i vostri dati.

## 1. Installazione della gcloud CLI (Linux x86_64)

Su Mac usate il pacchetto `darwin`, su Windows l'installer `.exe`: <https://cloud.google.com/sdk/docs/install-sdk?hl=it>

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

Inizializzate la CLI (login, progetto e zona di default):

```bash
gcloud init
```

Verificate l'installazione:

```bash
gcloud --version
```

Login (già incluso in `gcloud init`):

```bash
gcloud auth login
```

Credenziali per le librerie Python (servono nelle prossime lezioni):

```bash
gcloud auth application-default login
```

## 2. Creazione e configurazione del progetto

L'ID del progetto deve essere unico a livello globale:

```bash
export PROJECT_ID="ias-lab-<cognome>"
gcloud projects create $PROJECT_ID
```

Collegate l'account di fatturazione (l'ID lo trovate con il primo comando):

```bash
gcloud billing accounts list
gcloud billing projects link $PROJECT_ID --billing-account=<XXXXXX-XXXXXX-XXXXXX>
```

Impostate il progetto attivo e verificate la configurazione:

```bash
gcloud config set project $PROJECT_ID
gcloud config list
```

Elencate i progetti a cui avete accesso:

```bash
gcloud projects list
```

## 3. Abilitazione dei servizi

Elencate i servizi disponibili (lista molto lunga):

```bash
gcloud services list --available
```

Abilitate tutti i servizi usati nel laboratorio:

```bash
gcloud services enable \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  notebooks.googleapis.com
```

Verificate i servizi abilitati:

```bash
gcloud services list --enabled
```

## 4. Creazione della VM

```bash
gcloud compute instances create ml-lab-cpu-vm \
  --zone=europe-west4-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --metadata="startup-script=sudo apt-get update &&
    sudo apt-get install -y python3-pip &&
    pip3 install scikit-learn pandas numpy"
```

Elencate le VM del progetto (stato, zona, IP esterno):

```bash
gcloud compute instances list
```

## 5. Training di un classificatore sulla VM

Connessione SSH (le chiavi vengono gestite automaticamente):

```bash
gcloud compute ssh ml-lab-cpu-vm --zone=europe-west4-a
```

> **Da qui in poi siete DENTRO la VM.** Attendete 1-2 minuti che lo startup script installi i pacchetti.

Per seguirne l'avanzamento:

```bash
sudo journalctl -u google-startup-scripts -f
```

Avviate l'interprete Python:

```bash
python3
```

Nell'interprete Python:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
```

Uscite dall'interprete e dalla VM:

```python
exit()
```

```bash
exit
```

## 6. Esperimento di scalabilità

Sul **vostro** computer, create il file `scaling.py`:

```bash
cat > scaling.py <<'EOF'
import os, time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
X, y = make_classification(n_samples=50_000, n_features=20, random_state=42)
times = {}
for n_jobs in [1, -1]:
    t0 = time.perf_counter()
    RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=42).fit(X, y)
    times[n_jobs] = time.perf_counter() - t0
    print(f'n_jobs={n_jobs}: {times[n_jobs]:.1f} s')
print(f'vCPU: {os.cpu_count()}  speedup: {times[1] / times[-1]:.2f}x')
EOF
```

Copiatelo sulla VM ed eseguitelo come job:

```bash
gcloud compute scp scaling.py ml-lab-cpu-vm:~ --zone=europe-west4-a
gcloud compute ssh ml-lab-cpu-vm --zone=europe-west4-a --command="python3 scaling.py"
```

(Facoltativo) Ripetete su una VM con 8 vCPU e confrontate gli speedup:

```bash
gcloud compute instances create ml-lab-cpu-vm-8 \
  --zone=europe-west4-a \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --metadata="startup-script=sudo apt-get update &&
    sudo apt-get install -y python3-pip &&
    pip3 install scikit-learn pandas numpy"
gcloud compute scp scaling.py ml-lab-cpu-vm-8:~ --zone=europe-west4-a
gcloud compute ssh ml-lab-cpu-vm-8 --zone=europe-west4-a --command="python3 scaling.py"
```

## 7. Errori frequenti

**`API [compute.googleapis.com] not enabled` / `billing is disabled`**

```bash
gcloud services enable compute.googleapis.com
gcloud billing projects describe $PROJECT_ID
```

**`ZONE_RESOURCE_POOL_EXHAUSTED`**

Ricreate la VM in un'altra zona, ad esempio sostituendo `--zone=europe-west4-a` con `--zone=europe-west4-b` o `--zone=europe-west1-b` nel comando di creazione.

**`ssh: ... port 22: Connection timed out`** (rete che blocca la porta 22)

```bash
gcloud compute ssh ml-lab-cpu-vm --zone=europe-west4-a --tunnel-through-iap
```

In alternativa usate Cloud Shell: <https://shell.cloud.google.com>

**`ModuleNotFoundError: sklearn`**: lo startup script non ha ancora finito.

```bash
sudo journalctl -u google-startup-scripts -f
```

## 8. Prima di uscire: eliminate le risorse

```bash
gcloud compute instances delete ml-lab-cpu-vm --zone=europe-west4-a
```

Se avete creato anche la VM da 8 vCPU:

```bash
gcloud compute instances delete ml-lab-cpu-vm-8 --zone=europe-west4-a
```

Verifica: l'elenco deve essere vuoto.

```bash
gcloud compute instances list
```

Controllate che non restino dischi orfani:

```bash
gcloud compute disks list
```

> **Non eliminate il progetto:** servirà nelle prossime lezioni.
