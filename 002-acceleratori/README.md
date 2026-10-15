# IAS Laboratorio – Lezione 2: Acceleratori su Google Cloud (GPU e TPU)

**Slide della lezione:** [002-Acceleratori (Google Slides)](https://docs.google.com/presentation/d/1lgPl4FqEK8QbgR_6Km86P-irWEHhK2PU5EzdPpDFzQM/edit?usp=sharing)

Comandi da eseguire, nell'ordine delle slide. Sostituite i valori tra `< >` con i vostri dati.

```bash
export PROJECT_ID="ias-lab-<cognome>"
gcloud config set project $PROJECT_ID
```

## 1. Quali acceleratori sono disponibili

Zone in cui è offerta la T4:

```bash
gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4"
```

Tutte le GPU disponibili in una zona:

```bash
gcloud compute accelerator-types list --filter="zone:europe-west4-a"
```

## 2. IAM: verificate i vostri ruoli

Ruoli assegnati al vostro account sul progetto:

```bash
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:$(gcloud config get-value account)" \
  --format="value(bindings.role)"
```

Servono `roles/owner` oppure `roles/compute.instanceAdmin.v1`, più `roles/servicemanagement.quotaAdmin`
per chiedere un aumento di quota (`quotaViewer` permette solo di leggerle).

## 3. Quote GPU

Servono **due** quote maggiori di zero: quella globale e quella della regione in cui create la VM.

Quota globale `GPUS_ALL_REGIONS`:

```bash
gcloud compute project-info describe \
  --project $PROJECT_ID \
  --format="yaml(quotas)"
```

Nell'output cercate:

```
- limit: 0.0
  metric: GPUS_ALL_REGIONS
  usage: 0.0
```

`limit: 0.0` significa che non potete creare VM con GPU. Su un account Free Trial serve prima
l'upgrade a un account fatturato (i crediti residui restano validi).

Quota regionale della T4 (`NVIDIA_T4_GPUS`, e `PREEMPTIBLE_NVIDIA_T4_GPUS` per le VM Spot):

```bash
gcloud compute regions describe europe-west4 \
  --flatten="quotas[]" --filter="quotas.metric:T4" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)"
```

Per chiedere un aumento: Console GCP → **IAM & Admin → Quotas & System Limits** → filtrate per la
metrica (`gpus_all_regions` o `nvidia_t4_gpus`) → selezionate la voce → **Edit** → nuovo valore `1`
→ **Submit request**. Per 1 GPU la conferma arriva spesso in pochi minuti via email; dopo
l'approvazione possono servire fino a 15 minuti perché la quota sia visibile.

## 4. Creazione della VM con GPU

```bash
gcloud compute instances create ml-lab-vm \
  --zone=europe-west4-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

- La T4 si collega solo a VM della serie **N1**, con il flag `--accelerator`.
- `--maintenance-policy=TERMINATE` è obbligatorio: le VM con GPU non supportano la live migration.
- L'immagine Deep Learning VM include PyTorch 2.9, CUDA 12.9 e il driver NVIDIA 580.

Costo indicativo: circa $0.55/ora (VM $0.19 + GPU $0.35), più il disco.

### Se la zona non ha T4 libere (`ZONE_RESOURCE_POOL_EXHAUSTED`)

Riprovate in un'altra zona, per esempio `europe-west1-b`:

```bash
gcloud compute instances create ml-lab-vm \
  --zone=europe-west1-b \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

Se cambiate regione, controllate anche la quota regionale (sezione 3).

Alternativa con L4: il machine type `g2-standard-4` include già la GPU, quindi `--accelerator` non serve.

```bash
gcloud compute instances create ml-lab-vm \
  --zone=europe-west4-a \
  --machine-type=g2-standard-4 \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

Alternativa Spot (più economica, ma Google può interrompere la VM; ne parliamo nella lezione 7):
aggiungete `--provisioning-model=SPOT` al comando di creazione.

## 5. Connessione alla VM

Da terminale:

```bash
gcloud compute ssh ml-lab-vm --zone=<ZONA>
```

Per VS Code Remote-SSH, generate la configurazione SSH e poi aprite la VM da **Remote Explorer → SSH Targets**:

```bash
gcloud compute config-ssh
```

## 6. Verifica della GPU

**Sulla VM**:

```bash
nvidia-smi
```

Il campo `CUDA Version` indica la versione massima supportata dal driver, non il toolkit installato.

Lo script di verifica con PyTorch e l'esperimento CPU vs GPU sono nelle slide della sezione 05.

## 7. Errori frequenti

**`Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.`**: richiedete la quota globale (sezione 3).

**`Quota 'NVIDIA_T4_GPUS' exceeded. Limit: 0.0 in region ...`**: richiedete la quota regionale o cambiate regione.

**`Instances with guest accelerators do not support live migration.`**: aggiungete `--maintenance-policy=TERMINATE`.

**`ZONE_RESOURCE_POOL_EXHAUSTED`**: la zona non ha GPU libere, cambiate zona (sezione 4).

**`ssh: ... port 22: Connection timed out`**: usate `--tunnel-through-iap` oppure Cloud Shell.

## 8. Prima di uscire: eliminate le risorse

Una VM fermata (`stop`) non paga vCPU e GPU, ma il disco da 200 GB costa circa $8 (pd-standard)
o $20 (pd-balanced) al mese. Se non vi serve più, eliminatela:

```bash
gcloud compute instances delete ml-lab-vm --zone=<ZONA>
```

Verifica: entrambi gli elenchi devono essere vuoti.

```bash
gcloud compute instances list
gcloud compute disks list
```
