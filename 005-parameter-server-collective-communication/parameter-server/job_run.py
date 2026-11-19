# =============================================================================
# job_run.py — Lancio del Custom Training Job su Vertex AI
# =============================================================================
#
# Questo script crea e avvia un Custom Job su Vertex AI con topologia
# Parameter Server:
#
#   Pool 0 (1 replica)  →  Parameter Server
#   Pool 1 (2 repliche) →  Worker 0, Worker 1
#
# Vertex AI alloca 3 VM separate, scarica lo stesso container su ciascuna
# e inietta CLUSTER_SPEC per comunicare a ogni replica la topologia del
# cluster e il proprio ruolo.
#
# Prerequisiti:
#   pip install --upgrade google-cloud-aiplatform
#   gcloud auth application-default login
# =============================================================================

import os
import sys

from google.cloud import aiplatform

# =============================================================================
# Configurazione: tutti i valori arrivano dalle variabili del passo 0 del README
# =============================================================================
def env(name, default=None):
    value = os.environ.get(name, default)
    if not value:
        sys.exit(f"ERRORE: variabile d'ambiente {name} non impostata (vedi il passo 0 del README)")
    return value

PROJECT_ID  = env("PROJECT_ID")
REGION      = env("REGION", "europe-west4")
BUCKET      = env("BUCKET").replace("gs://", "").strip("/")
IMAGE       = env("IMAGE_PS")              # $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/lab5-ps:v1
N_WORKERS   = int(env("N_WORKERS", "2"))
N_EPOCHS    = env("N_EPOCHS", "5")
DISPLAY     = env("JOB_NAME", "lab5-parameter-server")

for k, v in [("PROJECT_ID", PROJECT_ID), ("REGION", REGION), ("BUCKET", BUCKET),
             ("IMAGE_PS", IMAGE), ("N_WORKERS", N_WORKERS), ("N_EPOCHS", N_EPOCHS)]:
    print(f"  {k:<10} = {v}")

# staging_bucket: bucket GCS usato da Vertex AI per file temporanei del job.
# DEVE essere nella stessa regione del job ($REGION): con un bucket in un'altra
# regione (es. "us") il job fallisce con FAILED_PRECONDITION prima di partire.
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

# Variabili passate a ogni container: train_ps.py le legge con os.environ.
# Il repository Artifact Registry ($REPO) deve esistere prima del docker push:
# Artifact Registry NON lo crea automaticamente al primo push.
CONTAINER_ENV = [
    {"name": "BUCKET",   "value": BUCKET},
    {"name": "N_EPOCHS", "value": N_EPOCHS},
]

# =============================================================================
# Definizione della topologia del job
# =============================================================================
# worker_pool_specs è una lista ordinata di pool di macchine:
#   - Pool 0 (indice 0): Vertex AI assegna TASK_TYPE="workerpool0" → PS
#   - Pool 1 (indice 1): Vertex AI assegna TASK_TYPE="workerpool1" → Worker
#
# Vertex AI inietta automaticamente CLUSTER_SPEC in ogni container con
# gli indirizzi hostname:porta di tutte le repliche del cluster.
job = aiplatform.CustomJob(
    display_name=DISPLAY,
    worker_pool_specs=[

        # ── Pool 0: Parameter Server ──────────────────────────────────────
        # Una sola replica: il PS è un singleton nel pattern classico.
        # n1-standard-4 = 4 vCPU, 15 GB RAM — sufficiente per gestire
        # i parametri di SimpleNet (~920 KB) senza acceleratori hardware.
        {
            "machine_spec": {
                "machine_type": "n1-standard-4"
            },
            "replica_count": 1,  # Sempre 1 per il PS nel pattern base
            "container_spec": {
                "image_uri": IMAGE,
                "env": CONTAINER_ENV,
                # Vertex AI inietterà automaticamente:
                #   CLUSTER_SPEC con tutti gli indirizzi del cluster
                #   TASK_TYPE="workerpool0", TASK_INDEX="" (vuoto)
            }
        },

        # ── Pool 1: Worker ────────────────────────────────────────────────
        # Due repliche: ciascuna riceve TASK_TYPE="workerpool1" e
        # TASK_INDEX=0 oppure TASK_INDEX=1 (anche se poi risultano vuoti
        # — il valore effettivo è in CLUSTER_SPEC["task"]["index"]).
        {
            "machine_spec": {
                "machine_type": "n1-standard-4"
            },
            "replica_count": N_WORKERS,  # $N_WORKERS, default 2
            "container_spec": {
                "image_uri": IMAGE,
                "env": CONTAINER_ENV,
            }
        }
    ]
)

# =============================================================================
# Lancio del job
# =============================================================================
# job.submit() invia il job a Vertex AI e restituisce il controllo
# immediatamente (non bloccante), a differenza di job.run(sync=True)
# che bloccherebbe il terminale fino al completamento.
# Dopo submit(), resource_name è disponibile non appena Vertex AI
# risponde con l'ID assegnato al job.
job.submit()

# Estrae il JOB_ID numerico dal resource_name completo.
# resource_name formato: projects/PROJECT/locations/REGION/customJobs/JOB_ID
job_id = job.resource_name.split("/")[-1]

print(f"Job creato con successo.")
print(f"Job name: {job.resource_name}")
print(f"JOB_ID:   {job_id}")
print(f"Stato:    {job.state}")
print()
print("Per monitorare i log:")
print(f"  gcloud ai custom-jobs stream-logs {job_id} --region={REGION}")
print()
print("Per controllare lo stato:")
print(f"  gcloud ai custom-jobs describe {job_id} --region={REGION} --format='value(state)'")