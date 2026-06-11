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
