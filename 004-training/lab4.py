#!/usr/bin/env python
"""Lezione 4 - Training di un RandomForestClassifier su iris.csv
eseguito come Custom Training Job su Vertex AI.

Tutti i parametri si leggono da variabili d'ambiente:

  BUCKET          (obbligatoria) bucket GCS con il dataset, senza gs://
  DATA_FILE       oggetto del dataset nel bucket          (default: iris.csv)
  LABEL_COLUMN    colonna da predire                      (default: variety)
  N_ESTIMATORS    alberi del RandomForest                 (default: 50)
  TEST_SIZE       frazione del test set                   (default: 0.2)
  SEED            seme per split e modello                (default: 42)
  AIP_MODEL_DIR   destinazione del modello: gs://... o cartella locale
                  (impostata da Vertex AI se il job ha baseOutputDirectory;
                   default: gs://BUCKET/models/)
  Progetto GCP:   CLOUD_ML_PROJECT_ID (impostata da Vertex AI)
                  oppure GOOGLE_CLOUD_PROJECT (test locale)
"""
import io
import os
import sys

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def env(name, default=None, cast=str):
    value = os.environ.get(name, default)
    if value is None or value == "":
        sys.exit("ERRORE: variabile d'ambiente %s non impostata" % name)
    try:
        return cast(value)
    except ValueError:
        sys.exit("ERRORE: valore non valido per %s: %r" % (name, value))


PROJECT_ID = os.environ.get("CLOUD_ML_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    sys.exit("ERRORE: impostare GOOGLE_CLOUD_PROJECT (su Vertex AI CLOUD_ML_PROJECT_ID e' automatica)")
BUCKET = env("BUCKET").replace("gs://", "").strip("/")
DATA_FILE = env("DATA_FILE", "iris.csv")
LABEL_COLUMN = env("LABEL_COLUMN", "variety")
N_ESTIMATORS = env("N_ESTIMATORS", "50", int)
TEST_SIZE = env("TEST_SIZE", "0.2", float)
SEED = env("SEED", "42", int)
MODEL_DIR = env("AIP_MODEL_DIR", "gs://%s/models/" % BUCKET)

print("Configurazione:")
for k, v in [("PROJECT_ID", PROJECT_ID), ("BUCKET", BUCKET), ("DATA_FILE", DATA_FILE),
             ("LABEL_COLUMN", LABEL_COLUMN), ("N_ESTIMATORS", N_ESTIMATORS),
             ("TEST_SIZE", TEST_SIZE), ("SEED", SEED), ("MODEL_DIR", MODEL_DIR)]:
    print("  %-13s %s" % (k, v))

# Caricamento dati da GCS
client = storage.Client(project=PROJECT_ID)
blob = client.bucket(BUCKET).blob(DATA_FILE)
df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
print(df.head())

# Addestramento del modello (seme fissato: esecuzioni ripetibili)
X = df.drop(LABEL_COLUMN, axis=1)
y = df[LABEL_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED)
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)
clf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

# Salvataggio del modello su GCS (gs://...) o in una cartella locale
local_path = "/tmp/model.joblib"
joblib.dump(clf, local_path)

if MODEL_DIR.startswith("gs://"):
    bucket_name, _, prefix = MODEL_DIR[len("gs://"):].partition("/")
    dest = prefix.rstrip("/") + "/model.joblib" if prefix else "model.joblib"
    client.bucket(bucket_name).blob(dest).upload_from_filename(local_path)
    print("Modello salvato su gs://%s/%s" % (bucket_name, dest))
else:  # test locale con una directory del container
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "model.joblib"))
    print("Modello salvato in", os.path.join(MODEL_DIR, "model.joblib"))
