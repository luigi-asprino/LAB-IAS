#!/usr/bin/env python
"""Lezione 4 - Training di un RandomForestClassifier su iris.csv
eseguito come Custom Training Job su Vertex AI."""
import io
import os

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Su Vertex AI CLOUD_ML_PROJECT_ID e' impostata automaticamente;
# in locale si passa GOOGLE_CLOUD_PROJECT con `docker run -e`.
PROJECT_ID = os.environ.get("CLOUD_ML_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
BUCKET = "ias-luigi-asprino-bucket"  # adattare al nome usato
print("Progetto:", PROJECT_ID)

# Caricamento dati da GCS
client = storage.Client(project=PROJECT_ID)
blob = client.bucket(BUCKET).blob("iris.csv")
df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
print(df.head())

# Addestramento del modello (random_state fissato: esecuzioni ripetibili)
X = df.drop("variety", axis=1)
y = df["variety"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

# Salvataggio del modello.
# Vertex AI imposta AIP_MODEL_DIR (= <baseOutputDirectory>/model/) solo se
# il job specifica baseOutputDirectory; altrimenti si usa gs://BUCKET/models/.
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "gs://%s/models/" % BUCKET)
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
