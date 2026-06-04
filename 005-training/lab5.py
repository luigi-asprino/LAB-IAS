#!/usr/bin/env python
import subprocess, sys
from google.cloud import storage
import pandas as pd
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
print(PROJECT_ID)

BUCKET     = f"ias-luigi-asprino-bucket"   # adattare al nome usato
client = storage.Client()
blob   = client.bucket(BUCKET).blob("iris.csv")
df     = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
print(df.head())

# Addestramento del modello
X = df.drop("variety", axis=1)
y = df["variety"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

# Salvataggio del modello su GCS
MODEL_DIR  = os.environ.get("AIP_MODEL_DIR", "/tmp/model")  # standard Vertex AI
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, f"{MODEL_DIR}/model.joblib", protocol=5)
client.bucket(BUCKET).blob("models/model.joblib").upload_from_filename(f"{MODEL_DIR}/model.joblib")
print(f"Modello salvato su GCS come {BUCKET}/models/model.joblib")

