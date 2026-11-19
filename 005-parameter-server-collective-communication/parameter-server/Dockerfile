FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .

# --no-cache-dir riduce la dimensione dell'immagine non salvando i pacchetti
# scaricati da pip nella cache locale del container
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Nessun valore di progetto o bucket nell'immagine: CLUSTER_SPEC è iniettata
# da Vertex AI, BUCKET (e gli altri parametri) arrivano da job_run.py.

CMD ["python", "train_ps.py"]