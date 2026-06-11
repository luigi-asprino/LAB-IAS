FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .

# --no-cache-dir riduce la dimensione dell'immagine non salvando i pacchetti
# scaricati da pip nella cache locale del container
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Variabili iniettate a runtime da Vertex AI — qui solo come placeholder
# documentativo; i valori reali arrivano dall'orchestratore al lancio del job
ENV CLUSTER_SPEC=""
ENV TASK_TYPE=""
ENV TASK_INDEX=""
ENV GCS_BUCKET="ias-luigi-asprino-bucket"

CMD ["python", "train_ps.py"]