# IAS Laboratorio – Lezione 3: Google Cloud Storage, sharding e batching

**Slide della lezione:** [003-Google Cloud Storage (Google Slides)](https://docs.google.com/presentation/d/1GTJJQ-dr10vBwBSEtHoTs73BFJtC12WgZF_cgkxLs9s/edit?usp=sharing)

Comandi da eseguire, nell'ordine delle slide. Sostituite i valori tra `< >` con i vostri dati.

```bash
export PROJECT_ID="ias-lab-<cognome>"
gcloud config set project $PROJECT_ID
```

## 1. Creare il bucket

Il nome del bucket è univoco in tutto GCS: usate per esempio `ias-<nome>-<cognome>-bucket`.
Create il bucket nella stessa regione della VM e dei job Vertex AI (`europe-west4`):
latenza minore e nessun costo di trasferimento tra regioni.

```bash
gcloud storage buckets create gs://<NOME_UNIVOCO> --location=europe-west4
```

## 2. Caricare il dataset

```bash
gcloud storage cp ~/Desktop/iris.csv gs://<NOME_UNIVOCO>/
```

## 3. Verificare il contenuto

```bash
gcloud storage ls --long gs://<NOME_UNIVOCO>
```

`--long` mostra dimensione e data di ogni oggetto; `--recursive` scende nei prefissi (per esempio `shards/`).

Spazio occupato dal bucket:

```bash
gcloud storage du -s gs://<NOME_UNIVOCO>
```

## 4. Notebook: dataset PyTorch su GCS con sharding

Aprite [`sharding-batching-with-gcp.ipynb`](sharding-batching-with-gcp.ipynb) e impostate
`NOME_DEL_BUCKET` con il nome del vostro bucket. Il notebook:

1. legge `iris.csv` dal bucket con `gcsfs`;
2. converte i dati in tensori e **li mescola** con un seed fisso: `iris.csv` è ordinato per classe,
   e senza mescolamento ogni shard conterrebbe una o due classi soltanto;
3. divide il dataset in 4 shard `shards/shard-000.pt … shard-003.pt` e li carica sul bucket;
4. legge gli shard on-demand con la classe `GCSShardDataset`
   ([`gcpbucketdataset.py`](gcpbucketdataset.py)) e un `DataLoader` con `num_workers=2`.

La classe del dataset è in un file separato per evitare errori di pickling con `num_workers > 0`:
tenete `gcpbucketdataset.py` nella stessa cartella del notebook.

Dipendenze: `pip install gcsfs torch pandas`. Le credenziali si ottengono con
`gcloud auth application-default login` (Step 1 del notebook).

## 5. Costi

- Conservazione (Standard, `europe-west4`): circa 0,020 $ per GB al mese. Il dataset Iris costa meno di un centesimo.
- Operazioni: scritture e list 0,005 $ ogni 1000 richieste, letture 0,0004 $ ogni 1000.
  Pochi file grandi (shard) costano meno di molti file piccoli.
- Prezzi aggiornati: [cloud.google.com/storage/pricing](https://cloud.google.com/storage/pricing).

## 6. Eliminate il bucket

Il bucket serve nelle prossime lezioni, ma per evitare costi, potete eliminare il contenuto per poi ricrearlo nelle prossime lezioni.

```bash
gcloud storage rm --recursive gs://<NOME_UNIVOCO>
```
