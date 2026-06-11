# =============================================================================
# upload_mnist.py — Caricamento del dataset MNIST su Google Cloud Storage
# =============================================================================
#
# Script da eseguire UNA VOLTA SOLA prima di lanciare il training job.
# Scarica il dataset MNIST dai server ufficiali di Google (mirror affidabile),
# lo converte in formato numpy (.npy) e lo carica su GCS.
#
# Struttura creata su GCS:
#   gs://BUCKET/datasets/mnist/train/X.npy  (60000, 784) float32
#   gs://BUCKET/datasets/mnist/train/y.npy  (60000,)     int64
#   gs://BUCKET/datasets/mnist/test/X.npy   (10000, 784) float32
#   gs://BUCKET/datasets/mnist/test/y.npy   (10000,)     int64
#
# Utilizzo:
#   python upload_mnist.py
#
# Prerequisiti:
#   pip install numpy google-cloud-storage
#   gcloud auth application-default login
# =============================================================================

import io
import struct
import gzip
import urllib.request
import numpy as np
from google.cloud import storage

# Nome del bucket GCS di destinazione
BUCKET = "ias-luigi-asprino-bucket"

# Prefisso del path all'interno del bucket
PREFIX = "datasets/mnist"

# URL del mirror Google per i file binari MNIST in formato IDX gzippato.
# NOTA: il sito originale yann.lecun.com restituisce 404 — usare questo mirror.
URLS = {
    "train": {
        "images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    },
    "test": {
        "images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    }
}


def download_and_parse_images(url: str) -> np.ndarray:
    """
    Scarica e decomprime un file di immagini MNIST in formato IDX3.

    Formato IDX3 (immagini):
      Byte 0-3:  magic number (0x00000803)
      Byte 4-7:  numero di immagini (N)
      Byte 8-11: numero di righe (28)
      Byte 12-15: numero di colonne (28)
      Byte 16+:  pixel come uint8, N*28*28 valori

    Restituisce array shape (N, 784) normalizzato in [0, 1].
    La normalizzazione replica il comportamento di transforms.ToTensor()
    di torchvision, garantendo compatibilità con il modello.
    """
    print(f"  Download {url} ...")
    with urllib.request.urlopen(url) as r:
        data = gzip.decompress(r.read())

    # Legge l'header: 4 interi big-endian da 4 byte ciascuno
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])

    # Legge i pixel saltando i 16 byte di header
    pixels = np.frombuffer(data[16:], dtype=np.uint8)

    # Reshape: (N*28*28,) → (N, 784) e normalizzazione uint8 [0,255] → float32 [0,1]
    return pixels.reshape(n, rows * cols).astype(np.float32) / 255.0


def download_and_parse_labels(url: str) -> np.ndarray:
    """
    Scarica e decomprime un file di label MNIST in formato IDX1.

    Formato IDX1 (label):
      Byte 0-3: magic number (0x00000801)
      Byte 4-7: numero di label (N)
      Byte 8+:  label come uint8, valori 0-9

    Restituisce array shape (N,) come int64.
    int64 è richiesto da nn.CrossEntropyLoss in PyTorch.
    """
    print(f"  Download {url} ...")
    with urllib.request.urlopen(url) as r:
        data = gzip.decompress(r.read())

    # Legge l'header: 2 interi big-endian
    magic, n = struct.unpack(">II", data[:8])

    # Converte uint8 in int64 per compatibilità con CrossEntropyLoss
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


def upload_npy(bucket_obj, arr: np.ndarray, blob_path: str):
    """
    Serializza un array numpy in formato .npy e lo carica su GCS.

    Usa un buffer in memoria (BytesIO) invece di un file temporaneo
    su disco: più veloce e non lascia file residui nel filesystem locale.
    """
    buf = io.BytesIO()
    np.save(buf, arr)   # Serializza in formato .npy (header + dati binari)
    buf.seek(0)          # Riporta il cursore all'inizio per la lettura

    blob = bucket_obj.blob(blob_path)
    blob.upload_from_file(buf, content_type="application/octet-stream")

    size_mb = buf.getbuffer().nbytes / 1024 / 1024
    print(f"  Caricato gs://{BUCKET}/{blob_path}  shape={arr.shape}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    # Inizializza il client GCS usando le Application Default Credentials.
    # Assicurarsi di aver eseguito: gcloud auth application-default login
    client = storage.Client()
    bucket_obj = client.bucket(BUCKET)

    for split in ["train", "test"]:
        print(f"\n=== {split} ===")

        # Scarica e converte immagini e label
        X = download_and_parse_images(URLS[split]["images"])
        y = download_and_parse_labels(URLS[split]["labels"])

        # Carica entrambi i file su GCS
        upload_npy(bucket_obj, X, f"{PREFIX}/{split}/X.npy")
        upload_npy(bucket_obj, y, f"{PREFIX}/{split}/y.npy")

    print("\nDataset MNIST disponibile su GCS.")
    print(f"Verifica: gsutil ls -lh gs://{BUCKET}/{PREFIX}/")