import os
import io
import torch
import gcsfs
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── GCS Filesystem ───────────────────────────────────────────────────────
# gcsfs espone Google Cloud Storage come un filesystem Python standard.
# token="google_default" usa le credenziali Application Default Credentials
# (ADC) già configurate nell'ambiente Vertex AI — nessuna chiave esplicita
# necessaria all'interno del container sul cluster.
fs = gcsfs.GCSFileSystem(token="google_default")


# ── Dataset class ────────────────────────────────────────────────────────
class GCSDataset(Dataset):
    """
    Dataset PyTorch che legge tensori pre-salvati da GCS.
    Si aspetta che su GCS esistano due file nella stessa cartella:
      - X.npy  → feature matrix, shape (N, 784) per MNIST flattenato
      - y.npy  → label vector,   shape (N,)

    I file vengono caricati in RAM all'inizializzazione (__init__) e
    restano in memoria per tutta la durata del training — adatto per
    dataset di dimensione medio-piccola come MNIST (< 1 GB).
    Per dataset grandi si userebbe uno streaming lazy, ma esula dagli
    obiettivi di questa lezione.
    """

    def __init__(self, bucket: str, path: str):
        """
        bucket : nome del bucket GCS, es. "ias-luigi-asprino-bucket"
        path   : percorso relativo alla radice del bucket,
                 es. "datasets/mnist/train"
        """
        gcs_path = f"{bucket}/{path}"

        # Legge X.npy direttamente da GCS in un buffer in memoria.
        # fs.open() restituisce un file-like object compatibile con np.load.
        # allow_pickle=False è una misura di sicurezza: i file .npy usati
        # qui contengono solo array numerici, non oggetti Python serializzati.
        with fs.open(f"{gcs_path}/X.npy", "rb") as f:
            self.X = torch.tensor(
                np.load(io.BytesIO(f.read()), allow_pickle=False),
                dtype=torch.float32
            )

        with fs.open(f"{gcs_path}/y.npy", "rb") as f:
            self.y = torch.tensor(
                np.load(io.BytesIO(f.read()), allow_pickle=False),
                dtype=torch.long   # CrossEntropyLoss richiede long per le label
            )

        print(f"[dataset] Caricati {len(self.X)} campioni da gs://{gcs_path}")

    def __len__(self):
        # Chiamato da DataLoader per sapere quanti campioni ci sono in totale
        return len(self.X)

    def __getitem__(self, idx):
        # Chiamato da DataLoader per recuperare il singolo campione idx.
        # Restituisce una tupla (features, label) — il formato atteso
        # dal training loop in train_ps.py.
        return self.X[idx], self.y[idx]


# ── Funzione pubblica chiamata da train_ps.py ────────────────────────────
def load_from_gcs(
    bucket: str,
    path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    Crea e restituisce un DataLoader PyTorch che itera sul dataset GCS.

    batch_size  : numero di campioni per batch; 64 è un buon default per
                  MNIST su CPU — aumentare se si usano GPU.
    shuffle     : True durante il training per rompere l'ordine dei campioni
                  ed evitare che il modello memorizzi la sequenza.
    num_workers : thread paralleli per il prefetching dei batch.
                  Con 2 worker il DataLoader prepara il batch N+1 mentre
                  il training processa il batch N, riducendo l'idle time.
                  Impostare a 0 per debugging (single-threaded, stack trace
                  più leggibili in caso di errore).
    """
    dataset = GCSDataset(bucket=bucket, path=path)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        # pin_memory=True velocizza il trasferimento CPU→GPU copiando i
        # batch in memoria pinned (non paginabile). Disabilitato di default
        # perché in questa lezione usiamo CPU; abilitare se si aggiungono GPU.
        pin_memory  = False
    )


# ── Script di upload del dataset (eseguito una volta, da Workbench) ──────
# Questo blocco non viene eseguito durante il training — serve solo
# al docente per caricare MNIST su GCS prima della lezione, da eseguire
# una volta nel notebook Workbench (come in L3).
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from google.cloud import storage

    BUCKET = os.environ.get("GCS_BUCKET", "ias-luigi-asprino-bucket")
    PREFIX = "datasets/mnist"

    print("Scarico MNIST da torchvision...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Flatten: converte (1, 28, 28) → (784,) per SimpleNet fully-connected
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_data = datasets.MNIST(root="/tmp", train=True,  download=True, transform=transform)
    test_data  = datasets.MNIST(root="/tmp", train=False, download=True, transform=transform)

    client = storage.Client()
    bucket = client.bucket(BUCKET)

    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        # Converte il dataset in numpy array per il salvataggio come .npy
        X = np.stack([x.numpy() for x, _ in split_data])
        y = np.array([label for _, label in split_data])

        for arr, name in [(X, "X"), (y, "y")]:
            buf = io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)

            blob_path = f"{PREFIX}/{split_name}/{name}.npy"
            blob = bucket.blob(blob_path)
            blob.upload_from_file(buf, content_type="application/octet-stream")
            print(f"Caricato gs://{BUCKET}/{blob_path}  shape={arr.shape}")

    print("Dataset MNIST disponibile su GCS.")