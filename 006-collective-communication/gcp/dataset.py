# =============================================================================
# dataset.py — Dataset per la Lezione 7: Collective Communication Pattern
# =============================================================================
#
# Questo modulo espone due dataset alternativi:
#
#   SyntheticDataset  — dataset generato interamente in RAM a partire da un
#                       seed fisso. Non richiede GCS, download o credenziali.
#                       Usato negli esercizi locali (Ex 0–4) e nei benchmark.
#
#   GCSDataset        — dataset MNIST precaricato su GCS, identico a quello
#                       della Lezione 6. Usato nel Custom Training Job su
#                       Vertex AI (versione GCP della lezione).
#
# Il training loop in train_ddp.py sceglie quale usare in base alla variabile
# d'ambiente USE_SYNTHETIC (default "1" → sintetico). Questo permette di
# eseguire lo stesso script sia in locale che su GCP senza modifiche.
#
# Vantaggi del dataset sintetico per questa lezione:
#   - Elimina la variabile I/O dal benchmark di scaling: T_comm dipende
#     solo dalla comunicazione collettiva, non dalla velocità di lettura da GCS.
#   - Permette di completare gli esercizi offline, senza account GCP.
#   - Dimensione e numero di feature configurabili per stressare il sistema.
# =============================================================================

import io
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# =============================================================================
# DATASET 1 — Sintetico (default)
# =============================================================================

class SyntheticDataset(Dataset):
    """
    Dataset di classificazione binaria generato sinteticamente.

    Ogni campione è un vettore casuale di dimensione n_features.
    La label è 1 se la norma L2 del vettore supera la mediana del dataset,
    0 altrimenti. Questo crea un problema di classificazione linearmente
    non separabile ma sufficientemente semplice da convergere in poche epoch.

    La scelta della norma come label garantisce che la distribuzione delle
    classi sia bilanciata (50/50 per costruzione) indipendentemente dal seed,
    evitando problemi di class imbalance nel benchmark.

    Parametri
    ----------
    n_samples  : numero totale di campioni (default 4000)
    n_features : dimensione del vettore di input (default 128)
    seed       : seme per la riproducibilità del dataset
    """

    def __init__(
        self,
        n_samples:  int = 4000,
        n_features: int = 128,
        seed:       int = 42,
    ):
        # Generator con seed fisso: garantisce che tutti i rank ottengano
        # lo stesso dataset, precondizione per DistributedSampler corretto.
        rng = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n_samples, n_features, generator=rng)

        # Label binaria basata sulla norma: bilanciata per costruzione.
        norms = self.X.norm(dim=1)
        self.y = (norms > norms.median()).long()

        self.n_features = n_features
        self.n_classes  = 2

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =============================================================================
# DATASET 2 — GCS (solo su Vertex AI)
# =============================================================================

class GCSDataset(Dataset):
    """
    Dataset PyTorch che legge array NumPy da Google Cloud Storage.

    Compatibile con il formato prodotto da upload_mnist.py nella Lezione 6.
    Si aspetta i file X.npy e y.npy in gs://<bucket>/<path>/.

    Usato solo nel Custom Training Job su Vertex AI; negli esercizi locali
    si usa SyntheticDataset per evitare dipendenze da credenziali GCP.
    """

    def __init__(self, bucket: str, path: str):
        # Import lazy: gcsfs non è disponibile in tutti gli ambienti.
        # Importarlo qui (invece che a livello di modulo) evita errori
        # di ImportError su macchine locali prive di gcsfs.
        try:
            import gcsfs
        except ImportError as e:
            raise ImportError(
                "gcsfs non trovato. Installa con: pip install gcsfs\n"
                "Oppure imposta USE_SYNTHETIC=1 per usare il dataset sintetico."
            ) from e

        fs = gcsfs.GCSFileSystem(token="google_default")
        gcs_path = f"{bucket}/{path}"

        with fs.open(f"{gcs_path}/X.npy", "rb") as f:
            self.X = torch.tensor(
                np.load(io.BytesIO(f.read()), allow_pickle=False),
                dtype=torch.float32,
            )

        with fs.open(f"{gcs_path}/y.npy", "rb") as f:
            self.y = torch.tensor(
                np.load(io.BytesIO(f.read()), allow_pickle=False),
                dtype=torch.long,
            )

        self.n_features = self.X.shape[1]
        self.n_classes  = int(self.y.max().item()) + 1
        print(f"[dataset] Caricati {len(self.X)} campioni da gs://{gcs_path}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =============================================================================
# Funzione pubblica: costruisce il DataLoader distribuito corretto
# =============================================================================

def build_dataloader(
    batch_size:   int  = 128,
    shuffle:      bool = True,
    num_workers:  int  = 0,
    distributed:  bool = True,
    rank:         int  = 0,
    world_size:   int  = 1,
    seed:         int  = 42,
) -> tuple[DataLoader, Dataset]:
    """
    Costruisce il dataset e il DataLoader appropriati, con o senza
    DistributedSampler a seconda del flag `distributed`.

    Parametri
    ----------
    batch_size  : campioni per batch per rank (non totale)
    shuffle     : attivo solo in modalità non-distribuita
    num_workers : thread di prefetching del DataLoader
    distributed : se True, usa DistributedSampler (obbligatorio con DDP)
    rank        : rank del processo corrente (usato da DistributedSampler)
    world_size  : numero totale di processi (usato da DistributedSampler)
    seed        : seme per il dataset e per il sampler

    Returns
    -------
    (DataLoader, Dataset) — il loader pronto all'uso e il dataset sottostante
    """
    use_synthetic = os.environ.get("USE_SYNTHETIC", "1") == "1"

    if use_synthetic:
        dataset = SyntheticDataset(
            n_samples=int(os.environ.get("N_SAMPLES", "4000")),
            n_features=int(os.environ.get("N_FEATURES", "128")),
            seed=seed,
        )
    else:
        dataset = GCSDataset(
            bucket=os.environ.get("GCS_BUCKET", "ias-luigi-asprino-bucket"),
            path="datasets/mnist/train",
        )

    if distributed:
        # DistributedSampler garantisce che ogni rank veda una partizione
        # disgiunta del dataset. Con world_size=2 e 4000 campioni, ogni
        # rank riceve esattamente 2000 campioni per epoch.
        #
        # IMPORTANTE: shuffle=True in DistributedSampler NON è sufficiente
        # da solo per garantire permutazioni diverse tra epoch. È necessario
        # chiamare sampler.set_epoch(epoch) nel loop di training — vedi train_ddp.py.
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,        # sampler e shuffle si escludono a vicenda
            num_workers=num_workers,
            pin_memory=False,       # True solo con GPU (trasferimento CPU→VRAM)
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return loader, dataset
