# =============================================================================
# dataset.py — Dataset per la Lezione 8: Elasticity e Fault-Tolerance
# =============================================================================
#
# Identico alla Lezione 7. Viene riusato direttamente.
#
# In questa lezione il dataset sintetico è la scelta obbligata per gli
# esercizi locali: il training deve durare abbastanza da permettere di
# osservare una preemption simulata. Aumentare N_EPOCHS e N_SAMPLES
# (vedi variabili d'ambiente in train_spot.py) allunga il training.
# =============================================================================

import io
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SyntheticDataset(Dataset):
    """
    Dataset di classificazione binaria generato sinteticamente.
    Label basata sulla norma L2 del vettore di input (bilanciata 50/50).
    """

    def __init__(
        self,
        n_samples:  int = 4000,
        n_features: int = 128,
        seed:       int = 42,
    ):
        rng = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n_samples, n_features, generator=rng)
        norms = self.X.norm(dim=1)
        self.y = (norms > norms.median()).long()
        self.n_features = n_features
        self.n_classes  = 2

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GCSDataset(Dataset):
    """Dataset che legge array NumPy da GCS. Usato solo su Vertex AI."""

    def __init__(self, bucket: str, path: str):
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


def build_dataloader(
    batch_size:  int  = 128,
    shuffle:     bool = True,
    num_workers: int  = 0,
    seed:        int  = 42,
) -> tuple[DataLoader, Dataset]:
    """
    Costruisce il dataset e il DataLoader.

    In questa lezione NON usiamo DistributedSampler: il job è single-node
    (una sola VM Spot). L'obiettivo è il checkpoint, non il parallelismo.
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
            bucket=os.environ.get("GCS_BUCKET", "ias-luigi-asprino-bucket-eu"),
            path="datasets/mnist/train",
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader, dataset
