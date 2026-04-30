import math
import gcsfs
from torch.utils.data import Dataset
import torch


class GCSShardDataset(Dataset):
    """
    Dataset lazy su GCS: non carica nulla in __init__.
    Ogni chiamata a __getitem__ legge solo il campione richiesto
    dallo shard corretto. Gli shard vengono cachati in memoria
    per worker, quindi ogni shard viene scaricato da GCS una
    sola volta per worker per epoch.
    """

    def __init__(self, bucket_prefix, num_shards, shard_size, total_samples):
        """
        bucket_prefix: es. 'ml-lab-bucket'
        num_shards:    numero di file shard-000.pt … shard-N.pt
        shard_size:    numero di campioni per shard
                       (l'ultimo shard può essere più corto)
        total_samples: lunghezza effettiva totale del dataset
        """
        self.num_shards    = num_shards
        self.shard_size    = shard_size
        self.total_samples = total_samples # Store the actual total samples
        self.shard_uris    = [
            f'gs://{bucket_prefix}/shards/shard-{i:03d}.pt'
            for i in range(num_shards)
        ]

        # Cache locale per worker: dizionario {shard_id: chunk}
        # Ogni processo worker ha la propria copia in memoria →
        # nessuna contesa tra worker
        self._cache = {}
        self._fs = None # Inizializza a None; verrà creato da ogni worker

    def __len__(self):
        # La lunghezza totale non richiede di leggere nulla da GCS
        return self.total_samples # Return the actual total number of samples

    def _get_fs(self):
        """Initializes gcsfs.GCSFileSystem lazily per worker."""
        if self._fs is None:
            self._fs = gcsfs.GCSFileSystem(token="google_default")

        return self._fs

    def _load_shard(self, shard_id):
        """Scarica uno shard da GCS solo se non è già in cache."""

        if shard_id not in self._cache:

            uri = self.shard_uris[shard_id]

            # Use the worker-local fs instance
            with self._get_fs().open(uri, 'rb') as f:
                self._cache[shard_id] = torch.load(f) # Directly pass the file object


            # Tieni in cache al massimo 2 shard per worker
            # per evitare di esaurire la RAM su dataset molto grandi
            if len(self._cache) > 2:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        return self._cache[shard_id]

    def __getitem__(self, idx):
        # Calcola in quale shard si trova l'indice globale idx
        shard_id     = idx // self.shard_size
        local_idx    = idx  % self.shard_size   # posizione dentro lo shard

        chunk = self._load_shard(shard_id)

        # Gestisce l'ultimo shard che può essere più corto
        if local_idx >= len(chunk['X']):
            raise IndexError(
                f"idx={idx} fuori range: shard {shard_id} "
                f"ha solo {len(chunk['X'])} campioni"
            )

        return chunk['X'][local_idx], chunk['y'][local_idx]
