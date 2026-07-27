"""
Esercizio 1 — Training distribuito resiliente (SOLUZIONE COMMENTATA)
=========================================================================

Implementazione completa dei blocchi (a), (b), (c). Il punto (d) è una
discussione, riportata in fondo al file.

Come lanciarlo (simulazione locale multi-processo su CPU, backend "gloo"):
    python train_solution.py --world-size 2 --epochs 2

Per testare la resilienza alla prelazione: lanciarlo e poi, da un altro
terminale, inviare SIGTERM a uno dei processi worker (es. `kill -TERM <pid>`)
e osservare che al riavvio lo script riparte dall'ultimo checkpoint invece
che da zero.
"""

import argparse
import functools
import hashlib
import json
import os
import pickle
import signal
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ---------------------------------------------------------------------------
# Modello e dataset
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Piccola CNN per immagini 3x32x32 (formato CIFAR-10)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


class SyntheticImageDataset(Dataset):
    """Dataset sintetico con la stessa forma di CIFAR-10.

    In produzione, sostituire con torchvision.datasets.CIFAR10 puntando a
    una copia del dataset su GCS montata/scaricata in locale (Lezione 3).
    Usiamo dati sintetici qui solo per rendere l'esercizio autosufficiente
    e velocissimo da eseguire in aula, senza download.
    """

    def __init__(self, num_samples: int = 512, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples, 3, 32, 32, generator=g)
        self.y = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# (c) Step memoization — Lezione 11
# ---------------------------------------------------------------------------
#
# Idea del pattern: se uno step della pipeline è puro (stesso input ->
# stesso output) e costoso, calcolarlo una volta e riusare il risultato
# cachato per tutte le esecuzioni successive (riavvii dopo prelazione,
# altri worker, run successive con lo stesso dataset).
#
# La cache key deve dipendere SOLO da cio' che determina il risultato
# (qui: i dati del dataset), non da dettagli accidentali come l'orario
# di esecuzione.

def _hash_dataset(dataset: Dataset) -> str:
    """Chiave deterministica basata sul contenuto del dataset.

    In un caso reale con dataset molto grandi si farebbe hash di metadati
    (path, dimensione, timestamp di ultima modifica) invece che dei dati
    stessi, per evitare di leggere tutto il dataset solo per calcolare la
    chiave.
    """
    h = hashlib.sha256()
    h.update(dataset.x.numpy().tobytes())
    h.update(dataset.y.numpy().tobytes())
    return h.hexdigest()[:16]


def memoize(cache_dir: str):
    """Decoratore di step memoization.

    Il risultato della funzione decorata viene salvato su disco (o su GCS,
    se cache_dir e' un path gs://) indicizzato da una chiave deterministica
    calcolata sugli argomenti. Le chiamate successive con lo stesso input
    leggono direttamente dalla cache.
    """
    os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(dataset, *args, **kwargs):
            key = _hash_dataset(dataset)
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{key}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                print(f"[memoize] cache HIT per {func.__name__} (key={key})")
                return result

            print(f"[memoize] cache MISS per {func.__name__} (key={key}): calcolo...")
            result = func(dataset, *args, **kwargs)

            # Scrittura atomica: file temporaneo + rename, per non lasciare
            # una cache corrotta se il processo viene interrotto a metà.
            fd, tmp_path = tempfile.mkstemp(dir=cache_dir)
            with os.fdopen(fd, "wb") as f:
                pickle.dump(result, f)
            os.replace(tmp_path, cache_file)

            return result
        return wrapper
    return decorator


@memoize(cache_dir="./cache")
def compute_dataset_stats(dataset: Dataset):
    """Step costoso e deterministico: calcola media/std del dataset."""
    xs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    return {"mean": xs.mean().item(), "std": xs.std().item()}


# ---------------------------------------------------------------------------
# (b) Elasticity e fault-tolerance — Lezione 8 (+ Lezione 3, GCS)
# ---------------------------------------------------------------------------
#
# Il punto chiave del pattern: una Spot VM su GCP può essere revocata in
# qualsiasi momento, con un preavviso di ~30s segnalato via SIGTERM. Il job
# deve quindi:
#   1. salvare periodicamente lo stato (checkpoint) durante il training;
#   2. salvare uno stato anche appena arriva il segnale di prelazione;
#   3. all'avvio, controllare se esiste un checkpoint e ripartire da lì
#      invece che da zero.
#
# I checkpoint vanno scritti su uno storage che sopravvive alla singola VM
# (GCS, Lezione 3), non sul disco locale della VM che viene distrutta.

def _checkpoint_file(checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "checkpoint.pt")


def load_checkpoint_if_any(model: nn.Module, optimizer, checkpoint_dir: str) -> int:
    """Ripristina lo stato da checkpoint, se presente.

    Nota: model e' il modulo DDP; si carica lo state_dict nel .module
    sottostante cosi' che tutti i rank ripartano con pesi identici (DDP
    sincronizza comunque i gradienti ad ogni backward, ma partire già
    allineati evita un primo passo di all-reduce "sprecato").
    """
    path = _checkpoint_file(checkpoint_dir)
    if not os.path.exists(path):
        return 0

    state = torch.load(path, map_location="cpu")
    model.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"[checkpoint] ripristinato da {path}, riparto dall'epoca {state['epoch']}")
    return state["epoch"]


def save_checkpoint(model: nn.Module, optimizer, epoch: int, checkpoint_dir: str) -> None:
    """Salva lo stato corrente in modo atomico.

    Solo rank 0 scrive il checkpoint: tutti i rank hanno pesi identici
    grazie alla sincronizzazione DDP, quindi scriverlo da più rank sarebbe
    ridondante (e rischierebbe una race condition sullo stesso file).
    """
    path = _checkpoint_file(checkpoint_dir)
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # rename atomico: niente checkpoint a metà
    print(f"[checkpoint] salvato in {path} (epoca {epoch})")


def install_preemption_handler(save_fn, rank: int):
    """Registra un handler di SIGTERM che esegue un checkpoint di emergenza.

    Su una Spot VM reale, GCP invia SIGTERM circa 30 secondi prima dello
    shutdown effettivo: e' il tempo che abbiamo per salvare lo stato.
    """
    def handler(signum, frame):
        print(f"[rank {rank}] SIGTERM ricevuto: prelazione in corso, checkpoint di emergenza...")
        save_fn()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------
# (a) Collective communication pattern — Lezione 7
# ---------------------------------------------------------------------------
#
# A differenza del parameter server pattern (Lezione 6), qui non esiste un
# nodo centrale che aggrega i gradienti: ogni worker calcola i propri
# gradienti locali e li sincronizza con tutti gli altri tramite un'
# operazione collettiva (all-reduce). DistributedDataParallel automatizza
# esattamente questo: dopo ogni backward(), i gradienti di tutti i
# parametri sono già mediati fra i rank.

def setup_process_group(rank: int, world_size: int, backend: str = "gloo"):
    """Inizializza il process group per la comunicazione collettiva.

    In un vero Custom Training Job su Vertex AI, MASTER_ADDR/MASTER_PORT
    e RANK/WORLD_SIZE sono già impostati dal servizio (tramite la variabile
    CLUSTER_SPEC); qui li impostiamo a mano per simulare più worker sulla
    stessa macchina.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_process_group():
    dist.destroy_process_group()


def build_ddp_model(rank: int, backend: str) -> nn.Module:
    """Costruisce il modello e lo avvolge in DistributedDataParallel.

    Con "nccl" (GPU) si passerebbe device_ids=[rank] dopo aver spostato
    modello e batch sulla GPU corrispondente; con "gloo" su CPU (come in
    questa simulazione locale) non serve device_ids.
    """
    model = SimpleCNN()
    if backend == "nccl":
        model = model.to(rank)
        return DDP(model, device_ids=[rank])
    return DDP(model)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_worker(rank: int, world_size: int, args: argparse.Namespace):
    setup_process_group(rank, world_size, backend=args.backend)

    dataset = SyntheticImageDataset(num_samples=args.num_samples)

    # (c) La chiamata e' identica su ogni worker: solo il primo che arriva
    # (di solito rank 0, ma non e' garantito) fa il calcolo, gli altri
    # trovano la cache già scritta e la leggono.
    stats = compute_dataset_stats(dataset)
    if rank == 0:
        print(f"[rank {rank}] dataset stats: {stats}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = build_ddp_model(rank, backend=args.backend)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    start_epoch = load_checkpoint_if_any(model, optimizer, args.checkpoint_dir)
    install_preemption_handler(
        lambda: save_checkpoint(model, optimizer, start_epoch, args.checkpoint_dir),
        rank=rank,
    )

    loss = torch.tensor(0.0)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)  # garantisce shuffling diverso e coordinato ad ogni epoca
        for step, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()      # qui avviene l'all-reduce dei gradienti (collective communication)
            optimizer.step()

            if step % args.checkpoint_every == 0 and rank == 0:
                save_checkpoint(model, optimizer, epoch, args.checkpoint_dir)

        if rank == 0:
            print(f"[rank {rank}] epoch {epoch} completata, loss={loss.item():.4f}")

    cleanup_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--backend", type=str, default="gloo")
    args = parser.parse_args()

    mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# (d) Discussione: parameter server pattern vs collective communication
# ---------------------------------------------------------------------------
#
# Con un parameter server pattern (Lezione 6) l'architettura cambierebbe
# radicalmente:
#   - servirebbero due ruoli distinti di processo: uno o più "parameter
#     server" che mantengono i pesi correnti, e i "worker" che calcolano i
#     gradienti su una porzione di batch e li inviano al server.
#   - non ci sarebbe più un DistributedDataParallel: i worker userebbero un
#     modello locale sincronizzato via pull/push espliciti (get_parameters /
#     push_gradients) invece dell'all-reduce automatico di DDP.
#   - il parameter server diventa un potenziale collo di bottiglia (riceve
#     traffico da tutti i worker) e un single point of failure da rendere
#     a sua volta resiliente (richiama il tema della Lezione 8 anche per il
#     server, non solo per i worker).
#   - per contro, la topologia è più flessibile per scenari asincroni
#     (worker che procedono a velocità diverse), cosa che collective
#     communication (sincrono per natura) non supporta bene.
