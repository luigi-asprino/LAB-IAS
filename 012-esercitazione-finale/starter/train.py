"""
Esercizio 1 — Training distribuito resiliente (STARTER)
=========================================================

Punto di partenza: un training loop PyTorch che funziona su un solo
processo/GPU. Il compito e' completare i blocchi TODO per trasformarlo in
un job distribuito, resiliente alla prelazione di una Spot VM ed efficiente
grazie alla step memoization.

Pattern coinvolti (richiamo alle lezioni del laboratorio):
  (a) Collective communication pattern      -> Lezione 7
  (b) Elasticity e fault-tolerance           -> Lezione 8 (+ Lezione 3, GCS)
  (c) Step memoization                       -> Lezione 11
  (d) Discussione: parameter server pattern  -> Lezione 6 (nessun codice da scrivere)

Come lanciarlo (simulazione locale multi-processo su CPU con backend "gloo"):
    python train.py --world-size 2 --epochs 1

In un vero Custom Training Job su Vertex AI, world-size e rank/master-addr
vengono impostati dal servizio stesso (variabili d'ambiente), e il backend
diventa "nccl" se si usano GPU.
"""

import argparse
import functools
import hashlib
import json
import os
import signal
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ---------------------------------------------------------------------------
# Modello e dataset (già pronti, non richiedono modifiche)
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

    In aula, sostituire con torchvision.datasets.CIFAR10 puntando a una
    copia del dataset scaricata su GCS (richiamo Lezione 3).
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
# TODO (c): completare il decoratore "memoize" cosi' che:
#   1. calcoli una chiave deterministica a partire dagli argomenti della
#      funzione decorata (es. hash del path del dataset + eventuali kwargs);
#   2. se in "cache_dir" esiste già un risultato per quella chiave, lo
#      carichi da disco ed evitare di ricalcolare;
#   3. altrimenti esegua la funzione originale e salvi il risultato prima
#      di restituirlo.
#
# In produzione "cache_dir" potrebbe essere un path gs:// su GCS condiviso
# fra i worker, cosi' che il calcolo venga fatto una sola volta in assoluto.

def memoize(cache_dir: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # TODO (c): costruire la cache key, controllare/leggere/scrivere
            # il file di cache in cache_dir, e restituire il risultato.
            raise NotImplementedError("TODO (c): implementare la memoization")
        return wrapper
    return decorator


@memoize(cache_dir="./cache")
def compute_dataset_stats(dataset: Dataset):
    """Step costoso e deterministico: calcola media/std del dataset.

    E' l'esempio "canonico" di step memoization: dato lo stesso dataset,
    il risultato è sempre lo stesso e non ha senso ricalcolarlo ad ogni
    riavvio/riesecuzione del job.
    """
    xs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    return {"mean": xs.mean().item(), "std": xs.std().item()}


# ---------------------------------------------------------------------------
# (b) Elasticity e fault-tolerance — Lezione 8 (+ Lezione 3, GCS)
# ---------------------------------------------------------------------------

def checkpoint_path(checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "checkpoint.pt")


def load_checkpoint_if_any(model, optimizer, checkpoint_dir: str) -> int:
    """Ripristina lo stato da checkpoint, se presente.

    Ritorna l'epoca da cui riprendere (0 se non c'e' alcun checkpoint).

    TODO (b): implementare la lettura del checkpoint (torch.load) e il
    ripristino di model.state_dict() / optimizer.state_dict(). Il path
    puo' essere locale o gs:// (in quel caso, scaricare prima il file
    con la libreria google-cloud-storage).
    """
    raise NotImplementedError("TODO (b): implementare load_checkpoint_if_any")


def save_checkpoint(model, optimizer, epoch: int, checkpoint_dir: str) -> None:
    """Salva lo stato corrente su disco/GCS.

    TODO (b): salvare un dict con model.state_dict(), optimizer.state_dict()
    ed epoch, in modo atomico (es. scrivere su file temporaneo e poi
    rinominare) per evitare checkpoint corrotti in caso di prelazione a
    metà scrittura.
    """
    raise NotImplementedError("TODO (b): implementare save_checkpoint")


def install_preemption_handler(save_fn):
    """Le Spot VM su GCP notificano la prelazione con un SIGTERM e ~30s
    di preavviso prima dello shutdown. Registrare un handler che esegue
    un checkpoint di emergenza appena arriva il segnale.
    """
    def handler(signum, frame):
        print(f"[rank?] SIGTERM ricevuto: prelazione Spot VM in corso, checkpoint di emergenza...")
        save_fn()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------
# (a) Collective communication pattern — Lezione 7
# ---------------------------------------------------------------------------

def setup_process_group(rank: int, world_size: int, backend: str = "gloo"):
    """TODO (a): inizializzare il process group distribuito.

    Suggerimento: impostare MASTER_ADDR/MASTER_PORT nell'ambiente (per la
    simulazione locale bastano "localhost" e una porta libera) e chiamare
    dist.init_process_group(backend=..., rank=rank, world_size=world_size).
    """
    raise NotImplementedError("TODO (a): implementare setup_process_group")


def cleanup_process_group():
    dist.destroy_process_group()


def build_ddp_model(rank: int) -> nn.Module:
    """TODO (a): costruire il modello e avvolgerlo in DistributedDataParallel.

    Con backend "gloo" su CPU non si passa device_ids; con "nccl" su GPU
    passare device_ids=[rank] dopo aver spostato modello e rank sulla GPU
    corrispondente.
    """
    raise NotImplementedError("TODO (a): implementare build_ddp_model")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_worker(rank: int, world_size: int, args: argparse.Namespace):
    setup_process_group(rank, world_size, backend=args.backend)

    dataset = SyntheticImageDataset(num_samples=args.num_samples)

    # (c) memoization: chiamata identica su ogni worker/riavvio, ma il
    # calcolo effettivo avviene una sola volta.
    stats = compute_dataset_stats(dataset)
    if rank == 0:
        print(f"[rank {rank}] dataset stats: {stats}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = build_ddp_model(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    start_epoch = load_checkpoint_if_any(model, optimizer, args.checkpoint_dir)
    install_preemption_handler(
        lambda: save_checkpoint(model, optimizer, start_epoch, args.checkpoint_dir)
    )

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

            if step % args.checkpoint_every == 0 and rank == 0:
                save_checkpoint(model, optimizer, epoch, args.checkpoint_dir)

        if rank == 0:
            print(f"[rank {rank}] epoch {epoch} completata, loss={loss.item():.4f}")

    cleanup_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--backend", type=str, default="gloo")
    args = parser.parse_args()

    mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
