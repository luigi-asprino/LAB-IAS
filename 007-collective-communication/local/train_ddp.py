# =============================================================================
# train_ddp.py — Training distribuito con PyTorch DDP e All-Reduce
# =============================================================================
#
# Questo script implementa il Collective Communication Pattern usando
# torch.nn.parallel.DistributedDataParallel (DDP).
#
# A differenza del Parameter Server (Lezione 6), qui NON esiste un nodo
# centrale: tutti i processi sono simmetrici. La sincronizzazione dei
# parametri avviene tramite all-reduce collettiva implementata da NCCL
# (su GPU) o gloo (su CPU).
#
# Lancio locale (esercizi):
#   torchrun --nproc_per_node=1 train_ddp.py   # baseline
#   torchrun --nproc_per_node=2 train_ddp.py   # 2 worker, all-reduce visibile
#   torchrun --nproc_per_node=4 train_ddp.py   # 4 worker
#
# Lancio su Vertex AI:
#   gcloud ai custom-jobs create --region=europe-west4 --config=job_config.yaml
#   (il container usa lo stesso script con backend='nccl' e device=cuda)
#
# Variabili d'ambiente configurabili:
#   USE_SYNTHETIC  "1" (default) = dataset sintetico | "0" = GCS
#   N_SAMPLES      campioni nel dataset sintetico (default 4000)
#   N_FEATURES     feature per campione (default 128)
#   BATCH_SIZE     campioni per batch per rank (default 128)
#   N_EPOCHS       numero di epoch (default 5)
#   LR             learning rate SGD (default 0.05)
#   GCS_BUCKET     bucket GCS (solo se USE_SYNTHETIC=0)
#   GCS_MODEL_PATH path GCS per il salvataggio del modello (default models/lezione7/)
# =============================================================================

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import build_dataloader
from model import MLP


# =============================================================================
# SEZIONE 1 — Setup del processo group
# =============================================================================

def setup() -> tuple[int, int, torch.device]:
    """
    Inizializza il processo group DDP e restituisce (rank, world_size, device).

    Scelta del backend:
      - Se CUDA è disponibile → "nccl"  (GPU-to-GPU, ottimale con NVLink)
      - Altrimenti            → "gloo"  (CPU, socket TCP/loopback)

    Su Vertex AI con GPU T4, il backend sarà "nccl"; in locale su CPU sarà "gloo".
    Il codice di training è identico in entrambi i casi: solo il backend cambia.
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",   # Legge RANK, WORLD_SIZE, MASTER_ADDR da env
    )

    rank       = dist.get_rank() # Indice univoco del processo (0, 1, ..., world_size-1)
    world_size = dist.get_world_size() # Numero totale di processi (es. 4 se --nproc_per_node=4)

    if torch.cuda.is_available():
        # Con più GPU per macchina, ogni rank usa una GPU diversa.
        # LOCAL_RANK è iniettato da torchrun/Vertex AI per indicare
        # l'indice della GPU su questa specifica macchina (0, 1, ...).
        local_rank = int(os.environ.get("LOCAL_RANK", rank)) # Fallback a rank se LOCAL_RANK non è definito (es. in lancio manuale)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"[init] backend={backend}  world_size={world_size}  device={device}")

    return rank, world_size, device


def teardown():
    dist.destroy_process_group()


# =============================================================================
# SEZIONE 2 — Training loop
# =============================================================================

def train(rank: int, world_size: int, device: torch.device):
    """
    Loop di training DDP.

    Il flusso è identico al training su singolo nodo, con tre differenze:
      1. Il modello è avvolto in DDP(model): questo intercetta .backward()
         e inserisce l'all-reduce sui gradienti prima di restituire il controllo.
      2. Il DataLoader usa DistributedSampler per garantire partizioni disgiunte.
      3. sampler.set_epoch(epoch) deve essere chiamato ad ogni epoch per
         garantire che il rimescolamento sia diverso da epoch a epoch.

    Tutto il resto (forward, loss, optimizer.step) è invariato rispetto al
    training su singolo nodo — questa è la principale forza di DDP.
    """
    # ── Iperparametri da variabili d'ambiente ───────────────────────────────
    batch_size = int(os.environ.get("BATCH_SIZE", "128"))
    n_epochs   = int(os.environ.get("N_EPOCHS", "5"))
    lr         = float(os.environ.get("LR", "0.05"))

    # ── Dataset e DataLoader ────────────────────────────────────────────────
    loader, dataset = build_dataloader(
        batch_size  = batch_size,
        distributed = True,
        rank        = rank,
        world_size  = world_size,
    )

    n_features = dataset.n_features
    n_classes  = dataset.n_classes

    if rank == 0: # Solo rank 0 stampa le informazioni sul dataset per evitare output duplicati
        n_per_rank = len(dataset) // world_size
        print(f"[data]  {len(dataset)} campioni totali | "
              f"{n_per_rank} per rank | "
              f"batch_size={batch_size}")

    # ── Modello ─────────────────────────────────────────────────────────────
    model = MLP(in_features=n_features, n_classes=n_classes).to(device) # Spostiamo il modello sulla GPU/CPU locale PRIMA di avvolgerlo in DDP.

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ DDP WRAPPING — il punto centrale di tutta la lezione               │
    # │                                                                     │
    # │ DDP(model) fa DUE cose importanti:                                 │
    # │                                                                     │
    # │ 1. BROADCAST iniziale: al momento del wrapping, DDP esegue un      │
    # │    broadcast dei parametri dal rank 0 a tutti gli altri rank.       │
    # │    Questo garantisce che tutti i processi partano dagli stessi      │
    # │    pesi, indipendentemente dal seed di inizializzazione locale.     │
    # │                                                                     │
    # │ 2. ALL-REDUCE implicita: DDP intercetta ogni chiamata a            │
    # │    loss.backward() e, prima di restituire il controllo, esegue     │
    # │    un'all-reduce (SUM, poi ÷ world_size) sui gradienti di ogni    │
    # │    parametro. Questo avviene tramite "gradient hooks" registrati    │
    # │    su ogni parametro — non è visibile nel codice utente.           │
    # │                                                                     │
    # │ Il risultato: dopo .backward(), ogni rank ha gradienti identici,   │
    # │ la media globale calcolata sull'intero dataset distribuito.         │
    # │ optimizer.step() aggiorna i parametri in modo coerente su tutti.   │
    # └─────────────────────────────────────────────────────────────────────┘
    model = DDP(model, device_ids=[device] if torch.cuda.is_available() else None)

    if rank == 0:
        inner = model.module  # model.module è il modello originale non-DDP
        print(f"[model] {inner.param_count():,} parametri | "
              f"{inner.param_bytes()/1024:.1f} KB all-reduce/iterazione")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # ── Loop di training ─────────────────────────────────────────────────
    t_train_start = time.perf_counter()

    for epoch in range(n_epochs):
        # FONDAMENTALE: set_epoch() aggiorna il seed interno del sampler.
        # Senza questa chiamata, DistributedSampler userebbe lo stesso
        # ordine di campionamento ad ogni epoch — equivalente a non fare shuffle.
        #
        # Accediamo al sampler tramite loader.sampler (attributo standard).
        loader.sampler.set_epoch(epoch)

        model.train()
        total_loss  = 0.0
        n_batches   = 0
        t_epoch_start = time.perf_counter()

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # loss.backward() ← QUI DDP esegue l'all-reduce sui gradienti
            #
            # Internamente DDP usa "gradient bucketing": raccoglie i gradienti
            # dei layer in bucket da ~25 MB e avvia l'all-reduce su ogni
            # bucket appena è pieno, SOVRAPPOSTO al backward dei layer
            # successivi. Questo nasconde parzialmente la latenza di rete.
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        t_epoch = time.perf_counter() - t_epoch_start

        # Riduzione della loss globale su rank 0 per il logging.
        # all_reduce(SUM) + divisione = media globale tra i rank.
        loss_tensor = torch.tensor(total_loss / n_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

        if rank == 0:
            throughput = len(dataset) / t_epoch
            print(f"Epoch {epoch+1:>2}/{n_epochs} | "
                  f"loss={avg_loss:.4f} | "
                  f"tempo={t_epoch:.2f}s | "
                  f"throughput={throughput:.0f} camp/s")

    t_total = time.perf_counter() - t_train_start
    if rank == 0:
        print(f"\nTraining completato in {t_total:.1f}s")

    # ── Salvataggio del modello (solo rank 0) ──────────────────────────────
    if rank == 0:
        _save_model(model.module, world_size)


def _save_model(model: MLP, world_size: int):
    """
    Salva i pesi del modello.
    Locale: /tmp/model_lezione7.pth
    GCS: gs://<GCS_BUCKET>/<GCS_MODEL_PATH>model_w<world_size>.pth
    """
    use_synthetic = os.environ.get("USE_SYNTHETIC", "1") == "1"

    if use_synthetic:
        local_path = f"/tmp/model_lezione7_w{world_size}.pth"
        torch.save(model.state_dict(), local_path)
        print(f"[save] Modello salvato in {local_path}")
        return

    # Salvataggio su GCS (solo in modalità Vertex AI)
    try:
        import io
        from google.cloud import storage as gcs_storage

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        bucket_name = os.environ.get("GCS_BUCKET", "ias-luigi-asprino-bucket")
        model_path  = os.environ.get("GCS_MODEL_PATH", "models/lezione7/")
        blob_name   = f"{model_path}model_w{world_size}.pth"

        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)
        blob.upload_from_file(buf, content_type="application/octet-stream")
        print(f"[save] Modello salvato su gs://{bucket_name}/{blob_name}")

    except Exception as e:
        print(f"[save] Errore nel salvataggio su GCS: {e}")
        local_path = f"/tmp/model_lezione7_w{world_size}.pth"
        torch.save(model.state_dict(), local_path)
        print(f"[save] Fallback: salvato in {local_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    rank, world_size, device = setup()
    try:
        train(rank, world_size, device)
    finally:
        # teardown() viene chiamato anche in caso di eccezione per evitare
        # che gli altri rank rimangano in attesa su una barriera fantasma.
        teardown()
