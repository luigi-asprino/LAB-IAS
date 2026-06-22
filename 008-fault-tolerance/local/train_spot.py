# =============================================================================
# train_spot.py — Training con Spot VM, SIGTERM handler e checkpoint automatico
# =============================================================================
#
# Questo script implementa il pattern fault-tolerant per il training ML:
#   1. All'avvio, cerca un checkpoint esistente su GCS (o locale) e riprende
#      dall'epoch successiva all'ultima completata (se esiste).
#   2. Ogni CHECKPOINT_INTERVAL epoch, salva il checkpoint.
#   3. Intercetta il segnale SIGTERM (inviato da GCP 30 secondi prima della
#      preemption) e salva immediatamente il checkpoint prima di uscire.
#
# Il codice è progettato per girare sia in locale (USE_SYNTHETIC=1, nessuna
# credenziale GCP) che su Vertex AI con Spot VM (USE_SYNTHETIC=0, GCS).
#
# Lancio locale (simulazione preemption con SIGTERM):
#   # Terminale 1:
#   USE_SYNTHETIC=1 N_EPOCHS=20 python train_spot.py
#
#   # Terminale 2 (dopo qualche epoch):
#   kill -SIGTERM $(pgrep -f train_spot.py)
#
#   # Riavvio:
#   USE_SYNTHETIC=1 N_EPOCHS=20 python train_spot.py
#   # → vedrai "Ripresa da epoch=N"
#
# Lancio su Vertex AI:
#   gcloud ai custom-jobs create --region=europe-west4 --config=gcp/job_config.yaml
#
# Variabili d'ambiente configurabili:
#   USE_SYNTHETIC        "1" (default) = dataset sintetico | "0" = GCS
#   N_SAMPLES            campioni nel dataset sintetico (default 4000)
#   N_FEATURES           feature per campione (default 128)
#   BATCH_SIZE           campioni per batch (default 128)
#   N_EPOCHS             numero totale di epoch (default 20)
#   LR                   learning rate SGD (default 0.05)
#   CHECKPOINT_INTERVAL  ogni quante epoch salvare (default 1)
#   GCS_BUCKET           bucket GCS (es. ias-luigi-asprino-bucket-eu)
#   GCS_MODEL_PATH       percorso GCS per il modello finale (default models/lezione8/)
# =============================================================================

import os
import signal
import sys
import time

import torch
import torch.nn as nn

from checkpoint import save_checkpoint, load_checkpoint
from dataset import build_dataloader
from model import MLP


# =============================================================================
# SEZIONE 1 — Configurazione
# =============================================================================

def get_config() -> dict:
    """
    Raccoglie tutti gli iperparametri dalle variabili d'ambiente.
    Centralizzare qui la configurazione rende lo script più facile da
    leggere e da sovrascrivere nel job_config.yaml.
    """
    use_gcs = os.environ.get("USE_SYNTHETIC", "1") != "1"
    bucket  = os.environ.get("GCS_BUCKET", "ias-luigi-asprino-bucket-eu")

    return {
        "batch_size":           int(os.environ.get("BATCH_SIZE",           "128")),
        "n_epochs":             int(os.environ.get("N_EPOCHS",             "20")),
        "lr":                 float(os.environ.get("LR",                   "0.05")),
        "checkpoint_interval":  int(os.environ.get("CHECKPOINT_INTERVAL",  "1")),
        "use_gcs":              use_gcs,
        "bucket":               bucket,
        "model_path":           os.environ.get("GCS_MODEL_PATH", "models/lezione8/"),
    }


# =============================================================================
# SEZIONE 2 — SIGTERM handler
# =============================================================================

# Stato globale del training — aggiornato dal loop, letto dall'handler.
# Usiamo variabili modulo (non classi) per semplicità: l'handler è una
# funzione globale e non può ricevere argomenti extra.
_training_state: dict = {}


def _sigterm_handler(signum, frame):
    """
    Handler per SIGTERM: salva il checkpoint e termina gracefully.

    GCP invia SIGTERM 30 secondi prima di terminare la VM Spot.
    Questo handler deve completare il salvataggio entro quel limite.

    Un checkpoint di ~400 KB (MLP) su GCS impiega <1 secondo.
    Per modelli più grandi (centinaia di MB) potrebbe essere necessario
    aumentare il numero di secondi di preavviso via il campo
    `scheduling.terminationTime` nel job_config.yaml.

    NOTA DIDATTICA:
    signal.signal() registra handler Python puri, che vengono eseguiti
    dal thread principale al termine dell'istruzione bytecode corrente.
    In un training loop PyTorch su CPU, questo avviene quasi subito.
    Su GPU, il loop potrebbe essere bloccato in una CUDA kernel call:
    in quel caso l'handler viene eseguito solo al termine del kernel.
    Per questo motivo, su GPU è preferibile usare un thread separato
    o un flag volatile controllato tra le iterazioni.
    """
    print("\n[SIGTERM] Segnale ricevuto! Salvataggio checkpoint di emergenza...")

    s = _training_state
    if s:
        save_checkpoint(
            model     = s["model"],
            optimizer = s["optimizer"],
            epoch     = s["epoch"],
            step      = s["step"],
            loss      = s["loss"],
            use_gcs   = s["use_gcs"],
            bucket    = s["bucket"],
        )
        print(f"[SIGTERM] Checkpoint salvato all'epoch {s['epoch']}. "
              f"Il job verrà riavviato da Vertex AI.")
    else:
        print("[SIGTERM] Training non ancora iniziato, nessun checkpoint da salvare.")

    sys.exit(0)


# =============================================================================
# SEZIONE 3 — Salvataggio del modello finale
# =============================================================================

def _save_final_model(model: MLP, world_size: int, cfg: dict) -> None:
    """
    Salva il modello finale al termine del training completo.
    Distinto dai checkpoint: non contiene lo stato dell'ottimizzatore.
    """
    if not cfg["use_gcs"]:
        path = f"/tmp/model_lezione8.pth"
        torch.save(model.state_dict(), path)
        print(f"[save] Modello finale salvato in {path}")
        return

    try:
        import io as _io
        from google.cloud import storage as gcs_storage

        buf = _io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        blob_name = f"{cfg['model_path']}model_final.pth"
        client = gcs_storage.Client()
        client.bucket(cfg["bucket"]).blob(blob_name).upload_from_file(
            buf, content_type="application/octet-stream"
        )
        print(f"[save] Modello finale salvato su gs://{cfg['bucket']}/{blob_name}")
    except Exception as e:
        print(f"[save] Errore nel salvataggio GCS: {e}")
        path = "/tmp/model_lezione8_fallback.pth"
        torch.save(model.state_dict(), path)
        print(f"[save] Fallback: salvato in {path}")


# =============================================================================
# SEZIONE 4 — Loop di training
# =============================================================================

def train():
    """
    Loop di training fault-tolerant con checkpoint automatico.

    Flusso:
        1. Carica il checkpoint se esiste (start_epoch > 0 → ripresa)
        2. Per ogni epoch:
           a. Training su tutti i batch
           b. Ogni CHECKPOINT_INTERVAL epoch → save_checkpoint()
        3. Al termine → salva il modello finale
    """
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[init] device={device}  n_epochs={cfg['n_epochs']}  "
          f"use_gcs={cfg['use_gcs']}")

    # ── Dataset e DataLoader ─────────────────────────────────────────────────
    loader, dataset = build_dataloader(
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    print(f"[data] {len(dataset)} campioni | batch_size={cfg['batch_size']} | "
          f"{len(loader)} batch per epoch")

    # ── Modello e ottimizzatore ──────────────────────────────────────────────
    model     = MLP(in_features=dataset.n_features, n_classes=dataset.n_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print(f"[model] {model.param_count():,} parametri | "
          f"checkpoint ~{model.param_bytes() / 1024:.1f} KB")

    # ── Ripristino dal checkpoint ────────────────────────────────────────────
    # load_checkpoint restituisce (0, 0) se non esiste alcun checkpoint.
    # In quel caso il training parte normalmente dall'epoch 0.
    start_epoch, start_step = load_checkpoint(
        model    = model,
        optimizer= optimizer,
        use_gcs  = cfg["use_gcs"],
        bucket   = cfg["bucket"],
        device   = device,
    )

    # ── Inizializzazione dello stato globale (per il SIGTERM handler) ────────
    # Aggiorniamo _training_state PRIMA di registrare il signal handler.
    _training_state.update({
        "model":     model,
        "optimizer": optimizer,
        "epoch":     start_epoch,
        "step":      start_step,
        "loss":      float("nan"),
        "use_gcs":   cfg["use_gcs"],
        "bucket":    cfg["bucket"],
    })

    # ── Registrazione del SIGTERM handler ────────────────────────────────────
    # Registriamo il handler DOPO aver inizializzato il training state.
    # Registrarlo prima significherebbe che il handler potrebbe essere
    # chiamato con _training_state vuoto.
    signal.signal(signal.SIGTERM, _sigterm_handler)
    print("[init] SIGTERM handler registrato")

    if start_epoch >= cfg["n_epochs"]:
        print(f"[init] Training già completato (epoch {start_epoch}/{cfg['n_epochs']}). "
              "Nessuna epoch da eseguire.")
        return

    # ── Loop principale ──────────────────────────────────────────────────────
    t_total_start = time.perf_counter()
    global_step   = start_step

    for epoch in range(start_epoch, cfg["n_epochs"]):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t_epoch_start = time.perf_counter()

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

        avg_loss  = total_loss / n_batches
        t_epoch   = time.perf_counter() - t_epoch_start

        print(f"Epoch {epoch+1:>3}/{cfg['n_epochs']} | "
              f"loss={avg_loss:.4f} | "
              f"step={global_step:>6} | "
              f"tempo={t_epoch:.2f}s")

        # ── Aggiorna lo stato globale (per SIGTERM handler) ──────────────────
        # Questo deve avvenire PRIMA del checkpoint per garantire che
        # il SIGTERM handler veda sempre uno stato coerente.
        _training_state["epoch"] = epoch
        _training_state["step"]  = global_step
        _training_state["loss"]  = avg_loss

        # ── Checkpoint periodico ─────────────────────────────────────────────
        if (epoch + 1) % cfg["checkpoint_interval"] == 0:
            save_checkpoint(
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                step      = global_step,
                loss      = avg_loss,
                use_gcs   = cfg["use_gcs"],
                bucket    = cfg["bucket"],
            )

    # ── Training completato ──────────────────────────────────────────────────
    t_total = time.perf_counter() - t_total_start
    print(f"\nTraining completato in {t_total:.1f}s ({cfg['n_epochs']} epoch totali)")

    _save_final_model(model, world_size=1, cfg=cfg)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    train()
