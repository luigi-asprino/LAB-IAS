# =============================================================================
# checkpoint.py — Logica di checkpoint per la Lezione 8
# =============================================================================
#
# Questo modulo isola tutta la logica di salvataggio e ripristino del
# checkpoint, rendendola riusabile e testabile indipendentemente dal loop
# di training. È il componente didattico centrale della lezione.
#
# Struttura del checkpoint:
#   {
#       "epoch":     int,               epoch appena completata (0-based)
#       "step":      int,               step globale (cumulativo tra epoch)
#       "model":     state_dict,        pesi del modello
#       "optimizer": state_dict,        stato dell'ottimizzatore (momentum, ecc.)
#       "loss":      float,             loss media dell'ultima epoch
#   }
#
# Percorso GCS:
#   gs://<BUCKET>/checkpoints/lezione8/checkpoint_epoch_<N>.pt
#   gs://<BUCKET>/checkpoints/lezione8/checkpoint_latest.pt   ← sempre l'ultimo
#
# Percorso locale (fallback e test):
#   /tmp/checkpoints/lezione8/checkpoint_epoch_<N>.pt
#   /tmp/checkpoints/lezione8/checkpoint_latest.pt
#
# Strategia di salvataggio:
#   - Scrive prima il file specifico per epoch (non sovrascrive mai)
#   - Poi sovrascrive checkpoint_latest.pt per indicare il punto di ripresa
#   - Al riavvio si carica SOLO checkpoint_latest.pt per semplicità
#
# Perché salvare anche per epoch e non solo "latest"?
#   - Rollback manuale in caso di training instabile (loss divergente)
#   - Ispezione dello stato del modello a diversi punti del training
#   - Analisi della curva di loss post-hoc
# =============================================================================

import io
import os
import tempfile

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

LOCAL_CKPT_DIR = "/tmp/checkpoints/lezione8"
GCS_CKPT_PATH  = "checkpoints/lezione8"   # prefisso relativo al bucket


# ---------------------------------------------------------------------------
# Funzioni di supporto GCS
# ---------------------------------------------------------------------------

def _gcs_upload(local_path: str, bucket_name: str, blob_name: str) -> None:
    """Carica un file locale su GCS."""
    from google.cloud import storage as gcs_storage
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(local_path)


def _gcs_download(bucket_name: str, blob_name: str, local_path: str) -> bool:
    """
    Scarica un file da GCS. Restituisce True se il file esiste, False altrimenti.
    Non solleva eccezione se il blob non esiste (primo avvio).
    """
    from google.cloud import storage as gcs_storage
    from google.cloud.exceptions import NotFound

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    try:
        blob.download_to_filename(local_path)
        return True
    except NotFound:
        return False


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    step:      int,
    loss:      float,
    use_gcs:   bool = False,
    bucket:    str  = "",
) -> None:
    """
    Salva il checkpoint del modello.

    Scrive due file:
      - checkpoint_epoch_<epoch>.pt  (immutabile, per audit/rollback)
      - checkpoint_latest.pt         (sovrascritto ad ogni chiamata)

    Con use_gcs=True entrambi vengono caricati su GCS DOPO il salvataggio
    locale in /tmp (scrittura locale sempre più veloce di GCS).

    Parametri
    ----------
    model     : modello PyTorch (deve esporre .state_dict())
    optimizer : ottimizzatore SGD/Adam (necessario per ripristinare momentum)
    epoch     : indice dell'epoch appena completata (0-based)
    step      : contatore globale degli step (cumulativo)
    loss      : loss media dell'ultima epoch (per logging)
    use_gcs   : se True, carica i checkpoint su GCS dopo il salvataggio locale
    bucket    : nome del bucket GCS (obbligatorio se use_gcs=True)
    """
    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

    state = {
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss":      loss,
    }

    # ── Salvataggio locale ──────────────────────────────────────────────────
    epoch_path  = os.path.join(LOCAL_CKPT_DIR, f"checkpoint_epoch_{epoch}.pt")
    latest_path = os.path.join(LOCAL_CKPT_DIR, "checkpoint_latest.pt")

    torch.save(state, epoch_path)
    torch.save(state, latest_path)

    size_kb = os.path.getsize(latest_path) / 1024
    print(f"[ckpt] Checkpoint salvato su {latest_path}: epoch={epoch}, step={step}, "
          f"loss={loss:.4f}, size={size_kb:.1f} KB")

    # ── Upload su GCS ────────────────────────────────────────────────────────
    if use_gcs and bucket:
        epoch_blob  = f"{GCS_CKPT_PATH}/checkpoint_epoch_{epoch}.pt"
        latest_blob = f"{GCS_CKPT_PATH}/checkpoint_latest.pt"

        _gcs_upload(epoch_path,  bucket, epoch_blob)
        _gcs_upload(latest_path, bucket, latest_blob)

        print(f"[ckpt] Caricato su gs://{bucket}/{latest_blob}")


def load_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    use_gcs:   bool = False,
    bucket:    str  = "",
    device:    torch.device = torch.device("cpu"),
) -> tuple[int, int]:
    """
    Carica il checkpoint più recente se esiste.

    Restituisce (start_epoch, start_step) per riprendere il training
    dall'epoch successiva all'ultima completata.

    Se il checkpoint non esiste (primo avvio), restituisce (0, 0).

    Con use_gcs=True, scarica prima il file da GCS in /tmp e poi lo carica.
    map_location=device è necessario perché il checkpoint potrebbe essere
    stato salvato su una GPU diversa (o su CPU) rispetto al nodo corrente.

    Parametri
    ----------
    model     : modello PyTorch (i pesi verranno sovrascritti)
    optimizer : ottimizzatore (lo stato verrà sovrascritto)
    use_gcs   : se True, cerca il checkpoint su GCS prima che in locale
    bucket    : nome del bucket GCS
    device    : dispositivo su cui caricare i tensori

    Returns
    -------
    (start_epoch, start_step) — da cui riprendere il training
    """
    latest_path = os.path.join(LOCAL_CKPT_DIR, "checkpoint_latest.pt")
    found = False

    if use_gcs and bucket:
        # Tenta di scaricare da GCS
        os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
        latest_blob = f"{GCS_CKPT_PATH}/checkpoint_latest.pt"
        found = _gcs_download(bucket, latest_blob, latest_path)
        if found:
            print(f"[ckpt] Checkpoint scaricato da gs://{bucket}/{latest_blob}")
        else:
            print("[ckpt] Nessun checkpoint su GCS — primo avvio")
    else:
        found = os.path.exists(latest_path)

    if not found:
        print("[ckpt] Nessun checkpoint trovato — training da zero")
        return 0, 0

    # ── Caricamento ──────────────────────────────────────────────────────────
    # weights_only=True è raccomandato da PyTorch >= 2.0 per sicurezza:
    # evita l'esecuzione di codice arbitrario nel file .pt (pickle).
    # Per i nostri checkpoint che contengono solo tensori è sempre sicuro.
    try:
        state = torch.load(latest_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[ckpt] Errore nel caricamento del checkpoint: {e}")
        print("[ckpt] Ripartenza da zero")
        return 0, 0

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    start_epoch = state["epoch"] + 1   # riprendiamo dall'epoch SUCCESSIVA
    start_step  = state["step"]
    prev_loss   = state.get("loss", float("nan"))

    print(f"[ckpt] Ripresa da epoch={start_epoch}, step={start_step}, "
          f"loss_precedente={prev_loss:.4f}")

    return start_epoch, start_step
