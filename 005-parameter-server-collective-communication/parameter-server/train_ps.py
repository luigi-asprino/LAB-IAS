# =============================================================================
# train_ps.py — Parameter Server Pattern con PyTorch RPC su Vertex AI
# =============================================================================
#
# Questo script implementa il pattern Parameter Server (PS) usando
# torch.distributed.rpc. Lo stesso container viene eseguito su tre VM:
#
#   workerpool0-0  →  Parameter Server (PS)
#     - detiene i pesi del modello e l'optimizer
#     - riceve gradienti dai worker, aggiorna i pesi
#     - non esegue il forward pass
#
#   workerpool1-0  →  Worker 0
#   workerpool1-1  →  Worker 1
#     - eseguono forward + backward su batch locali
#     - scaricano i pesi dal PS (pull, bloccante)
#     - inviano i gradienti al PS (push, non bloccante)
#
# Il ruolo di ogni replica è determinato a runtime leggendo CLUSTER_SPEC,
# una variabile d'ambiente JSON iniettata automaticamente da Vertex AI.
# =============================================================================

import os
import io
import json
import torch
import torch.distributed.rpc as rpc
import threading
import torch.nn as nn
from model import SimpleNet
from dataset import load_dataset, require_bucket


# =============================================================================
# SEZIONE 1 — Lettura della topologia del cluster
# =============================================================================
#
# Vertex AI inietta CLUSTER_SPEC come stringa JSON in ogni container.
# La struttura reale è:
#
#   {
#     "cluster": {
#       "workerpool0": ["<hostname-ps>:2222"],
#       "workerpool1": ["<hostname-w0>:2222", "<hostname-w1>:2222"]
#     },
#     "task": { "type": "workerpool0", "index": 0 }
#   }
#
# =============================================================================

# Parsing difensivo: "or \"{}\"" gestisce il caso in cui la variabile
# esiste nell'ambiente ma è una stringa vuota (comportamento Vertex AI).
cluster_spec = json.loads(os.environ.get("CLUSTER_SPEC", "{}") or "{}")

# Ruolo di questa replica: "workerpool0" (PS) o "workerpool1" (worker).
task_type  = cluster_spec.get("task", {}).get("type", "workerpool0")

# Indice numerico della replica all'interno del suo pool (0, 1, 2, ...).
# Usato per: assegnare il rank RPC ai worker, decidere quale worker salva il modello.
task_index = int(cluster_spec.get("task", {}).get("index", 0))

# Flag booleani per rendere il codice più leggibile
is_ps     = (task_type == "workerpool0")
is_worker = (task_type == "workerpool1")

# Indirizzo del PS: hostname:porta del primo (e unico) nodo workerpool0.
# Usato come rendezvous point per l'inizializzazione RPC di tutti i processi.
ps_address = cluster_spec["cluster"]["workerpool0"][0]

# world_size = numero totale di processi nel gruppo RPC distribuito.
# Tutti i processi (PS + worker) devono concordare su questo valore
# durante rpc.init_rpc(), altrimenti l'inizializzazione si blocca.
world_size = 1 + len(cluster_spec["cluster"]["workerpool1"])

print(f"[init] task_type={task_type} task_index={task_index} is_ps={is_ps}")
print(f"[init] ps_address={ps_address} world_size={world_size}")


# Configurazione letta dall'ambiente (vedi il passo 0 del README).
# Con USE_SYNTHETIC=1 non serve alcun bucket; altrimenti BUCKET è obbligatoria
# e il controllo avviene subito, prima di avviare il gruppo RPC.
USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC", "0") == "1"
if not USE_SYNTHETIC:
    require_bucket()
print(f"[config] ruolo={'ps' if is_ps else f'worker-{task_index}'} "
      f"world_size={world_size} USE_SYNTHETIC={int(USE_SYNTHETIC)} "
      f"BUCKET={os.environ.get('BUCKET', '-')} N_EPOCHS={os.environ.get('N_EPOCHS', '5')}")


# =============================================================================
# SEZIONE 2 — Classe ParameterServer
# =============================================================================
#
# Questa classe viene istanziata UNA SOLA VOLTA, sul processo PS
# (workerpool0-0). I worker non la istanziano: ottengono un RRef
# (Remote Reference) all'istanza che vive sul PS tramite get_ps_rref()
# e la usano attraverso le funzioni standalone della Sezione 4.
#
# ATTENZIONE (bug della versione precedente): se ogni worker chiama
# rpc.remote("ps", ParameterServer), sul PS nascono TANTE istanze quanti
# sono i worker. Ogni worker addestra allora il "suo" modello privato:
# non c'è nessun parametro condiviso, nessuno stale gradient, e il modello
# salvato dal PS (un'altra istanza ancora) resta con i pesi iniziali.
# =============================================================================

class ParameterServer:
    def __init__(self):
        # Il modello risiede interamente sul PS: è lui l'unico detentore
        # dei pesi "canonici" del sistema distribuito.
        self.model = SimpleNet()

        # L'optimizer aggiorna i pesi sul PS ogni volta che riceve gradienti.
        # SGD con lr=0.01 è scelto per semplicità didattica.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Lock necessario per gestire la concorrenza: il PS serve le chiamate
        # RPC con un pool di thread, quindi due worker possono eseguire
        # update_params() contemporaneamente. Senza il lock, due
        # optimizer.step() simultanei corromperebbero i pesi in modo
        # silenzioso. È un lock tra thread dello stesso processo:
        # threading.Lock è lo strumento giusto.
        self.lock = threading.Lock()

        # Versione dei parametri: cresce di 1 a ogni optimizer.step().
        # Serve a MISURARE la staleness: un gradiente calcolato sui pesi
        # alla versione v e applicato quando il PS è alla versione v' ha
        # staleness = v' - v (quanti aggiornamenti di altri worker si sono
        # inseriti nel frattempo). Con un solo worker sincrono vale 0.
        self.version = 0
        self.staleness_sum = 0
        self.staleness_max = 0
        self.updates_per_worker = {}


# Unica istanza del PS: creata PRIMA di init_rpc (Sezione 3), così è già
# pronta quando arriva la prima chiamata di un worker: init_rpc è una
# barriera, e subito dopo i worker chiamano get_ps_rref.
ps_instance = ParameterServer() if is_ps else None


# =============================================================================
# SEZIONE 3 — Inizializzazione del gruppo RPC
# =============================================================================
#
# rpc.init_rpc() registra questo processo nel gruppo distribuito.
# Ogni processo deve avere un nome univoco e un rank numerico univoco.
#
# Il PS prende rank=0 per convenzione e nome="ps".
# I worker prendono rank=task_index+1 (1, 2, ...) e nome="worker-N".
#
# init_method="tcp://<ps_address>" indica l'indirizzo del processo rank=0
# che funziona da punto di coordinamento iniziale: tutti gli altri processi
# si connettono qui per scambiarsi gli indirizzi e completare l'handshake.
#
# Tutti i processi devono chiamare rpc.init_rpc() prima che qualsiasi
# chiamata RPC possa avere luogo — init è una barriera implicita.
# =============================================================================

rpc.init_rpc(
    name       = "ps" if is_ps else f"worker-{task_index}",
    rank       = 0    if is_ps else task_index + 1,
    world_size = world_size,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        # Indirizzo TCP del processo rank=0 (il PS).
        # TensorPipe è il backend di trasporto di default in PyTorch >= 1.8;
        # gestisce automaticamente la selezione del canale ottimale
        # (shared memory per processi locali, TCP per processi remoti).
        init_method=f"tcp://{ps_address}"
    )
)


# =============================================================================
# SEZIONE 4 — Funzioni standalone per RPC
# =============================================================================
#
# IMPORTANTE: come target di rpc_sync/rpc_async usiamo funzioni standalone
# e non metodi della classe. Passando un metodo d'istanza
# (es. ParameterServer.get_params) con args=(ps_rref,), PyTorch passa
# l'RRef al posto di self, causando AttributeError a runtime
# ("PyRRef has no attribute 'model'").
#
# Le funzioni standalone:
#   1. Ricevono l'RRef come argomento esplicito
#   2. Chiamano ps_rref.local_value() per accedere all'oggetto reale
#      (local_value() funziona solo sul processo che possiede l'oggetto)
# =============================================================================

LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))


def get_ps_rref():
    """
    Eseguita SUL PS quando un worker chiama rpc_sync("ps", get_ps_rref).
    Restituisce un RRef all'unica istanza ParameterServer: tutti i worker
    ottengono così un riferimento allo STESSO oggetto.
    """
    return rpc.RRef(ps_instance)


def get_params(ps_rref):
    """
    Eseguita SUL PS quando un worker chiama rpc_sync(get_params).

    Restituisce (versione, parametri). I tensori sono clonati sotto lock:
    senza clone() il worker potrebbe ricevere pesi modificati a metà
    da un optimizer.step() concorrente.
    """
    ps = ps_rref.local_value()
    with ps.lock:
        return ps.version, [p.data.clone() for p in ps.model.parameters()]


def update_params(ps_rref, grads, worker_name, read_version):
    """
    Eseguita SUL PS quando un worker chiama rpc_async(update_params).

    Applica i gradienti calcolati dal worker sul suo batch locale.
    read_version è la versione dei pesi su cui il worker ha calcolato
    i gradienti: la differenza con la versione corrente è la staleness.
    """
    ps = ps_rref.local_value()
    with ps.lock:
        staleness = ps.version - read_version
        ps.optimizer.zero_grad()
        for p, g in zip(ps.model.parameters(), grads):
            p.grad = g
        ps.optimizer.step()
        ps.version += 1

        # Statistiche per i log
        ps.staleness_sum += staleness
        ps.staleness_max = max(ps.staleness_max, staleness)
        n = ps.updates_per_worker.get(worker_name, 0) + 1
        ps.updates_per_worker[worker_name] = n
        if ps.version % LOG_EVERY == 0:
            print(f"[ps] version={ps.version} da {worker_name} "
                  f"(update n.{n} di questo worker) staleness={staleness}")


# =============================================================================
# SEZIONE 5 — Funzione run_worker: loop di training
# =============================================================================

def run_worker(ps_rref):
    """
    Eseguita su ogni processo worker (workerpool1-N).

    ps_rref è un RRef: un riferimento remoto all'oggetto ParameterServer
    che vive sul PS. Il worker non ha mai una copia locale del PS;
    ogni accesso ai pesi o aggiornamento passa attraverso la rete via RPC.
    """
    worker_name = f"worker-{task_index}"
    num_workers = world_size - 1
    n_epochs    = int(os.environ.get("N_EPOCHS", "5"))

    # Copia locale del modello: usata per il forward/backward pass.
    # I pesi vengono sovrascritti ad ogni batch con quelli scaricati dal PS.
    local_model = SimpleNet()

    # SimpleNet restituisce log-probabilità (log_softmax): la loss corretta
    # è NLLLoss. CrossEntropyLoss applicherebbe log_softmax una seconda volta.
    criterion = nn.NLLLoss()

    # Ogni worker legge solo il SUO shard del dataset (campioni con
    # indice ≡ task_index mod num_workers): data parallelism vero,
    # come farà DistributedSampler nella seconda parte della lezione.
    dataset = load_dataset(num_shards=num_workers, shard_index=task_index)

    pending = None
    step = 0
    for epoch in range(n_epochs):
        for batch_x, batch_y in dataset:

            # ── PULL: scarica i pesi aggiornati dal PS ──────────────────
            # rpc_sync è BLOCCANTE: il forward pass ha bisogno dei pesi.
            # ps_rref.owner() è il WorkerInfo del processo PS.
            version, params = rpc.rpc_sync(
                ps_rref.owner(),
                get_params,
                args=(ps_rref,)
            )
            with torch.no_grad():
                for p, new_val in zip(local_model.parameters(), params):
                    p.copy_(new_val)

            # ── FORWARD + BACKWARD: eseguiti localmente ─────────────────
            local_model.zero_grad()
            out  = local_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()

            # ── PUSH: invia i gradienti al PS ────────────────────────────
            # rpc_async è NON BLOCCANTE: il worker prosegue subito al batch
            # successivo. QUI NASCE L'ASINCRONICITÀ: mentre questo gradiente
            # viaggia, altri worker possono aggiornare i pesi sul PS.
            grads = [p.grad.clone() for p in local_model.parameters()]
            pending = rpc.rpc_async(
                ps_rref.owner(),
                update_params,
                args=(ps_rref, grads, worker_name, version)
            )

            step += 1
            if step % LOG_EVERY == 0:
                print(f"[{worker_name}] epoch={epoch} step={step} "
                      f"version_letta={version} loss={loss.item():.4f}")

    # Attende che l'ultimo gradiente sia stato applicato prima di uscire.
    if pending is not None:
        pending.wait()
    print(f"[{worker_name}] Training terminato dopo {step} step.")


# =============================================================================
# SEZIONE 6 — Entry point: divergenza PS / Worker
# =============================================================================
#
# Lo stesso container esegue codice diverso in base a is_ps.
# Questo è il pattern standard per il PS: un unico artefatto Docker
# che si comporta in modo diverso a seconda del ruolo assegnato
# dall'orchestratore a runtime.
# =============================================================================

def save_model(state_dict):
    """Salva su GCS (Vertex AI) oppure in /tmp (USE_SYNTHETIC=1, test locale)."""
    if os.environ.get("USE_SYNTHETIC", "0") == "1":
        torch.save(state_dict, "/tmp/model_ps.pth")
        print("[ps] Modello salvato in /tmp/model_ps.pth")
        return
    from google.cloud import storage as gcs
    bucket_name = require_bucket()
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    blob_name   = os.environ.get("MODEL_PATH", "lab5/ps/model.pth")
    gcs.Client().bucket(bucket_name).blob(blob_name).upload_from_file(
        buf, content_type="application/octet-stream")
    print(f"[ps] Modello salvato su gs://{bucket_name}/{blob_name}")


if is_ps:
    print("[ps] ParameterServer in attesa di connessioni RPC...")

    # rpc.shutdown(graceful=True) blocca il PS qui finché tutti i worker
    # non chiamano rpc.shutdown(). Nel frattempo il PS serve le chiamate
    # get_ps_rref / get_params / update_params nei suoi thread RPC.
    rpc.shutdown(graceful=True)
    print("[ps] Shutdown completato.")

    ps = ps_instance
    print(f"[ps] Aggiornamenti totali: {ps.version} | per worker: {ps.updates_per_worker}")
    if ps.version:
        print(f"[ps] Staleness media={ps.staleness_sum / ps.version:.2f} "
              f"massima={ps.staleness_max}")
    save_model(ps.model.state_dict())

else:
    print(f"[worker-{task_index}] Avvio training...")

    # Chiede al PS un RRef all'unica istanza ParameterServer.
    # L'oggetto NON viene trasferito sul worker: l'RRef è un riferimento
    # opaco, e ogni accesso successivo passa per la rete.
    ps_rref = rpc.rpc_sync("ps", get_ps_rref)

    run_worker(ps_rref)

    # Segnala al PS che questo worker ha terminato.
    rpc.shutdown(graceful=True)
    print(f"[worker-{task_index}] Shutdown completato.")
