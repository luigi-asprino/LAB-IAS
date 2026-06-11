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
import torch.nn as nn
from google.cloud import storage as gcs
from model import SimpleNet
from dataset import load_from_gcs


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


# =============================================================================
# SEZIONE 2 — Inizializzazione del gruppo RPC
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
# SEZIONE 3 — Classe ParameterServer
# =============================================================================
#
# Questa classe viene istanziata solo sul processo PS (workerpool0-0).
# I worker non la istanziano localmente: ottengono un RRef (Remote Reference)
# all'istanza che vive sul PS e la chiamano tramite le funzioni standalone
# definite nella Sezione 4.
# =============================================================================

class ParameterServer:
    def __init__(self):
        # Il modello risiede interamente sul PS: è lui l'unico detentore
        # dei pesi "canonici" del sistema distribuito.
        self.model = SimpleNet()

        # L'optimizer aggiorna i pesi sul PS ogni volta che riceve gradienti.
        # SGD con lr=0.01 è scelto per semplicità didattica.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Lock necessario per gestire la concorrenza: due worker possono
        # chiamare update_params() contemporaneamente. Senza il lock,
        # due optimizer.step() simultanei corromperebbero i pesi in modo
        # silenzioso (nessun errore esplicito, solo training anomalo).
        self.lock = torch.multiprocessing.Lock()


# =============================================================================
# SEZIONE 4 — Funzioni standalone per RPC
# =============================================================================
#
# IMPORTANTE: PyTorch RPC non supporta i metodi di classe come target.
# Se si passa un metodo (es. ParameterServer.get_params) come callable
# a rpc_sync/rpc_async, PyTorch trasmette l'RRef stesso come primo
# argomento invece dell'istanza — causando AttributeError a runtime.
#
# La soluzione è usare funzioni standalone che:
#   1. Ricevono l'RRef come argomento esplicito
#   2. Chiamano ps_rref.local_value() per accedere all'oggetto reale
#      (local_value() funziona solo sul processo che possiede l'oggetto)
# =============================================================================

def get_params(ps_rref):
    """
    Eseguita SUL PS quando un worker chiama rpc_sync(get_params).

    ps_rref.local_value() restituisce l'istanza ParameterServer reale
    che vive su questo processo. Funziona solo se chiamata localmente
    (cioè se questo processo è effettivamente il PS).

    Restituisce una lista di tensori clonati: clone() è necessario
    perché i tensori originali potrebbero essere modificati dall'optimizer
    mentre il worker li sta leggendo.
    """
    ps = ps_rref.local_value()
    return [p.data.clone() for p in ps.model.parameters()]


def update_params(ps_rref, grads):
    """
    Eseguita SUL PS quando un worker chiama rpc_async(update_params).

    Riceve la lista di gradienti calcolati dal worker sul suo batch locale
    e li applica ai parametri del modello tramite l'optimizer.

    Il Lock garantisce che un solo worker alla volta esegua lo step,
    evitando race condition quando due worker inviano gradienti
    contemporaneamente (comportamento frequente nel training asincrono).
    """
    ps = ps_rref.local_value()
    with ps.lock:
        ps.optimizer.zero_grad()
        for p, g in zip(ps.model.parameters(), grads):
            # clone() sul gradiente ricevuto evita che modifiche
            # successive del worker (es. zero_grad implicito) alterino
            # il tensore mentre il PS lo sta usando.
            p.grad = g.clone()
        ps.optimizer.step()


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
    # Copia locale del modello: usata per il forward/backward pass.
    # I pesi vengono sovrascritti ad ogni batch con quelli scaricati dal PS.
    local_model = SimpleNet()
    criterion   = nn.CrossEntropyLoss()

    # Carica il dataset da GCS. GCS_BUCKET è letto dalla variabile d'ambiente
    # definita nel Dockerfile; può essere sovrascritta al lancio del job.
    dataset = load_from_gcs(
        bucket=os.environ.get("GCS_BUCKET", "ias-luigi-asprino-bucket"),
        path="datasets/mnist/train"
    )

    for epoch in range(5):
        for batch_x, batch_y in dataset:

            # ── PULL: scarica i pesi aggiornati dal PS ──────────────────
            # rpc_sync è BLOCCANTE: il worker aspetta che il PS risponda
            # prima di procedere. Necessario qui perché il forward pass
            # dipende dai parametri aggiornati.
            # ps_rref.owner() restituisce il WorkerInfo del processo PS,
            # usato come destinazione della chiamata RPC.
            params = rpc.rpc_sync(
                ps_rref.owner(),
                get_params,
                args=(ps_rref,)
            )
            # Sovrascrive i pesi locali con quelli ricevuti dal PS.
            # data.copy_() è un'operazione in-place che non spezza
            # il grafo computazionale di autograd.
            for p, new_val in zip(local_model.parameters(), params):
                p.data.copy_(new_val)

            # ── FORWARD + BACKWARD: eseguiti localmente ─────────────────
            # Il calcolo rimane sul worker: solo i gradienti vengono
            # trasmessi al PS, non i dati di input né i layer intermedi.
            out  = local_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()  # Calcola i gradienti per tutti i parametri

            # ── PUSH: invia i gradienti al PS ────────────────────────────
            # rpc_async è NON BLOCCANTE: il worker invia i gradienti e
            # prosegue immediatamente al batch successivo senza aspettare
            # la conferma dell'aggiornamento.
            # QUESTO È IL PUNTO ESATTO IN CUI NASCE L'ASINCRONICITÀ:
            # al prossimo batch il worker farà pull di pesi che potrebbero
            # già includere aggiornamenti da altri worker — stale gradient.
            grads = [p.grad.clone() for p in local_model.parameters()]
            rpc.rpc_async(
                ps_rref.owner(),
                update_params,
                args=(ps_rref, grads)
            )

            print(f"[worker-{task_index}] epoch={epoch} loss={loss.item():.4f}")

    # ── SALVATAGGIO MODELLO SU GCS ────────────────────────────────────────
    # Solo worker-0 salva per evitare scritture concorrenti sullo stesso blob.
    # NOTA: il filesystem /gcs/ NON è montato nei container Vertex AI.
    # Bisogna usare l'SDK google-cloud-storage per scrivere su GCS.
    if task_index == 0:
        bucket_name = os.environ.get("GCS_BUCKET", "ias-luigi-asprino-bucket")

        # Serializza lo state_dict (solo i pesi, non l'architettura)
        # in un buffer in memoria invece di un file temporaneo su disco.
        buf = io.BytesIO()
        torch.save(local_model.state_dict(), buf)
        buf.seek(0)  # Riporta il cursore all'inizio per la lettura

        # Carica il buffer direttamente su GCS
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob("models/lezione6/model.pth")
        blob.upload_from_file(buf, content_type="application/octet-stream")
        print(f"[worker-0] Modello salvato in gs://{bucket_name}/models/lezione6/model.pth")


# =============================================================================
# SEZIONE 6 — Entry point: divergenza PS / Worker
# =============================================================================
#
# Lo stesso container esegue codice diverso in base a is_ps.
# Questo è il pattern standard per il PS: un unico artefatto Docker
# che si comporta in modo diverso a seconda del ruolo assegnato
# dall'orchestratore a runtime.
# =============================================================================

if is_ps:
    print("[ps] ParameterServer in attesa di connessioni RPC...")

    # Crea l'istanza del PS e un RRef locale ad essa.
    # L'RRef verrà distribuito ai worker tramite rpc.remote().
    ps      = ParameterServer()
    ps_rref = rpc.RRef(ps)  # RRef locale: punta all'oggetto su questo processo

    # rpc.shutdown(graceful=True) blocca il PS qui finché tutti i worker
    # non chiamano rpc.shutdown(). Garantisce che nessun gradiente in volo
    # vada perso per terminazione prematura del PS.
    rpc.shutdown(graceful=True)
    print("[ps] Shutdown completato.")

else:
    print(f"[worker-{task_index}] Avvio training...")

    # rpc.remote() crea un RRef REMOTO all'oggetto ParameterServer sul PS.
    # A differenza di RRef() locale, questo NON trasferisce l'oggetto
    # sul worker: crea solo un riferimento opaco che punta al PS.
    # Ogni accesso successivo tramite questo RRef passa per la rete.
    ps_rref = rpc.remote("ps", ParameterServer)

    run_worker(ps_rref)

    # Segnala al PS che questo worker ha terminato.
    # Quando tutti i worker chiamano shutdown, il PS esce dal suo
    # rpc.shutdown(graceful=True) e termina a sua volta.
    rpc.shutdown(graceful=True)
    print(f"[worker-{task_index}] Shutdown completato.")