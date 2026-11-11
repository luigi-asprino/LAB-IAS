"""
Pipeline KFP — Fan-out / Fan-in, eseguita in locale.

Kubeflow Pipelines SDK e' la STESSA API usata da Vertex AI Pipelines:
cambia solo il runner di esecuzione.

    kfp.local.SubprocessRunner()   -> ogni componente gira come subprocess
                                       sulla macchina locale (usato oggi)
    aiplatform.PipelineJob(...)    -> stesso codice, eseguito su Vertex AI
                                       Pipelines (lezione futura in cloud)

Struttura del DAG:

    preprocess_shard (Sez. 2)          <- eseguito UNA sola volta
            |                             produce UN solo dataset
       ParallelFor(seed in [1,2,3,4])   <- FAN-OUT
            |                             lo STESSO dataset viene passato
      train_on_shard(seed)  x4             a tutti e 4 i branch, che
            |                             differiscono solo per il seed
       collect_and_select_best          <- FAN-IN
            |
        best_model.json

Nota didattica: questo DAG rappresenta lo scenario "un solo dataset,
training ripetuto con seed diversi" (utile per misurare la varianza
dovuta all'inizializzazione random / model selection tra run).
Non fa fan-out sui DATI: preprocess_shard produce un unico
output_dataset, il cui .path viene condiviso da tutti i branch del
ParallelFor. Per un fan-out anche sul preprocessing (shard di dati
realmente distinti) andrebbe spostato dsl.ParallelFor sopra la chiamata
a preprocess_shard.

Esecuzione:
    pip install "kfp>=2.7"
    python kfp_pipeline/pipeline.py
"""

import os
from typing import List, NamedTuple

import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

# NOTA: l'inizializzazione del runner locale (kfp.local.init) e' stata
# spostata nel blocco `if __name__ == "__main__"` in fondo al file.
# Questo rende il modulo importabile senza side-effect: lo script
# run_on_vertex_ai.py importa training_pipeline da qui senza attivare
# per sbaglio il runner locale.


# ---------------------------------------------------------------------------
# Componente 1 — preprocessing di uno shard di dati
# ---------------------------------------------------------------------------
@dsl.component(base_image="python:3.11-slim")
def preprocess_shard(shard_path: str, output_dataset: Output[Dataset]):
    """
    Simula il preprocessing di uno shard locale (in produzione: uno shard GCS).

    Eseguito UNA sola volta nel DAG (non e' dentro il ParallelFor): produce
    un unico output_dataset, il cui .path verra' condiviso da TUTTI i branch
    di training in Sez. 2-3 (che differiscono solo per il seed, non per i
    dati). shard_path qui serve solo a creare la cartella di lavoro locale;
    i dati stessi sono generati sinteticamente con torch.randn/randint.

    NOTA: nessun packages_to_install qui. kfp.local.SubprocessRunner esegue
    il codice del componente nell'ambiente Python corrente (l'ambiente conda
    della lezione, dove torch e' gia' installato da requirements.txt), non
    in un vero container Docker: base_image viene ignorata dal runner
    locale ed e' rilevante solo su Vertex AI/Kubernetes. Specificare
    packages_to_install=["torch==..."] qui farebbe eseguire un pip install
    per OGNI branch del ParallelFor, e kfp.local li serializza di default
    (max_concurrent_pip_installs=1, per evitare race condition) — questo
    nasconderebbe il vero parallelismo del fan-out dietro un collo di
    bottiglia di installazioni pip in coda.
    """
    import torch

    x = torch.randn(256, 20)
    y = torch.randint(0, 2, (256,))
    # output_dataset.path e' un path assegnato automaticamente da KFP:
    # qui scriviamo il file fisico che verra' poi letto da ogni branch
    # del ParallelFor tramite lo stesso Input[Dataset].path
    torch.save({"x": x, "y": y}, output_dataset.path)
    print(f"Dataset unico preprocessato e salvato in {output_dataset.path}")
    print("Questo stesso file verra' condiviso da tutti i branch del ParallelFor")


# ---------------------------------------------------------------------------
# Componente 2 — training PyTorch con un dato seed (eseguito in FAN-OUT)
# ---------------------------------------------------------------------------
@dsl.component(base_image="python:3.11-slim")
def train_on_shard(
    dataset: Input[Dataset],
    seed: int,
    epochs: int,
    model_out: Output[Model],
    metrics_out: Output[Metrics],
):
    """
    Ogni branch del ParallelFor esegue questo componente con un seed diverso.

    NOTA: nessun packages_to_install (vedi commento in preprocess_shard):
    l'assenza di installazioni pip per-componente permette ai 4 branch di
    girare realmente in parallelo su kfp.local, senza il collo di
    bottiglia della serializzazione dei pip install.
    """
    import torch

    torch.manual_seed(seed)
    data = torch.load(dataset.path)
    x, y = data["x"], data["y"]

    model = torch.nn.Sequential(
        torch.nn.Linear(20, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = torch.argmax(model(x), dim=1)
        accuracy = (preds == y).float().mean().item()

    torch.save(model.state_dict(), model_out.path)
    metrics_out.log_metric("seed", seed)
    metrics_out.log_metric("accuracy", accuracy)
    metrics_out.log_metric("final_loss", loss.item())
    print(f"[seed={seed}] accuracy={accuracy:.4f} loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# Componente 3 — FAN-IN: aggrega i risultati di tutti i branch paralleli
# ---------------------------------------------------------------------------
@dsl.component(base_image="python:3.11-slim")
def collect_and_select_best(
    all_metrics: Input[List[Metrics]],
    best_model_report: Output[Dataset],
):
    """
    Riceve la LISTA di output prodotti da tutti i branch del ParallelFor
    (uno per ogni seed) e seleziona il modello con accuracy migliore.
    Questo e' il punto di FAN-IN del DAG.
    """
    import json

    risultati = []
    for m in all_metrics:
        risultati.append(
            {
                "seed": m.metadata["seed"],
                "accuracy": m.metadata["accuracy"],
                "final_loss": m.metadata["final_loss"],
            }
        )

    migliore = max(risultati, key=lambda r: r["accuracy"])
    accuracy_media = sum(r["accuracy"] for r in risultati) / len(risultati)

    report = {
        "risultati_per_seed": risultati,
        "accuracy_media": accuracy_media,
        "modello_migliore": migliore,
    }

    with open(best_model_report.path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Accuracy media tra {len(risultati)} run: {accuracy_media:.4f}")
    print(f"Modello migliore: seed={migliore['seed']} accuracy={migliore['accuracy']:.4f}")


# ---------------------------------------------------------------------------
# Definizione del DAG: fan-out con dsl.ParallelFor + fan-in
# ---------------------------------------------------------------------------
@dsl.pipeline(name="fan-out-fan-in-training-locale")
def training_pipeline(
    shard_path: str = "./data/shard_0",
    seeds: List[int] = [1, 2, 3, 4],
    epochs: int = 3,
):
    preprocess_task = preprocess_shard(shard_path=shard_path)

    # --- FAN-OUT ---
    # ParallelFor lancia un'istanza di train_on_shard per ogni seed.
    # In locale: subprocess paralleli. Su Vertex AI: job paralleli gestiti
    # dal backend, con lo stesso identico codice del DAG.
    #
    # NOTA: parallelism=4 e' il valore corretto/dichiarativo del DAG e
    # viene rispettato interamente su Vertex AI (4 esecuzioni realmente
    # simultanee). Il runner locale (kfp.local.SubprocessRunner), pero',
    # limita SEMPRE la concorrenza effettiva a un massimo di 2 task
    # paralleli, indipendentemente dal valore di parallelism specificato
    # qui (e' un limite fisso, hardcoded nell'SDK, per evitare esplosione
    # di thread con ParallelFor annidati). In locale si vedranno quindi
    # al massimo 2 branch in esecuzione contemporanea, mai tutti e 4
    # insieme — non e' un errore del nostro codice.
    with dsl.ParallelFor(seeds, parallelism=4) as seed:
        train_task = train_on_shard(
            dataset=preprocess_task.outputs["output_dataset"],
            seed=seed,
            epochs=epochs,
        )

    # --- FAN-IN ---
    # dsl.Collected raccoglie in una lista gli output di TUTTI i branch
    # del ParallelFor, permettendo al componente successivo di aggregarli.
    collect_and_select_best(
        all_metrics=dsl.Collected(train_task.outputs["metrics_out"]),
    )


# ---------------------------------------------------------------------------
# Esecuzione locale del DAG
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import kfp.local

    # Inizializza il runner locale SOLO quando lo script viene eseguito
    # direttamente (non quando viene importato, es. da run_on_vertex_ai.py)
    kfp.local.init(runner=kfp.local.SubprocessRunner(use_venv=False))

    pipeline_task = training_pipeline(
        shard_path="./data/shard_0",
        seeds=[1, 2, 3, 4],
        epochs=3,
    )

    print("\nPipeline completata. Output del componente fan-in:")
    print(pipeline_task.outputs)

