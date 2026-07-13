"""
Esecuzione della pipeline su GCP — Vertex AI Pipelines.

Riusa la STESSA definizione di pipeline (@dsl.pipeline, componenti,
ParallelFor, Collected) scritta in kfp_pipeline/pipeline.py: cambia
solo il backend di esecuzione, non una riga di logica del DAG.

    Locale (pipeline.py)             GCP (questo script)
    ---------------------------------------------------------------
    kfp.local.SubprocessRunner        aiplatform.PipelineJob
    Artefatti su filesystem locale     Artefatti su Google Cloud Storage
    Esecuzione sincrona in-process      Job asincrono monitorabile in
                                        console Vertex AI

Prerequisiti:
    pip install "kfp>=2.7" "google-cloud-aiplatform>=1.60"
    gcloud auth application-default login
    Un bucket GCS esistente per pipeline_root (es. gs://<bucket>/pipeline-root)
    API abilitate: aiplatform.googleapis.com, storage.googleapis.com

Esecuzione:
    python kfp_pipeline/run_on_vertex_ai.py \
        --project test-project-493915 \
        --location europe-west4 \
        --pipeline-root gs://ias-luigi-asprino-bucket-eu/pipeline-root \
        --seeds 1 2 3 4 \
        --epochs 3
"""

import argparse

from google.cloud import aiplatform
from kfp import compiler

# Riusa la stessa identica definizione di pipeline usata in locale.
# Nessuna riga di logica del DAG viene duplicata o riscritta qui.
from pipeline import training_pipeline

COMPILED_PIPELINE_PATH = "training_pipeline.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compila e sottomette la pipeline fan-out/fan-in su Vertex AI Pipelines."
    )
    parser.add_argument("--project", required=True, help="ID del progetto GCP (es. test-project-493915)")
    parser.add_argument("--location", default="europe-west4", help="Region Vertex AI (default: europe-west4)")
    parser.add_argument(
        "--pipeline-root",
        required=True,
        help="Bucket/prefisso GCS per gli artefatti della pipeline (es. gs://bucket/pipeline-root)",
    )
    parser.add_argument("--shard-path", default="./data/shard_0", help="Path passato al componente preprocess_shard")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4], help="Lista di seed per il fan-out")
    parser.add_argument("--epochs", type=int, default=3, help="Numero di epoche di training per branch")
    parser.add_argument(
        "--job-display-name",
        default="fan-out-fan-in-training-vertex-ai",
        help="Nome visualizzato del job nella console Vertex AI",
    )
    parser.add_argument(
        "--service-account",
        default=None,
        help="Service account da usare per l'esecuzione (opzionale, usa il default del progetto se omesso)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Sottomette il job senza attendere il completamento (non blocca il terminale)",
    )
    return parser.parse_args()


def compile_pipeline():
    """
    Compila @dsl.pipeline in IR YAML.
    Stesso identico passo eseguito implicitamente da kfp.local, ma qui
    esplicito perche' aiplatform.PipelineJob richiede il file compilato.
    """
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=COMPILED_PIPELINE_PATH,
    )
    print(f"Pipeline compilata in IR YAML: {COMPILED_PIPELINE_PATH}")


def submit_pipeline_job(args):
    """Sottomette il job a Vertex AI Pipelines usando il backend gestito da Google."""
    aiplatform.init(project=args.project, location=args.location)

    job = aiplatform.PipelineJob(
        display_name=args.job_display_name,
        template_path=COMPILED_PIPELINE_PATH,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "shard_path": args.shard_path,
            "seeds": args.seeds,
            "epochs": args.epochs,
        },
        enable_caching=False,  # disabilitato per la demo: ogni run riesegue tutto da zero
    )

    print(f"Sottomissione job '{args.job_display_name}' su Vertex AI Pipelines...")
    print(f"  project={args.project}  location={args.location}")
    print(f"  pipeline_root={args.pipeline_root}")

    job.submit(service_account=args.service_account)

    print(f"\nJob sottomesso. Segui l'esecuzione nella console:")
    print(f"  {job._dashboard_uri()}")

    if not args.no_wait:
        print("\nIn attesa del completamento (usa --no-wait per non bloccare)...")
        job.wait()
        print(f"Stato finale: {job.state}")


if __name__ == "__main__":
    args = parse_args()
    compile_pipeline()
    submit_pipeline_job(args)
