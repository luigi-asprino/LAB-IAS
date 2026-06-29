#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build e deploy degli shard su Cloud Run
# =============================================================================
#
# Questo script esegue l'intera procedura di deployment:
#   1. Copia i file Python dalla directory local/ (DRY: un solo sorgente)
#   2. Build dell'immagine Docker
#   3. Push su Artifact Registry
#   4. Deploy degli shard su Cloud Run in ordine inverso (dal più alto al più basso)
#   5. Stampa gli URL di tutti gli shard e il comando di test
#
# Prerequisiti:
#   gcloud auth login
#   gcloud config set project test-project-493915
#   gcloud auth configure-docker europe-west4-docker.pkg.dev
#
# Uso:
#   ./deploy.sh           # 2 shard (default)
#   N_SHARDS=4 ./deploy.sh
#   ./deploy.sh --delete  # elimina tutti i service Cloud Run
# =============================================================================

set -euo pipefail

# ── Configurazione GCP ───────────────────────────────────────────────────────
PROJECT="test-project-493915"
REGION="europe-west4"
REPO="ias-repo"
N_SHARDS="${N_SHARDS:-2}"
SERVICE_PREFIX="lab09-shard"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/lab09-sharded:v1"

# ── Gestione --delete ────────────────────────────────────────────────────────
if [[ "${1:-}" == "--delete" ]]; then
    echo "=== Eliminazione dei service Cloud Run ==="
    for (( sid=0; sid<N_SHARDS; sid++ )); do
        SERVICE="${SERVICE_PREFIX}-${sid}"
        echo "  Eliminazione ${SERVICE}..."
        gcloud run services delete "$SERVICE" \
            --region="$REGION" \
            --project="$PROJECT" \
            --quiet || echo "  (non trovato, skip)"
    done
    echo "=== Cleanup completato ==="
    exit 0
fi

echo "=== Deploy Sharded Service Pattern su Cloud Run ==="
echo "    Progetto:  $PROJECT"
echo "    Regione:   $REGION"
echo "    N_SHARDS:  $N_SHARDS"
echo "    Immagine:  $IMAGE"
echo ""

# ── Step 1: Copia dei sorgenti Python nel contesto Docker ────────────────────
# Il Dockerfile si trova in gcp/, ma i file Python sono in local/.
# Li copiamo temporaneamente in gcp/ per il build context.
echo "[1/4] Copia sorgenti Python da ../local/ ..."
cp ../local/model.py        ./model.py
cp ../local/shard_server.py ./shard_server.py
cp ../local/requirements.txt ./requirements.txt
echo "      OK"

# ── Step 2: Build dell'immagine ──────────────────────────────────────────────
echo ""
echo "[2/4] Build immagine Docker..."
echo "      $IMAGE"
docker build -t "$IMAGE" .
echo "      Build completato"

# ── Cleanup dei sorgenti copiati ─────────────────────────────────────────────
rm -f model.py shard_server.py requirements.txt

# ── Step 3: Push su Artifact Registry ───────────────────────────────────────
echo ""
echo "[3/4] Push su Artifact Registry..."
docker push "$IMAGE"
echo "      Push completato"

# ── Step 4: Deploy degli shard in ordine inverso ────────────────────────────
echo ""
echo "[4/4] Deploy degli shard..."
echo ""

# Array degli URL per stampa finale
declare -A SHARD_URLS

NEXT_URL=""

for (( sid=N_SHARDS-1; sid>=0; sid-- )); do
    SERVICE="${SERVICE_PREFIX}-${sid}"
    echo "  → Deploy shard ${sid} (${SERVICE})..."

    # Variabili d'ambiente per questo shard
    ENV_VARS="SHARD_ID=${sid},N_SHARDS=${N_SHARDS}"
    if [[ -n "$NEXT_URL" ]]; then
        ENV_VARS="${ENV_VARS},NEXT_SHARD_URL=${NEXT_URL}"
    fi

    # Deploy su Cloud Run
    # --no-allow-unauthenticated in produzione: solo shard 0 è pubblico,
    # gli shard interni vengono chiamati con credenziali service account.
    # Per semplicità didattica usiamo --allow-unauthenticated su tutti.
    gcloud run deploy "$SERVICE" \
        --image="$IMAGE" \
        --region="$REGION" \
        --project="$PROJECT" \
        --set-env-vars="$ENV_VARS" \
        --allow-unauthenticated \
        --memory=512Mi \
        --cpu=1 \
        --min-instances=1 \
        --max-instances=3 \
        --timeout=60s \
        --quiet

    # Recupera l'URL del service appena deployato
    SHARD_URL=$(gcloud run services describe "$SERVICE" \
        --region="$REGION" \
        --project="$PROJECT" \
        --format="value(status.url)")

    SHARD_URLS[$sid]="$SHARD_URL"
    echo "    URL shard ${sid}: $SHARD_URL"

    # Il prossimo shard (indice più basso) punterà a questo
    NEXT_URL="$SHARD_URL"
done

# ── Riepilogo ────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Deploy completato — ${N_SHARDS} shard attivi"
echo ""
echo "  Shard e URL:"
for (( sid=0; sid<N_SHARDS; sid++ )); do
    echo "    Shard ${sid}: ${SHARD_URLS[$sid]}"
done
echo ""
echo "  Punto di ingresso (shard 0):"
echo "    ${SHARD_URLS[0]}"
echo ""
echo "  Test rapido:"
ENTRY="${SHARD_URLS[0]}"
echo "    curl -X POST ${ENTRY}/forward \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"token_ids\": [[101, 2054, 2003, 3435, 102]]}'"
echo ""
echo "  Da client.py:"
echo "    ROUTER_URL=${ENTRY} python3 ../local/client.py --mode bench --n 20"
echo "═══════════════════════════════════════════════════"
