#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

##############################################################################
# Unified Deployment Script with Auto-Cleanup
##############################################################################

# Configuration
ECR_REGION="us-west-2"
MAX_IMAGES_TO_KEEP=5
DRY_RUN_CLEANUP="${DRY_RUN_CLEANUP:-false}"
CLEANUP_AFTER_PUSH="${CLEANUP_AFTER_PUSH:-false}" # Auto-cleanup local images after successful push
SKIP_PUSH="${SKIP_PUSH:-false}" # Skip pushing to ECR (for local testing)
SKIP_DEPLOY="${SKIP_DEPLOY:-false}" # Skip Kubernetes deployment
PARALLEL_BUILDS="${PARALLEL_BUILDS:-true}" # Build service images in parallel

# Service definitions (single source of truth): name:dockerfile
SERVICES=(
    "floods:floods/Dockerfile"
    "burn_scars:burn_scars/Dockerfile"
    "crop_classification:crop_classification/Dockerfile"
    "surya_rollout:surya/Dockerfile"
)

# Map service names to Terraform variable names (name:tf_var)
SERVICE_TF_VARS=(
    "floods:TF_VAR_floods_app_image_url"
    "burn_scars:TF_VAR_burnScar_app_image_url"
    "crop_classification:TF_VAR_crop_app_image_url"
    "surya_rollout:TF_VAR_surya_rollout_image_url"
)

# Lookup helper: get_tf_var <service_name> prints the matching TF var (or empty)
get_tf_var() {
    local name=$1
    for entry in "${SERVICE_TF_VARS[@]}"; do
        if [[ "${entry%%:*}" == "$name" ]]; then
            echo "${entry#*:}"
            return
        fi
    done
}

# Tracking for cleanup
PUSHED_IMAGES=()

# Ensure required environment variables are set (only if we aren't doing a local-only run)
if [[ "$SKIP_PUSH" != "true" || "$SKIP_DEPLOY" != "true" ]]; then
    if [[ -z "${ECR_URL:-}" ]]; then
        echo "Error: ECR_URL environment variable must be set (e.g. 123456789.dkr.ecr.us-west-2.amazonaws.com)" >&2
        exit 1
    fi
    if [[ -z "${INGRESS_HOST:-}" ]]; then
        echo "Error: INGRESS_HOST environment variables must be set" >&2
        exit 1
    fi
fi

export TF_VAR_fm_services_release_version=$(git describe --tags --exact-match 2>/dev/null || git branch --show-current || git rev-parse --short HEAD)

echo "=========================================" >&2
echo "Docker Build & Deploy" >&2
echo "=========================================" >&2
echo "ECR URL: ${ECR_URL:-not set}" >&2
echo "Region: $ECR_REGION" >&2
echo "Keep Images: $MAX_IMAGES_TO_KEEP per repo" >&2
echo "=========================================" >&2

##############################################################################
# Helper Functions
##############################################################################

cleanup_on_exit() {
    echo "Pruning dangling images and old build cache..." >&2
    docker image prune -f >/dev/null 2>&1 || true
    docker builder prune -f --filter "until=24h" >/dev/null 2>&1 || true
}
trap cleanup_on_exit EXIT

check_disk_space() {
    local threshold_gb=${1:-20}
    local available_kb
    available_kb=$(df -k . | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    if [[ $available_gb -lt $threshold_gb ]]; then
        echo "WARNING: Only ${available_gb}GB disk space available (threshold: ${threshold_gb}GB)" >&2
    else
        echo "Disk space check: ${available_gb}GB available" >&2
    fi
}

# Cleanup old ECR images
cleanup_ecr_repo() {
    local repo_name=$1
    local keep_count=$2

    echo "Cleaning up $repo_name (keeping $keep_count newest)..." >&2

    # Check if repo exists
    if ! aws ecr describe-repositories --repository-names "$repo_name" --region "$ECR_REGION" &>/dev/null; then
        echo "  Repo doesn't exist yet, skipping" >&2
        return 0
    fi

    # Get total count
    local total
    total=$(aws ecr describe-images \
        --repository-name "$repo_name" \
        --region "$ECR_REGION" \
        --query 'length(imageDetails)' \
        --output text 2>/dev/null || echo "0")

    if [[ $total -le $keep_count ]]; then
        echo "  Only $total images, no cleanup needed" >&2
        return 0
    fi

    local to_delete=$((total - keep_count))
    echo "  Deleting $to_delete old images..." >&2

    if [[ "$DRY_RUN_CLEANUP" == "true" ]]; then
        echo "  [DRY RUN - would delete $to_delete images]" >&2
        return 0
    fi

    # Get oldest images and delete (tr splits tab-delimited --output text into lines)
    local digests
    digests=$(aws ecr describe-images \
        --repository-name "$repo_name" \
        --region "$ECR_REGION" \
        --query 'sort_by(imageDetails,& imagePushedAt)[*].imageDigest' \
        --output text | tr '\t' '\n' | head -n "$to_delete")

    for digest in $digests; do
        aws ecr batch-delete-image \
            --repository-name "$repo_name" \
            --region "$ECR_REGION" \
            --image-ids imageDigest="$digest" &>/dev/null || true
    done

    echo "  Cleaned up $to_delete images" >&2
}

##############################################################################
# Main Build Process
##############################################################################

check_disk_space 20

# Docker login to ECR
if [[ "$SKIP_PUSH" == "true" ]]; then
    echo "SKIP_PUSH=true: Skipping ECR login." >&2
else
    echo "" >&2
    echo "Logging in to ECR..." >&2
    aws ecr get-login-password --region "$ECR_REGION" | docker login --username AWS --password-stdin "$ECR_URL"
fi

# Build main inference image
echo "" >&2
echo "Building main inference image..." >&2
INFERENCE_IMAGE="inference:latest-local"

MAIN_CACHE_ARGS=()
if [[ "$SKIP_PUSH" != "true" ]]; then
    echo "  Pulling latest for cache..." >&2
    docker pull "$ECR_URL/inference:latest" >/dev/null 2>&1 || true
    MAIN_CACHE_ARGS=(--cache-from "$ECR_URL/inference:latest")
fi

docker buildx build --platform linux/amd64 \
    ${MAIN_CACHE_ARGS[@]+"${MAIN_CACHE_ARGS[@]}"} \
    --build-arg DATABASE_URL="${DATABASE_URL:-}" \
    -t "$INFERENCE_IMAGE" \
    -f Dockerfile .

IMAGE_DIGEST=$(docker inspect --format='{{.Id}}' "$INFERENCE_IMAGE" | cut -d: -f2 | cut -c1-12)
INFERENCE_TAG="inference:${IMAGE_DIGEST}"

cd pipelines

# Build base image with caching
echo "" >&2
echo "Building base pipeline image..." >&2

CACHE_FROM_ARGS=()
if [[ "$SKIP_PUSH" != "true" ]]; then
    docker pull "$ECR_URL/inference_pipelines/base:latest" 2>/dev/null || true
    CACHE_FROM_ARGS=(--cache-from "$ECR_URL/inference_pipelines/base:latest")
fi

BASE_IMAGE="base:latest-local"
docker buildx build --platform linux/amd64 ${CACHE_FROM_ARGS[@]+"${CACHE_FROM_ARGS[@]}"} -t "$BASE_IMAGE" -f Dockerfile.base .

BASE_DIGEST=$(docker inspect --format='{{.Id}}' "$BASE_IMAGE" | cut -d: -f2 | cut -c1-12)
BASE_TAG="inference_pipelines/base:${BASE_DIGEST}"

# Define the final base reference for sub-services
if [[ "$SKIP_PUSH" == "true" ]]; then
    INTERNAL_BASE_REF="$BASE_IMAGE"
else
    docker tag "$BASE_IMAGE" "$ECR_URL/$BASE_TAG"
    INTERNAL_BASE_REF="$ECR_URL/$BASE_TAG"
fi

# Build service images
echo "" >&2
if [[ "$PARALLEL_BUILDS" == "true" ]]; then
    echo "Building service images in parallel..." >&2
else
    echo "Building service images sequentially..." >&2
fi
BUILD_TMPDIR=$(mktemp -d)
PIDS=()

for s_data in "${SERVICES[@]}"; do
    service="${s_data%%:*}"
    dockerfile="${s_data#*:}"

    (
        echo "  Building $service..." >&2

        S_CACHE_FROM_ARGS=()
        if [[ "$SKIP_PUSH" != "true" ]]; then
            docker pull "$ECR_URL/inference_pipelines/${service}:latest" 2>/dev/null || true
            S_CACHE_FROM_ARGS=(--cache-from "$ECR_URL/inference_pipelines/${service}:latest")
        fi

        svc_img="${service}:latest-local"
        docker buildx build --platform linux/amd64 \
            ${S_CACHE_FROM_ARGS[@]+"${S_CACHE_FROM_ARGS[@]}"} \
            --build-arg BASE_IMAGE="$INTERNAL_BASE_REF" \
            -t "$svc_img" \
            -f "$dockerfile" .

        s_digest=$(docker inspect --format='{{.Id}}' "$svc_img" | cut -d: -f2 | cut -c1-12)
        echo "inference_pipelines/${service}:${s_digest}" > "$BUILD_TMPDIR/${service}.tag"
    ) &
    PIDS+=($!)

    # If sequential mode, wait for each build before starting the next
    if [[ "$PARALLEL_BUILDS" != "true" ]]; then
        wait "${PIDS[-1]}" || { echo "Error: $service build failed!" >&2; rm -rf "$BUILD_TMPDIR"; exit 1; }
    fi
done

# Wait for all parallel builds (no-op in sequential mode since we already waited)
BUILD_FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || BUILD_FAILED=1
done

if [[ $BUILD_FAILED -ne 0 ]]; then
    echo "Error: One or more service builds failed!" >&2
    rm -rf "$BUILD_TMPDIR"
    exit 1
fi

# Collect service tags
SERVICE_TAGS=()
for s_data in "${SERVICES[@]}"; do
    service="${s_data%%:*}"
    SERVICE_TAGS+=("$(cat "$BUILD_TMPDIR/${service}.tag")")
done
rm -rf "$BUILD_TMPDIR"

cd - >/dev/null

# Build tile server
echo "" >&2
echo "Building tile server..." >&2

T_CACHE_FROM_ARGS=()
if [[ "$SKIP_PUSH" != "true" ]]; then
    docker pull "$ECR_URL/tile_server/tiler:latest" 2>/dev/null || true
    T_CACHE_FROM_ARGS=(--cache-from "$ECR_URL/tile_server/tiler:latest")
fi

TILER_IMAGE="tiler:latest-local"
docker buildx build --platform linux/amd64 \
    ${T_CACHE_FROM_ARGS[@]+"${T_CACHE_FROM_ARGS[@]}"} \
    --build-arg BASE_IMAGE="$INTERNAL_BASE_REF" \
    -t "$TILER_IMAGE" \
    -f tile_server/Dockerfile .

TILER_DIGEST=$(docker inspect --format='{{.Id}}' "$TILER_IMAGE" | cut -d: -f2 | cut -c1-12)
TILER_TAG="tile_server/tiler:${TILER_DIGEST}"

##############################################################################
# Tag and Push Images
##############################################################################

if [[ "$SKIP_PUSH" == "true" ]]; then
    echo "SKIP_PUSH=true: Skipping image tagging and pushing to ECR." >&2
else
    echo "" >&2
    echo "Tagging and pushing images..." >&2

    # Tag and push inference
    docker tag "$INFERENCE_IMAGE" "$ECR_URL/$INFERENCE_TAG"
    docker tag "$INFERENCE_IMAGE" "$ECR_URL/inference:latest"
    echo "  Pushing inference..." >&2
    docker push "$ECR_URL/$INFERENCE_TAG" && PUSHED_IMAGES+=("$ECR_URL/$INFERENCE_TAG")
    docker push "$ECR_URL/inference:latest" && PUSHED_IMAGES+=("$ECR_URL/inference:latest")

    # Tag and push base
    echo "  Pushing base..." >&2
    docker push "$ECR_URL/$BASE_TAG" && PUSHED_IMAGES+=("$ECR_URL/$BASE_TAG")
    docker tag "$BASE_IMAGE" "$ECR_URL/inference_pipelines/base:latest"
    docker push "$ECR_URL/inference_pipelines/base:latest" && PUSHED_IMAGES+=("$ECR_URL/inference_pipelines/base:latest")

    # Tag and push services
    for tag in "${SERVICE_TAGS[@]}"; do
        svc_name=$(echo "$tag" | cut -d/ -f2 | cut -d: -f1)
        local_img="${svc_name}:latest-local"
        docker tag "$local_img" "$ECR_URL/$tag"
        docker tag "$local_img" "$ECR_URL/inference_pipelines/$svc_name:latest"
        echo "  Pushing $svc_name..." >&2
        docker push "$ECR_URL/$tag" && PUSHED_IMAGES+=("$ECR_URL/$tag")
        docker push "$ECR_URL/inference_pipelines/$svc_name:latest" && PUSHED_IMAGES+=("$ECR_URL/inference_pipelines/$svc_name:latest")
    done

    # Tag and push tiler
    docker tag "$TILER_IMAGE" "$ECR_URL/$TILER_TAG"
    docker tag "$TILER_IMAGE" "$ECR_URL/tile_server/tiler:latest"
    echo "  Pushing tiler..." >&2
    docker push "$ECR_URL/$TILER_TAG" && PUSHED_IMAGES+=("$ECR_URL/$TILER_TAG")
    docker push "$ECR_URL/tile_server/tiler:latest" && PUSHED_IMAGES+=("$ECR_URL/tile_server/tiler:latest")
fi

##############################################################################
# Export Terraform Variables
##############################################################################

if [[ "$SKIP_PUSH" != "true" ]]; then
    export TF_VAR_prediction_app_image_url="$ECR_URL/$INFERENCE_TAG"

    for tag in "${SERVICE_TAGS[@]}"; do
        svc_name=$(echo "$tag" | cut -d/ -f2 | cut -d: -f1)
        tf_var=$(get_tf_var "$svc_name")
        if [[ -n "$tf_var" ]]; then
            export "$tf_var=$ECR_URL/$tag"
        fi
    done

    export TF_VAR_tiler_image_url="$ECR_URL/$TILER_TAG"
fi

##############################################################################
# Cleanup
##############################################################################

echo "" >&2
echo "=========================================" >&2
echo "Cleaning up..." >&2
echo "=========================================" >&2

# Cleanup ECR (only if pushed)
if [[ "$SKIP_PUSH" != "true" ]]; then
    cleanup_ecr_repo "inference" "$MAX_IMAGES_TO_KEEP"
    cleanup_ecr_repo "inference_pipelines/base" "$MAX_IMAGES_TO_KEEP"
    for s_data in "${SERVICES[@]}"; do
        service="${s_data%%:*}"
        cleanup_ecr_repo "inference_pipelines/${service}" "$MAX_IMAGES_TO_KEEP"
    done
    cleanup_ecr_repo "tile_server/tiler" "$MAX_IMAGES_TO_KEEP"
fi

# Optional: Targeted cleanup of pushed images
if [[ "$CLEANUP_AFTER_PUSH" == "true" && ${#PUSHED_IMAGES[@]} -gt 0 ]]; then
    echo "  CLEANUP_AFTER_PUSH=true: Removing locally tagged images that were pushed..." >&2
    for img in "${PUSHED_IMAGES[@]}"; do
        if [[ $(docker ps -a --filter "ancestor=$img" --format '{{.ID}}') ]]; then
            echo "    Skipping removal of $img: containers are using it" >&2
        else
            echo "    Removing $img..." >&2
            docker rmi "$img" 2>/dev/null || echo "    Failed to remove $img" >&2
        fi
    done
fi

# Remove stale local tags from prior builds (keeps only the current build's tag per repo)
if [[ "$CLEANUP_AFTER_PUSH" == "true" && "$SKIP_PUSH" != "true" ]]; then
    echo "  Removing stale local tags from previous builds..." >&2

    # repo:keep_tag pairs
    IMAGES_TO_CLEAN=(
        "$ECR_URL/inference:${IMAGE_DIGEST}"
        "$ECR_URL/inference_pipelines/base:${BASE_DIGEST}"
        "$ECR_URL/tile_server/tiler:${TILER_DIGEST}"
    )
    for tag in "${SERVICE_TAGS[@]}"; do
        svc_repo=$(echo "$tag" | cut -d: -f1)       # e.g. inference_pipelines/floods
        svc_digest=$(echo "$tag" | cut -d: -f2)      # e.g. abc123def456
        IMAGES_TO_CLEAN+=("$ECR_URL/$svc_repo:$svc_digest")
    done

    for entry in "${IMAGES_TO_CLEAN[@]}"; do
        IMAGE_NAME="${entry%:*}"
        KEEP_TAG="${entry##*:}"
        docker images "$IMAGE_NAME" --format "{{.Tag}}" | grep -v "^${KEEP_TAG}$" | grep -v "^<none>$" | while read TAG; do
            echo "    Removing old tag: $IMAGE_NAME:$TAG" >&2
            docker rmi "$IMAGE_NAME:$TAG" 2>/dev/null || echo "    Failed to remove $IMAGE_NAME:$TAG" >&2
        done
    done
fi

# Remove local temp images (dangling prune handled by EXIT trap)
all_temp_imgs=("$INFERENCE_IMAGE" "$BASE_IMAGE" "$TILER_IMAGE")
for tag in "${SERVICE_TAGS[@]}"; do
    svc_name=$(echo "$tag" | cut -d/ -f2 | cut -d: -f1)
    all_temp_imgs+=("${svc_name}:latest-local")
done

for img in "${all_temp_imgs[@]}"; do
    docker rmi "$img" 2>/dev/null || true
done

##############################################################################
# Deploy to Kubernetes
##############################################################################

if [[ "$SKIP_DEPLOY" == "true" ]]; then
    echo "SKIP_DEPLOY=true: Skipping Kubernetes deployment steps." >&2
else
    echo "" >&2
    echo "=========================================" >&2
    echo "Deploying to Kubernetes..." >&2
    echo "=========================================" >&2

    envsubst < services-helm/configMap.yaml.tmpl > services-helm/configMap.yaml
    kubectl apply -f services-helm/configMap.yaml
fi

echo "" >&2
echo "=========================================" >&2
echo "Deployment completed successfully!" >&2
echo "=========================================" >&2
echo "Images deployed:" >&2
echo "  Inference: $INFERENCE_TAG" >&2
echo "  Base: $BASE_TAG" >&2
for tag in "${SERVICE_TAGS[@]}"; do
    echo "  Service: $tag" >&2
done
echo "  Tiler: $TILER_TAG" >&2
echo "" >&2
