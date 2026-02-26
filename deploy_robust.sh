#!/bin/bash

set -e  # Exit on any error

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

# Tracking for cleanup
PUSHED_IMAGES=()

# Ensure required environment variables are set (only if we aren't doing a local-only run)
if [[ "$SKIP_PUSH" != "true" || "$SKIP_DEPLOY" != "true" ]]; then
    if [[ -z "$ECR_URL" ]]; then
        echo "Error: ECR_URL environment variable must be set (e.g. 123456789.dkr.ecr.us-west-2.amazonaws.com)" >&2
        exit 1
    fi
    if [[ -z "$INGRESS_HOST" ]]; then
        echo "Error: INGRESS_HOST environment variables must be set" >&2
        exit 1
    fi
fi

export TF_VAR_fm_services_release_version=$(git describe --tags --exact-match 2>/dev/null || git branch --show-current || git rev-parse --short HEAD)

echo "=========================================" >&2
echo "Docker Build & Deploy" >&2
echo "=========================================" >&2
echo "ECR URL: $ECR_URL" >&2
echo "Region: $ECR_REGION" >&2
echo "Keep Images: $MAX_IMAGES_TO_KEEP per repo" >&2
echo "=========================================" >&2

##############################################################################
# Helper Functions
##############################################################################

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
    local total=$(aws ecr describe-images \
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
    
    # Get oldest images and delete
    local digests=$(aws ecr describe-images \
        --repository-name "$repo_name" \
        --region "$ECR_REGION" \
        --query 'sort_by(imageDetails,& imagePushedAt)[*].imageDigest' \
        --output text | head -n $to_delete)
    
    for digest in $digests; do
        aws ecr batch-delete-image \
            --repository-name "$repo_name" \
            --region "$ECR_REGION" \
            --image-ids imageDigest="$digest" &>/dev/null || true
    done
    
    echo "  ✓ Cleaned up $to_delete images" >&2
}

##############################################################################
# Main Build Process
##############################################################################

# Docker login to ECR
if [[ "$SKIP_PUSH" == "true" ]]; then
    echo "SKIP_PUSH=true: Skipping ECR login." >&2
else
    echo "" >&2
    echo "Logging in to ECR..." >&2
    aws ecr get-login-password --region "$ECR_REGION" | docker login --username AWS --password-stdin $ECR_URL
fi

# Build main inference image
echo "" >&2
echo "Building main inference image..." >&2
INFERENCE_IMAGE="inference:latest-local"

if [[ "$SKIP_PUSH" != "true" ]]; then
    echo "  Pulling latest for cache..." >&2
    docker pull $ECR_URL/inference:latest >/dev/null 2>&1 || true
    MAIN_CACHE="--cache-from $ECR_URL/inference:latest"
else
    MAIN_CACHE=""
fi

docker build \
    $MAIN_CACHE \
    --build-arg DATABASE_URL=$DATABASE_URL \
    -t $INFERENCE_IMAGE \
    -f Dockerfile.multistage .

IMAGE_DIGEST=$(docker inspect --format='{{.Id}}' $INFERENCE_IMAGE | cut -d: -f2 | cut -c1-12)
INFERENCE_TAG="inference:${IMAGE_DIGEST}"

cd pipelines

# Build base image with caching
echo "" >&2
echo "Building base pipeline image..." >&2
if [[ "$SKIP_PUSH" != "true" ]]; then
    docker pull $ECR_URL/inference_pipelines/base:latest 2>/dev/null || true
    CACHE_FROM="--cache-from $ECR_URL/inference_pipelines/base:latest"
else
    CACHE_FROM=""
fi

BASE_IMAGE="base:latest-local"
docker build $CACHE_FROM -t $BASE_IMAGE -f Dockerfile.base .

BASE_DIGEST=$(docker inspect --format='{{.Id}}' $BASE_IMAGE | cut -d: -f2 | cut -c1-12)
BASE_TAG="inference_pipelines/base:${BASE_DIGEST}"

# Define the final base reference for sub-services
if [[ "$SKIP_PUSH" == "true" ]]; then
    INTERNAL_BASE_REF="$BASE_IMAGE"
else
    docker tag "$BASE_IMAGE" "$ECR_URL/$BASE_TAG"
    INTERNAL_BASE_REF="$ECR_URL/$BASE_TAG"
fi

# Build service images
SERVICES="floods:floods/Dockerfile burn_scars:burn_scars/Dockerfile crop_classification:crop_classification/Dockerfile surya_rollout:surya/Dockerfile"

SERVICE_TAGS_LIST=""

for s_data in $SERVICES; do
    service=$(echo $s_data | cut -d: -f1)
    dockerfile=$(echo $s_data | cut -d: -f2)
    
    echo "" >&2
    echo "Building $service..." >&2
    
    if [[ "$SKIP_PUSH" != "true" ]]; then
        docker pull $ECR_URL/inference_pipelines/${service}:latest 2>/dev/null || true
        S_CACHE_FROM="--cache-from $ECR_URL/inference_pipelines/${service}:latest"
    else
        S_CACHE_FROM=""
    fi

    local svc_img="${service}:latest-local"
    docker build \
        $S_CACHE_FROM \
        --build-arg BASE_IMAGE=$INTERNAL_BASE_REF \
        -t $svc_img \
        -f $dockerfile .
    
    s_digest=$(docker inspect --format='{{.Id}}' $svc_img | cut -d: -f2 | cut -c1-12)
    s_tag="inference_pipelines/${service}:${s_digest}"
    SERVICE_TAGS_LIST="$SERVICE_TAGS_LIST $s_tag"
done

cd -

# Build tile server
echo "" >&2
echo "Building tile server..." >&2
if [[ "$SKIP_PUSH" != "true" ]]; then
    docker pull $ECR_URL/tile_server/tiler:latest 2>/dev/null || true
    T_CACHE_FROM="--cache-from $ECR_URL/tile_server/tiler:latest"
else
    T_CACHE_FROM=""
fi

TILER_IMAGE="tiler:latest-local"
docker build \
    $T_CACHE_FROM \
    --build-arg BASE_IMAGE=$INTERNAL_BASE_REF \
    -t $TILER_IMAGE \
    -f tile_server/Dockerfile .

TILER_DIGEST=$(docker inspect --format='{{.Id}}' $TILER_IMAGE | cut -d: -f2 | cut -c1-12)
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
    docker tag $INFERENCE_IMAGE $ECR_URL/$INFERENCE_TAG
    docker tag $INFERENCE_IMAGE $ECR_URL/inference:latest
    echo "  Pushing inference..." >&2
    docker push $ECR_URL/$INFERENCE_TAG && PUSHED_IMAGES+=("$ECR_URL/$INFERENCE_TAG")
    docker push $ECR_URL/inference:latest && PUSHED_IMAGES+=("$ECR_URL/inference:latest")

    # Tag and push base
    echo "  Pushing base..." >&2
    docker push $ECR_URL/$BASE_TAG && PUSHED_IMAGES+=("$ECR_URL/$BASE_TAG")
    docker tag $BASE_IMAGE $ECR_URL/inference_pipelines/base:latest
    docker push $ECR_URL/inference_pipelines/base:latest && PUSHED_IMAGES+=("$ECR_URL/inference_pipelines/base:latest")

    # Tag and push services
    tags=($SERVICE_TAGS_LIST)
    for tag in "${tags[@]}"; do
        svc_name=$(echo $tag | cut -d/ -f2 | cut -d: -f1)
        local_img="${svc_name}:latest-local"
        docker tag $local_img $ECR_URL/$tag
        docker tag $local_img $ECR_URL/inference_pipelines/$svc_name:latest
        echo "  Pushing $svc_name..." >&2
        docker push $ECR_URL/$tag && PUSHED_IMAGES+=("$ECR_URL/$tag")
        docker push $ECR_URL/inference_pipelines/$svc_name:latest && PUSHED_IMAGES+=("$ECR_URL/inference_pipelines/$svc_name:latest")
    done

    # Tag and push tiler
    docker tag $TILER_IMAGE $ECR_URL/$TILER_TAG
    docker tag $TILER_IMAGE $ECR_URL/tile_server/tiler:latest
    echo "  Pushing tiler..." >&2
    docker push $ECR_URL/$TILER_TAG && PUSHED_IMAGES+=("$ECR_URL/$TILER_TAG")
    docker push $ECR_URL/tile_server/tiler:latest && PUSHED_IMAGES+=("$ECR_URL/tile_server/tiler:latest")
fi

##############################################################################
# Export Terraform Variables
##############################################################################

if [[ "$SKIP_PUSH" != "true" ]]; then
    export TF_VAR_prediction_app_image_url=$ECR_URL/$INFERENCE_TAG
    
    tags=($SERVICE_TAGS_LIST)
    for tag in "${tags[@]}"; do
        if [[ $tag == *"floods"* ]]; then export TF_VAR_floods_app_image_url=$ECR_URL/$tag; fi
        if [[ $tag == *"burn_scars"* ]]; then export TF_VAR_burnScar_app_image_url=$ECR_URL/$tag; fi
        if [[ $tag == *"crop_classification"* ]]; then export TF_VAR_crop_app_image_url=$ECR_URL/$tag; fi
        if [[ $tag == *"surya_rollout"* ]]; then export TF_VAR_surya_rollout_image_url=$ECR_URL/$tag; fi
    done
    export TF_VAR_tiler_image_url=$ECR_URL/$TILER_TAG
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
    cleanup_ecr_repo "inference" $MAX_IMAGES_TO_KEEP
    cleanup_ecr_repo "inference_pipelines/base" $MAX_IMAGES_TO_KEEP
    cleanup_ecr_repo "inference_pipelines/floods" $MAX_IMAGES_TO_KEEP
    cleanup_ecr_repo "inference_pipelines/burn_scars" $MAX_IMAGES_TO_KEEP
    cleanup_ecr_repo "inference_pipelines/crop_classification" $MAX_IMAGES_TO_KEEP
    cleanup_ecr_repo "inference_pipelines/surya_rollout" $MAX_IMAGES_TO_KEEP
    cleanup_ecr_repo "tile_server/tiler" $MAX_IMAGES_TO_KEEP
fi

# Optional: Targeted cleanup of pushed images
if [[ "$CLEANUP_AFTER_PUSH" == "true" ]]; then
    echo "  CLEANUP_AFTER_PUSH=true: Removing locally tagged images that were pushed..." >&2
    for img in "${PUSHED_IMAGES[@]}"; do
        if [[ $(docker ps -a --filter "ancestor=$img" --format '{{.ID}}') ]]; then
            echo "    ⚠ Skipping removal of $img: containers are using it" >&2
        else
            echo "    Removing $img..." >&2
            docker rmi "$img" 2>/dev/null || echo "    ⚠ Failed to remove $img" >&2
        fi
    done
fi

# Always prune dangling layers and old build cache
docker image prune -f >/dev/null
docker builder prune -f --filter "until=24h" >/dev/null

# Remove local temp images
all_temp_imgs="$INFERENCE_IMAGE $BASE_IMAGE $TILER_IMAGE"
tags=($SERVICE_TAGS_LIST)
for tag in "${tags[@]}"; do
    svc_name=$(echo $tag | cut -d/ -f2 | cut -d: -f1)
    all_temp_imgs="$all_temp_imgs ${svc_name}:latest-local"
done

for img in $all_temp_imgs; do
    docker rmi $img 2>/dev/null || true
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
echo "✓ Deployment completed successfully!" >&2
echo "=========================================" >&2
echo "Images deployed:" >&2
echo "  Inference: $INFERENCE_TAG" >&2
echo "  Base: $BASE_TAG" >&2
tags=($SERVICE_TAGS_LIST)
for tag in "${tags[@]}"; do
    echo "  Service: $tag" >&2
done
echo "  Tiler: $TILER_TAG" >&2
echo "" >&2