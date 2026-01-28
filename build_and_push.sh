#!/bin/bash

set -e

# --- CONFIGURATION ---
MAX_IMAGES=5
CACHE_RETENTION="168h" # 7 days
# ---------------------

if [[ -z "$ECR_URL" || -z "$INGRESS_HOST" ]]; then
    echo "Error: ECR_URL and INGRESS_HOST environment variables must be set"
    exit 1
fi

# Set version tag (used by Terraform/Helm)
export TF_VAR_fm_services_release_version=$(git describe --tags --exact-match 2>/dev/null || git branch --show-current || git rev-parse --short HEAD)
echo "Release Version: $TF_VAR_fm_services_release_version"

# Initialize Docker Buildx (Persistent Builder)
if ! docker buildx inspect prediction-builder > /dev/null 2>&1; then
    docker buildx create --name prediction-builder --use
else
    docker buildx use prediction-builder
fi

# Login to ECR (Once for all builds)
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_URL

# --- REUSABLE BUILD FUNCTION ---
# Args: 1=RepoName, 2=Context, 3=Dockerfile, 4=BuildArgs, 5=ExportVarName
build_and_deploy() {
    local REPO_NAME="$1"
    local CONTEXT="$2"
    local FILE="$3"
    local ARGS="$4"
    local VAR_NAME="$5"
    local TEMP_TAG="$REPO_NAME:temp"
    
    echo "--- Processing $REPO_NAME ---"
    
    # 1. Build (Persistent Cache automatically used)
    # We use eval to expand the build args string properly
    docker buildx build \
      --platform linux/amd64 \
      -t $TEMP_TAG \
      --load \
      -f $FILE \
      $ARGS \
      $CONTEXT

    # 2. Generate Hash & Tag
    local DIGEST=$(docker inspect --format='{{.Id}}' $TEMP_TAG | cut -d: -f2 | cut -c1-12)
    local FULL_URL="$ECR_URL/$REPO_NAME:$DIGEST"

    # 3. Push
    docker tag $TEMP_TAG $FULL_URL
    docker push $FULL_URL

    # 4. Export the variable for envsubst
    if [ -n "$VAR_NAME" ]; then
        export $VAR_NAME=$FULL_URL
        echo "Exported $VAR_NAME=$FULL_URL"
    fi

    # 5. Cleanup ECR (Keep last 5)
    # We silence the output to keep the logs clean, only reporting errors
    local TO_DELETE=$(aws ecr describe-images --repository-name $REPO_NAME --query "imageDetails[? not_null(imageTags)].{digest: imageDigest, date: imagePushedAt}" --output json | jq -c "sort_by(.date) | .[:-${MAX_IMAGES}] | .[].digest")
    
    if [ -n "$TO_DELETE" ]; then
        for digest in $TO_DELETE; do
            clean_digest=$(echo $digest | tr -d '"')
            aws ecr batch-delete-image --repository-name $REPO_NAME --image-ids imageDigest=$clean_digest > /dev/null
        done
        echo "Cleaned up old ECR images."
    fi

    # 6. Cleanup Local
    docker rmi $TEMP_TAG $FULL_URL 2>/dev/null || true
}

# --- STEP 1: Build Base Image (Dependency) ---
echo ">>> Building Base Image..."
build_and_deploy "inference_pipelines/base" "pipelines" "pipelines/Dockerfile.base" "" "ECR_BASE_IMAGE_NAME"

# --- STEP 2: Build Dependent Services ---
# These use the Base Image we just built
BASE_ARG="--build-arg BASE_IMAGE=$ECR_BASE_IMAGE_NAME"

echo ">>> Building Dependent Services..."
build_and_deploy "inference_pipelines/floods" "pipelines" "pipelines/floods/Dockerfile" "$BASE_ARG" "TF_VAR_floods_app_image_url"
build_and_deploy "inference_pipelines/burn_scars" "pipelines" "pipelines/burn_scars/Dockerfile" "$BASE_ARG" "TF_VAR_burnScar_app_image_url"
build_and_deploy "inference_pipelines/crop_classification" "pipelines" "pipelines/crop_classification/Dockerfile" "$BASE_ARG" "TF_VAR_crop_app_image_url"
build_and_deploy "inference_pipelines/surya_rollout" "pipelines" "pipelines/surya/Dockerfile" "$BASE_ARG" "TF_VAR_surya_rollout_image_url"
build_and_deploy "tile_server/tiler" "." "tile_server/Dockerfile" "$BASE_ARG" "TF_VAR_tiler_image_url"

# --- STEP 3: Build Main Inference App ---
echo ">>> Building Inference App..."
# Note: DATABASE_URL is passed here as requested
build_and_deploy "inference" "." "Dockerfile" "--build-arg DATABASE_URL=$DATABASE_URL" "TF_VAR_prediction_app_image_url"

# --- STEP 4: Global Cleanup ---
echo ">>> Performing Final System Prune..."
docker image prune -f
docker buildx prune -f --filter "until=$CACHE_RETENTION"

# --- STEP 5: Deploy ---
echo ">>> Generating Manifests..."
# Generate ConfigMap from template
envsubst < services-helm/configMap.yaml.tmpl > services-helm/configMap.yaml

# DEBUG: Print the generated ConfigMap to verify variables were substituted
echo "--- DEBUG: Generated ConfigMap Preview ---"
cat services-helm/configMap.yaml
echo "------------------------------------------"

# Apply Kubernetes manifests
kubectl apply -f services-helm/configMap.yaml

echo "Deployment Complete."