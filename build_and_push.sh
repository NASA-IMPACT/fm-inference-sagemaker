#!/bin/bash

# Ensure required environment variables are set
if [[ -z "$ECR_URL" || -z "$INGRESS_HOST" ]]; then
    echo "Error: ECR_URL and INGRESS_HOST environment variables must be set"
    exit 1
fi

export TF_VAR_fm_services_release_version=$(git describe --tags --exact-match 2>/dev/null || git branch --show-current || git rev-parse --short HEAD)

# Rest of the code remains the same...



# Build the image first to get the digest
TEMP_IMAGE_NAME="inference:temp"
echo "Building temporary image to get digest: $TEMP_IMAGE_NAME"
# Run the migration during the build
docker build --build-arg DATABASE_URL=$DATABASE_URL -t $TEMP_IMAGE_NAME .
cd pipelines

# Docker login to ECR
ECR_PASSWORD=$(aws ecr get-login-password --region us-west-2)
echo $ECR_PASSWORD | docker login --username AWS --password-stdin $ECR_URL

# Build and push base image first
BASE_IMAGE_NAME="inference_pipelines:temp"
echo "Building temporary image to get digest: $BASE_IMAGE_NAME"
docker build -t $BASE_IMAGE_NAME . -f Dockerfile.base
BASE_DIGEST=$(docker inspect --format='{{.Id}}' $BASE_IMAGE_NAME | cut -d: -f2 | cut -c1-12)
export ECR_BASE_IMAGE_NAME="inference_pipelines/base:${BASE_DIGEST}"
docker tag $BASE_IMAGE_NAME $ECR_URL/$ECR_BASE_IMAGE_NAME
docker push $ECR_URL/$ECR_BASE_IMAGE_NAME

# Build floods and burn scars images
TEMP_FLOOD_IMAGE_NAME="floods:temp"
echo "Building temporary image to get digest: $TEMP_FLOOD_IMAGE_NAME"
docker build -t $TEMP_FLOOD_IMAGE_NAME . -f floods/Dockerfile --build-arg BASE_IMAGE=$ECR_URL/$ECR_BASE_IMAGE_NAME

TEMP_BURN_IMAGE_NAME="burn_scars:temp"
echo "Building temporary image to get digest: $TEMP_BURN_IMAGE_NAME"
docker build -t $TEMP_BURN_IMAGE_NAME . -f burn_scars/Dockerfile --build-arg BASE_IMAGE=$ECR_URL/$ECR_BASE_IMAGE_NAME

TEMP_CROP_IMAGE_NAME="crop_classification:temp"
echo "Building temporary image to get digest: $TEMP_CROP_IMAGE_NAME"
docker build -t $TEMP_CROP_IMAGE_NAME . -f crop_classification/Dockerfile --build-arg BASE_IMAGE=$ECR_URL/$ECR_BASE_IMAGE_NAME

TEMP_SURYA_ROLLOUT_IMAGE_NAME="surya_rollout:temp"
echo "Building temporary image to get digest: $TEMP_SURYA_ROLLOUT_IMAGE_NAME"
docker build -t $TEMP_SURYA_ROLLOUT_IMAGE_NAME . -f surya/Dockerfile --build-arg BASE_IMAGE=$ECR_URL/$ECR_BASE_IMAGE_NAME

GO_SERVICES_TAG=$(uuidgen)
GO_SERVICES_APP="inference_pipelines/go_services:${GO_SERVICES_TAG}"
# Build Go services
docker build -t $ECR_URL/$GO_SERVICES_APP -f go-service/Dockerfile .


cd -

TILER_IMAGE_NAME="tile_server:temp"
echo "Building temporary image to get digest: $TILER_IMAGE_NAME"
docker build -t $TILER_IMAGE_NAME . -f tile_server/Dockerfile --build-arg BASE_IMAGE=$ECR_URL/$ECR_BASE_IMAGE_NAME

TILER_DIGEST=$(docker inspect --format='{{.Id}}' $TILER_IMAGE_NAME | cut -d: -f2 | cut -c1-12)
# Get the image digest (content-based hash) - extract only the hash portion
IMAGE_DIGEST=$(docker inspect --format='{{.Id}}' $TEMP_IMAGE_NAME | cut -d: -f2 | cut -c1-12)
# Get the flood image digest (content-based hash) - extract only the hash portion
FLOOD_IMAGE_DIGEST=$(docker inspect --format='{{.Id}}' $TEMP_FLOOD_IMAGE_NAME | cut -d: -f2 | cut -c1-12)
BURN_IMAGE_DIGEST=$(docker inspect --format='{{.Id}}' $TEMP_BURN_IMAGE_NAME | cut -d: -f2 | cut -c1-12)
CROP_IMAGE_DIGEST=$(docker inspect --format='{{.Id}}' $TEMP_CROP_IMAGE_NAME | cut -d: -f2 | cut -c1-12)
SURYA_IMAGE_DIGEST=$(docker inspect --format='{{.Id}}' $TEMP_SURYA_ROLLOUT_IMAGE_NAME | cut -d: -f2 | cut -c1-12)

# Create final tag using just the short hash (no colons or special characters)
IMAGE_TAG="${IMAGE_DIGEST}"
PREDICTION_APP="inference:${IMAGE_TAG}"
FLOODS_APP="inference_pipelines/floods:${FLOOD_IMAGE_DIGEST}"
BURN_SCAR_APP="inference_pipelines/burn_scars:${BURN_IMAGE_DIGEST}"
CROP_APP="inference_pipelines/crop_classification:${CROP_IMAGE_DIGEST}"
SURYA_ROLLOUT_APP="inference_pipelines/surya_rollout:${SURYA_IMAGE_DIGEST}"


export ECR_TILER_IMAGE_NAME="tile_server/tiler:${TILER_DIGEST}"

# Tag the temp image with final name
docker tag $TEMP_IMAGE_NAME $ECR_URL/$PREDICTION_APP
docker tag $TEMP_FLOOD_IMAGE_NAME $ECR_URL/$FLOODS_APP
docker tag $TEMP_BURN_IMAGE_NAME $ECR_URL/$BURN_SCAR_APP
docker tag $TEMP_CROP_IMAGE_NAME $ECR_URL/$CROP_APP
docker tag $TEMP_SURYA_ROLLOUT_IMAGE_NAME $ECR_URL/$SURYA_ROLLOUT_APP
docker tag $TILER_IMAGE_NAME $ECR_URL/$ECR_TILER_IMAGE_NAME



echo "Final image: $ECR_URL/$PREDICTION_APP"
echo "Using ingress host: $INGRESS_HOST"

export TF_VAR_prediction_app_image_url=$ECR_URL/$PREDICTION_APP
export TF_VAR_floods_app_image_url=$ECR_URL/$FLOODS_APP
export TF_VAR_burnScar_app_image_url=$ECR_URL/$BURN_SCAR_APP
export TF_VAR_crop_app_image_url=$ECR_URL/$CROP_APP
export TF_VAR_surya_rollout_image_url=$ECR_URL/$SURYA_ROLLOUT_APP
export TF_VAR_tiler_image_url=$ECR_URL/$ECR_TILER_IMAGE_NAME
export TF_VAR_go_services_image_url=$ECR_URL/$GO_SERVICES_APP
# Push to ECR
docker push $TF_VAR_prediction_app_image_url
docker push $TF_VAR_floods_app_image_url
docker push $TF_VAR_burnScar_app_image_url
docker push $TF_VAR_crop_app_image_url
docker push $TF_VAR_surya_rollout_image_url
docker push $TF_VAR_tiler_image_url
docker push $TF_VAR_go_services_image_url

# Clean up temporary image
docker rmi $TEMP_IMAGE_NAME
docker rmi $BASE_IMAGE_NAME
docker rmi $TILER_IMAGE_NAME
docker rmi $TEMP_FLOOD_IMAGE_NAME
docker rmi $TEMP_BURN_IMAGE_NAME
docker rmi $TEMP_CROP_IMAGE_NAME
docker rmi $TEMP_SURYA_ROLLOUT_IMAGE_NAME

# Generate deployment.yaml and ingress.yaml from templates using envsubst
envsubst < services-helm/configMap.yaml.tmpl > services-helm/configMap.yaml

# Apply Kubernetes manifests
kubectl apply -f services-helm/configMap.yaml

# Optional: Load image to kind cluster if needed
# kind load docker-image $ECR_URL/$PREDICTION_APP --name neo-cluster

# Clean up images
declare -A IMAGES_TO_CLEAN=(
    ["$ECR_URL/inference"]="${IMAGE_TAG}"
    ["$ECR_URL/inference_pipelines/floods"]="${FLOOD_IMAGE_DIGEST}"
    ["$ECR_URL/inference_pipelines/burn_scars"]="${BURN_IMAGE_DIGEST}"
    ["$ECR_URL/inference_pipelines/crop_classification"]="${CROP_IMAGE_DIGEST}"
    ["$ECR_URL/inference_pipelines/surya_rollout"]="${SURYA_IMAGE_DIGEST}"
    ["$ECR_URL/tile_server/tiler"]="${TILER_DIGEST}"
    ["$ECR_URL/inference_pipelines/base"]="${BASE_DIGEST}"
    ["$ECR_URL/inference_pipelines/go_services"]="${GO_SERVICES_TAG}"
)

for IMAGE_NAME in "${!IMAGES_TO_CLEAN[@]}"; do
    KEEP_TAG="${IMAGES_TO_CLEAN[$IMAGE_NAME]}"
    docker images "$IMAGE_NAME" --format "{{.Tag}}" | grep -v "^${KEEP_TAG}$" | grep -v "^<none>$" | while read TAG; do
        echo "Removing old tag: $IMAGE_NAME:$TAG"
        docker rmi "$IMAGE_NAME:$TAG" 2>/dev/null || true
    done
done
