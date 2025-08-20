#!/bin/bash

# Build the image first to get the digest
TEMP_IMAGE_NAME="prediction:temp"
echo "Building temporary image to get digest: $TEMP_IMAGE_NAME"
docker buildx build --platform linux/amd64 -t $TEMP_IMAGE_NAME .

# Get the image digest (content-based hash)
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' $TEMP_IMAGE_NAME 2>/dev/null || docker inspect --format='{{.Id}}' $TEMP_IMAGE_NAME | cut -d: -f2 | cut -c1-12)

# Create final tag using just the digest
export IMAGE_TAG="${IMAGE_DIGEST}"
export ECR_IMAGE_NAME="prediction:${IMAGE_TAG}"

# Tag the temp image with final name
docker tag $TEMP_IMAGE_NAME $ECR_URL/$ECR_IMAGE_NAME

echo "Final image: $ECR_URL/$ECR_IMAGE_NAME"
echo "Using ingress host: $INGRESS_HOST"

# Push to ECR
ECR_PASSWORD=$(aws ecr get-login-password --region us-west-2)
echo $ECR_PASSWORD | docker login --username AWS --password-stdin $ECR_URL
docker push $ECR_URL/$ECR_IMAGE_NAME

# Clean up temporary image
docker rmi $TEMP_IMAGE_NAME

# Generate deployment.yaml and ingress.yaml from templates using envsubst
envsubst < k8s-manifests/deployment.yaml.tmpl > k8s-manifests/deployment.yaml
envsubst < k8s-manifests/ingress.yaml.tmpl > k8s-manifests/ingress.yaml

# Apply Kubernetes manifests
kubectl apply -f k8s-manifests/

# Optional: Load image to kind cluster if needed
# kind load docker-image $ECR_URL/$ECR_IMAGE_NAME --name neo-cluster
