#!/bin/bash

# Generate dynamic tag from branch name and git commit
BRANCH_NAME=$(git branch --show-current | sed 's/[^a-zA-Z0-9]/-/g')  # Replace special chars with hyphens
GIT_COMMIT=$(git rev-parse --short HEAD)
export IMAGE_TAG="${BRANCH_NAME}-${GIT_COMMIT}"
export ECR_IMAGE_NAME="prediction:${IMAGE_TAG}"

echo "Building and pushing image: $ECR_URL/$ECR_IMAGE_NAME"

# Build and push Docker image
docker buildx build --platform linux/amd64 -t $ECR_URL/$ECR_IMAGE_NAME .
ECR_PASSWORD=$(aws ecr get-login-password --region us-west-2)
echo $ECR_PASSWORD | docker login --username AWS --password-stdin $ECR_URL
docker push $ECR_URL/$ECR_IMAGE_NAME

# Generate deployment.yaml from template using envsubst
envsubst < k8s-manifests/deployment.template.yaml > k8s-manifests/deployment.yaml

# Apply Kubernetes manifests
kubectl apply -f k8s-manifests/

# Optional: Load image to kind cluster if needed
# kind load docker-image $ECR_URL/$ECR_IMAGE_NAME --name neo-cluster
