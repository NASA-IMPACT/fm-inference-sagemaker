#! /bin/bash

export ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"

export ECR_IMAGE_NAME=prediction:latest

docker buildx build --platform linux/amd64 -t $ECR_URL/$ECR_IMAGE_NAME .


aws ecr get-login-password --region us-west-2 | docker login --password-stdin --username AWS $ECR_URL

# docker push $ECR_URL/fm_inference:latest

kind load docker-image $ECR_URL/$ECR_IMAGE_NAME --name neo-cluster
