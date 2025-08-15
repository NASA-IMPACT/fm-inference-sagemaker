#! /bin/bash

#export ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"
#
#export ECR_IMAGE_NAME=prediction:latest

docker buildx build --platform linux/amd64 -t $ECR_URL/$ECR_IMAGE_NAME .

ECR_PASSWORD=$(aws ecr get-login-password --region us-west-2)
echo $ECR_PASSWORD | docker login --username AWS --password-stdin $ECR_URL

docker push $ECR_URL/$ECR_IMAGE_NAME

kubectl apply -f k8s-manifests/
# kind load docker-image $ECR_URL/$ECR_IMAGE_NAME --name neo-cluster
