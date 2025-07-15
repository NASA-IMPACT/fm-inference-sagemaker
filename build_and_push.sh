%%sh
docker build . -fu Dockerfile.inference --platform linux/amd64 -t fm_inference

export ECR_URL="$AWS_ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com"

aws ecr get-login-password --region us-west-2 --profile summer| \
  docker login --password-stdin --username AWS $ECR_URL

docker tag fm_inference $ECR_URL/fm_inference:latest

docker push $ECR_URL/fm_inference:latest
